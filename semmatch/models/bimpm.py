import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn


#Bilateral Multi-Perspective Matching for Natural Language Sentences
#https://arxiv.org/abs/1702.03814
@register.register_subclass('model', 'bimpm')
class BIMPM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 100, dropout_rate: float = 0.1, char_hidden_size: int = 50, num_perspective: int = 20,
                 model_name: str = 'bilstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate
        self._char_hidden_size = char_hidden_size
        self._num_perspective = num_perspective

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            premise_chars = features_embedding.get('premise/chars', None)
            hypothesis_chars = features_embedding.get('hypothesis/chars', None)

            # ----- Matching Layer -----
            # with tf.variable_scope("weight", reuse = tf.AUTO_REUSE):
            mp_ws = []
            for i in range(1, 9):
                mp_ws.append(tf.get_variable("mp_w{}".format(i),
                                        shape=[self._num_perspective, self._hidden_dim],
                                        initializer=tf.initializers.he_normal()
                                        )
                        )

            def cosine_similarity(lfs, rhs):
                """
                :params lfs: [...,d]
                :params hfs: [...,d]
                :return [...]
                """
                dot = tf.reduce_sum(lfs * rhs, axis=-1)
                base = tf.sqrt(tf.reduce_sum(tf.square(lfs), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.square(rhs), axis=-1))
                # return dot / base
                return div_with_small_value(dot, base, eps=1e-8)

            def div_with_small_value(dot, base, eps=1e-8):
                # too small values are replaced by 1e-8 to prevent it from exploding.
                eps = tf.ones_like(base) * eps
                base = tf.where(base > eps, base, eps)
                return dot / base

            def mp_matching_func(v1, v2, w):
                """
                :params v1: (batch, seq_len, hidden_size)
                :params v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
                :param w: (l, hidden_size)
                :return: (batch, l)
                """
                v1 = tf.expand_dims(v1, axis=2) * w  # (batch, seq_len, l, hidden_size)

                if v2.shape.ndims == 3:  # (batch, seq_len, hidden_size)
                    v2 = tf.expand_dims(v2, axis=2) * w  # (batch, seq_len, 1, hidden_size)
                else:
                    v2 = tf.expand_dims(tf.expand_dims(v2, axis=1), axis=1) * w  # (batch, 1, 1, hidden_size)

                m = cosine_similarity(v1, v2)
                return m

            def mp_matching_func_pairwise(v1, v2, w):
                """
                :params v1: (batch, seq_len1, hidden_size)
                :params v2: (batch, seq_len2, hidden_size)
                :params w: (l, hidden_size)
                :return: (batch, l, seq_len1, seq_len2)
                """
                w = tf.expand_dims(w, axis=1)  # (l, 1, hidden_size)
                v1 = tf.expand_dims(v1, axis=1)  # (batch, 1, seq_len, hidden_size)
                v2 = tf.expand_dims(v2, axis=1)
                v1 = v1 * w  # (batch, l, seq_len, hidden_size)
                v2 = v2 * w

                v1_norm = tf.norm(v1, axis=3, keepdims=True)  # (batch, l, seq_len, 1)
                v2_norm = tf.norm(v2, axis=3, keepdims=True)

                n = tf.matmul(v1, v2, transpose_b=True)  # (batch, l, seq_len1, seq_len2)
                d = v1_norm * tf.transpose(v2_norm, [0, 1, 3, 2])  # (batch, l, seq_len1, 1) * (batch, l, 1, seq_len2)

                m = div_with_small_value(n, d)  # (batch, l, seq_len1, seq_len2)
                m = tf.transpose(m, [0, 2, 3, 1])

                return m

            def attention(v1, v2):
                """
                :param v1: (batch, seq_len1, hidden_size)
                :param v2: (batch, seq_len2, hidden_size)
                :return: (batch, seq_len1, seq_len2)
                """
                v1_norm = tf.norm(v1, axis=2, keepdims=True)
                v2_norm = tf.norm(v2, axis=2, keepdims=True)

                # (batch, seq_len1, seq_len2)
                a = tf.matmul(v1_norm, v2_norm, transpose_b=True)
                d = v1_norm * tf.transpose(v2_norm, [0, 2, 1])

                return div_with_small_value(a, d)

            def get_char_emb(chars, name="char_emb"):
                with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
                    seq_len_p = tf.shape(chars)[1]
                    max_word_len = chars.get_shape()[2]
                    char_dim = chars.get_shape()[-1]

                    chars = tf.reshape(chars, [-1, max_word_len, char_dim])

                    _, (chars, _) = nn.lstm(chars, self._char_hidden_size)

                    # (batch, seq_len, char_hidden_size)
                    chars = tf.reshape(chars, [-1, seq_len_p, self._char_hidden_size])

                    return chars

            # ----- Word Representation Layer -----
            # (batch, seq_len) -> (batch, seq_len, word_dim)
            p = premise_tokens
            h = hypothesis_tokens

            if premise_chars is not None and hypothesis_chars is not None:
                # (batch, seq_len, word_len) -> (batch * seq_len, max_word_len)
                char_p = get_char_emb(premise_chars)
                char_h = get_char_emb(hypothesis_chars)
                # (batch, seq_len, word_dim + char_hidden_size)
                p = tf.concat([p, char_p], axis=-1)
                h = tf.concat([h, char_h], axis=-1)

            p = tf.layers.dropout(p, self._dropout_rate, training=is_training)
            h = tf.layers.dropout(h, self._dropout_rate, training=is_training)

            # ----- Context Representation Layer -----
            with tf.variable_scope("context_representation", reuse=tf.AUTO_REUSE):
                # (batch, seq_len, hidden_size * 2)
                con_p, _ = nn.bi_lstm(p, self._hidden_dim, name="context_representation_bilstm")
                con_p = tf.concat(con_p, axis=-1)

                con_h, _ = nn.bi_lstm(h, self._hidden_dim, name="context_representation_bilstm")
                con_h = tf.concat(con_h, axis=-1)

            con_p = tf.layers.dropout(con_p, self._dropout_rate, training=is_training)
            con_h = tf.layers.dropout(con_h, self._dropout_rate, training=is_training)

            # (batch, seq_len, hidden_size)
            con_p_fw, con_p_bw = tf.split(con_p, num_or_size_splits=2, axis=-1)
            con_h_fw, con_h_bw = tf.split(con_h, num_or_size_splits=2, axis=-1)

            # 1. Full-Matching

            # (batch, seq_len, hidden_size), (batch, hidden_size)
            # -> (batch, seq_len, l)
            mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], mp_ws[0])
            mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], mp_ws[1])
            mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], mp_ws[0])
            mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], mp_ws[1])

            # 2. Maxpooling-Matching
            # (batch, seq_len1, seq_len2, l)
            mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, mp_ws[2])
            mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, mp_ws[3])

            # (batch, seq_len, l)
            mv_p_max_fw = tf.reduce_max(mv_max_fw, axis=2)
            mv_p_max_bw = tf.reduce_max(mv_max_bw, axis=2)
            mv_h_max_fw = tf.reduce_max(mv_max_fw, axis=1)
            mv_h_max_bw = tf.reduce_max(mv_max_bw, axis=1)

            # 3. Attentive-Matching

            # (batch, seq_len1, seq_len2)
            att_fw = attention(con_p_fw, con_h_fw)
            att_bw = attention(con_p_bw, con_h_bw)

            # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_h_fw = tf.expand_dims(con_h_fw, axis=1) * tf.expand_dims(att_fw, axis=3)
            att_h_bw = tf.expand_dims(con_h_bw, axis=1) * tf.expand_dims(att_bw, axis=3)

            # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_p_fw = tf.expand_dims(con_p_fw, axis=2) * tf.expand_dims(att_fw, axis=3)
            att_p_bw = tf.expand_dims(con_p_bw, axis=2) * tf.expand_dims(att_bw, axis=3)

            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
            att_mean_h_fw = div_with_small_value(tf.reduce_sum(att_h_fw, axis=2),
                                                 tf.reduce_sum(att_fw, axis=2, keepdims=True))
            att_mean_h_bw = div_with_small_value(tf.reduce_sum(att_h_bw, axis=2),
                                                 tf.reduce_sum(att_bw, axis=2, keepdims=True))

            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
            att_mean_p_fw = div_with_small_value(tf.reduce_sum(att_p_fw, axis=1),
                                                 tf.transpose(tf.reduce_sum(att_fw, axis=1, keepdims=True), [0, 2, 1]))
            att_mean_p_bw = div_with_small_value(tf.reduce_sum(att_p_bw, axis=1),
                                                 tf.transpose(tf.reduce_sum(att_bw, axis=1, keepdims=True), [0, 2, 1]))

            # (batch, seq_len, l)
            mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, mp_ws[4])
            mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, mp_ws[5])
            mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, mp_ws[4])
            mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, mp_ws[5])

            # 4. Max-Attentive-Matching

            # (batch, seq_len1, hidden_size)
            att_max_h_fw = tf.reduce_max(att_h_fw, axis=2)
            att_max_h_bw = tf.reduce_max(att_h_bw, axis=2)
            # (batch, seq_len2, hidden_size)
            att_max_p_fw = tf.reduce_max(att_p_fw, axis=1)
            att_max_p_bw = tf.reduce_max(att_p_bw, axis=1)

            # (batch, seq_len, l)
            mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, mp_ws[6])
            mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, mp_ws[7])
            mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, mp_ws[6])
            mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, mp_ws[7])

            # print("mv_h_att_max_bw = ", mv_h_att_max_bw.shape.as_list())
            # (batch, seq_len, l * 8)
            mv_p = tf.concat(
                [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
                 mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], axis=2
            )

            mv_h = tf.concat(
                [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
                 mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], axis=2
            )

            mv_p = tf.layers.dropout(mv_p, self._dropout_rate, training=is_training)
            mv_h = tf.layers.dropout(mv_h, self._dropout_rate, training=is_training)

            # ----- Aggregation Layer -----
            with tf.variable_scope("aggregation_layer", reuse=tf.AUTO_REUSE):
                # (batch, seq_len, l * 8) -> (batch, 2 * hidden_size)

                _, (p_states, _) = nn.bi_lstm(mv_p, self._hidden_dim)

                _, (h_states, _) = nn.bi_lstm(mv_h, self._hidden_dim)

                x = tf.concat([p_states, h_states], axis=-1)
            x = tf.layers.dropout(x, self._dropout_rate, training=is_training)

            # ----- Prediction Layer -----

            # ----- Prediction Layer -----

            x = tf.layers.dense(x, self._hidden_dim * 2,
                                            activation=tf.tanh,
                                            kernel_initializer=tf.random_uniform_initializer(-0.005, 0.005),
                                            bias_initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
            x = tf.layers.dropout(x, self._dropout_rate, training=is_training)
            logits = tf.layers.dense(x, self._num_classes,
                                            activation=tf.tanh,
                                            kernel_initializer=tf.random_uniform_initializer(-0.005, 0.005),
                                            bias_initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            predictions = tf.argmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            output_dict = {'logits': logits, 'predictions': probs}

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels_embedding = features_embedding['label/labels']
                labels = features['label/labels']

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_embedding, logits=logits))
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predictions)
                metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predictions)
                metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predictions)
                metrics['auc'] = tf.metrics.auc(labels=labels, predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [tf.shape(hypothesis_tokens), tf.shape(hypothesis_chars), tf.shape(char_h),
                #                          tf.shape(premise_tokens), tf.shape(premise_chars), tf.shape(char_p)]
            return output_dict
