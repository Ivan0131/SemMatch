import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn


#Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction
#https://arxiv.org/abs/1806.00778
@register.register_subclass('model', 'mcan')
class MCAN(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 200, dropout_rate: float = 0.2,
                 model_name: str = 'mcan'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate
        self._eps = 1e-15

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            premise_tokens_ids = features.get('premise/tokens', None)
            if premise_tokens_ids is None:
                premise_tokens_ids = features.get('premise/elmo_characters', None)
            hypothesis_tokens_ids = features.get('hypothesis/tokens', None)
            if hypothesis_tokens_ids is None:
                hypothesis_tokens_ids = features.get('hypothesis/elmo_characters', None)

            if premise_tokens_ids is None:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens "
                                     "or elmo_characters.")
            if hypothesis_tokens_ids is None:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens "
                                     "or elmo_characters.")

            prem_seq_lengths, prem_mask = nn.length(premise_tokens_ids)
            hyp_seq_lengths, hyp_mask = nn.length(hypothesis_tokens_ids)
            if features.get('premise/elmo_characters', None) is not None or isinstance(self._embedding_mapping.get_encoder('tokens'), Bert):
                prem_mask = nn.remove_bos_eos(prem_mask, prem_seq_lengths)
                prem_seq_lengths -= 2
            if features.get('hypothesis/elmo_characters', None) is not None or isinstance(self._embedding_mapping.get_encoder('tokens'), Bert):
                hyp_mask = nn.remove_bos_eos(hyp_mask, hyp_seq_lengths)
                hyp_seq_lengths -= 2
            prem_mask = tf.expand_dims(prem_mask, -1)
            hyp_mask = tf.expand_dims(hyp_mask, -1)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            # 2.Input Encoder
            # 2.1 Highway Encoder
            query_emb = premise_tokens
            doc_emb = hypothesis_tokens
            query_len = prem_seq_lengths
            doc_len = hyp_seq_lengths
            query_mask = prem_mask
            doc_mask = hyp_mask
            project_dim = premise_tokens.shape[-1].value
            query_length = tf.shape(premise_tokens)[1]
            doc_length = tf.shape(hypothesis_tokens)[1]

            query_output = nn.highway_network(query_emb, 1, dropout_rate=self._dropout_rate, is_trainging=is_training,
                                              scope="query_highway")
            doc_output = nn.highway_network(doc_emb, 1, dropout_rate=self._dropout_rate, is_trainging=is_training,
                                            scope="doc_highway")

            # # 2.2 Co-Attention
            M = tf.Variable(tf.random_normal([project_dim, project_dim], stddev=0.1))
            tmp = tf.einsum("ijk,kl->ijl", query_output, M)
            S = tf.matmul(tmp, doc_output, transpose_b=True)  # [batch, q, d]
            S_mask = tf.matmul(query_mask, doc_mask, transpose_b=True)
            S_mean = S * S_mask #
            S_align_max = S + (1. - S_mask) * tf.float32.min

            # 2.2.1 Extractive Pooling
            # Max Pooling
            query_score = tf.nn.softmax(tf.reduce_max(S_align_max, axis=2, keepdims=True), axis=1)
            query_maxpooling = tf.reduce_sum(query_score * query_output, axis=1) # [batch, r]

            doc_score = tf.nn.softmax(tf.reduce_max(S_align_max, axis=1, keepdims=True), axis=2)
            doc_maxpooling = tf.reduce_sum(tf.transpose(doc_score, [0, 2, 1]) * doc_output, axis=1) # [batch, r]

            # Mean Pooling
            query_score = tf.nn.softmax(tf.reduce_sum(S_mean, axis=2, keepdims=True)/(tf.expand_dims(tf.expand_dims(tf.cast(doc_len, tf.float32)+self._eps, -1), -1)), axis=1)
            query_meanpooling = tf.reduce_sum(query_score * query_output, axis=1)  # [batch, r]
            doc_score = tf.nn.softmax(tf.reduce_sum(S_mean, axis=1, keepdims=True)/(tf.expand_dims(tf.expand_dims(tf.cast(query_len, tf.float32)+self._eps, -1), -1)), axis=2)
            doc_meanpooling = tf.reduce_sum(tf.transpose(doc_score, [0, 2, 1]) * doc_output, axis=1)  # [batch, r]

            # 2.2.2 Alignment Pooling
            query_alignment = tf.matmul(tf.nn.softmax(S_align_max, axis=2), doc_output)  # [batch, q, r]
            doc_alignment = tf.matmul(tf.nn.softmax(S_align_max, axis=1), query_output, transpose_a=True)  # [batch, d, r]

            # 2.2.3 Intra Attention
            query_selfattn = nn.self_attention(query_output, query_len)
            doc_selfattn = nn.self_attention(doc_output, doc_len)

            # 2.3 Multi-Cast Attention
            query_maxpooling = tf.tile(tf.expand_dims(query_maxpooling, axis=1), [1, query_length, 1])
            query_meanpooling = tf.tile(tf.expand_dims(query_meanpooling, axis=1), [1, query_length, 1])
            doc_maxpooling = tf.tile(tf.expand_dims(doc_maxpooling, axis=1), [1, doc_length, 1])
            doc_meanpooling = tf.tile(tf.expand_dims(doc_meanpooling, axis=1), [1, doc_length, 1])

            query_max_fc, query_max_fm, query_max_fs = self.cast_attention(query_maxpooling, query_emb, self.nn_fc, name="query_max_pooling")
            query_mean_fc, query_mean_fm, query_mean_fs = self.cast_attention(query_meanpooling, query_emb, self.nn_fc, name="query_mean_pooling")
            query_align_fcm, query_align_fm, query_align_fs = self.cast_attention(query_alignment, query_emb, self.nn_fc, name="query_align_pooling")
            query_selfattn_fc, query_selfattn_fm, query_selfattn_fs = self.cast_attention(query_selfattn, query_emb, self.nn_fc, name="query_self_pooling")

            doc_max_fc, doc_max_fm, doc_max_fs = self.cast_attention(doc_maxpooling, doc_emb, self.nn_fc, name="doc_max_pooling")
            doc_mean_fc, doc_mean_fm, doc_mean_fs = self.cast_attention(doc_meanpooling, doc_emb, self.nn_fc, name="doc_mean_pooling")
            doc_align_fcm, doc_align_fm, doc_align_fs = self.cast_attention(doc_alignment, doc_emb, self.nn_fc, name="doc_align_pooling")
            doc_selfattn_fc, doc_selfattn_fm, doc_selfattn_fs = self.cast_attention(doc_selfattn, doc_emb, self.nn_fc, name="doc_self_pooling")

            query_cast = tf.concat(
                [query_max_fc, query_max_fm, query_max_fs, query_mean_fc, query_mean_fm, query_mean_fs, query_align_fcm,
                 query_align_fm, query_align_fs, query_selfattn_fc, query_selfattn_fm, query_selfattn_fs, query_output],
                axis=2)
            doc_cast = tf.concat(
                [doc_max_fc, doc_max_fm, doc_max_fs, doc_mean_fc, doc_mean_fm, doc_mean_fs, doc_align_fcm,
                 doc_align_fm, doc_align_fs, doc_selfattn_fc, doc_selfattn_fm, doc_selfattn_fs, doc_output], axis=2)

            # query_cast = tf.concat(
            #     [
            #      query_output],
            #     axis=2)
            # doc_cast = tf.concat(
            #     [doc_output], axis=2)

            query_cast = tf.layers.dropout(query_cast, self._dropout_rate, training=is_training)
            doc_cast = tf.layers.dropout(doc_cast, self._dropout_rate, training=is_training)

            query_hidden, _ = nn.bi_lstm(query_cast, self._hidden_dim, name="query_lstm")
            doc_hidden, _ = nn.bi_lstm(doc_cast, self._hidden_dim, name="doc_lstm")

            query_hidden = tf.concat(query_hidden, axis=2)
            doc_hidden = tf.concat(doc_hidden, axis=2)
            query_hidden = tf.layers.dropout(query_hidden, self._dropout_rate, training=is_training)
            doc_hidden = tf.layers.dropout(doc_hidden, self._dropout_rate, training=is_training)

            #query_hidden_max = query_hidden + (1. - query_mask) * tf.float32.min
            #doc_hidden_max = doc_hidden + (1. - doc_mask) * tf.float32.min
            query_hidden_mean = query_hidden * query_mask
            doc_hidden_mean = doc_hidden * doc_mask

            query_sum = tf.reduce_sum(query_hidden_mean, axis=1)
            query_mean = tf.div(query_sum, tf.expand_dims(tf.cast(query_len, tf.float32), -1) + self._eps)

            query_max = tf.reduce_max(query_hidden_mean, axis=1)
            query_final = tf.concat([query_mean, query_max], axis=1)

            doc_sum = tf.reduce_sum(doc_hidden_mean, axis=1)
            doc_mean = tf.div(doc_sum, tf.expand_dims(tf.cast(doc_len, tf.float32), -1) + self._eps)

            doc_max = tf.reduce_max(doc_hidden_mean, axis=1)
            doc_final = tf.concat([doc_mean, doc_max], axis=1)

            final = tf.concat([query_final, doc_final, query_final * doc_final, query_final - doc_final], axis=1)
            #yout = nn.highway_network(final, 2, dropout_rate=self._drop_rate, is_trainging=is_training)
            # MLP layer
            yout = tf.contrib.layers.fully_connected(final, self._hidden_dim, scope='fc1')
            # Dropout applied to classifier

            output_dict = self._make_output(yout, params)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels_embedding = features_embedding['label/labels']
                labels = features['label/labels']

                loss = self._make_loss(labels=labels_embedding, logits=output_dict['logits'], params=params)
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=output_dict['predictions'])
                metrics['precision'] = tf.metrics.precision(labels=labels, predictions=output_dict['predictions'])
                metrics['recall'] = tf.metrics.recall(labels=labels, predictions=output_dict['predictions'])
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = []
                # debug_ops = [query_mean_fs]#[query_maxpooling, query_max_fc] [query_max_fm, query_max_fs],[query_mean_fc, query_mean_fm] , ,
                # for op in debug_ops:
                #     output_dict['debugs'].append(tf.shape(op))
                # output_dict['debugs'].append(query_length)
            return output_dict

    def cast_attention(self, x, y, func, name="cast_attention"):
        with tf.variable_scope(name):
            fc = self.nn_fc(tf.concat([x, y], axis=2))
            fm = self.nn_fc(tf.multiply(x, y))
            fs = self.nn_fc(tf.subtract(x, y))
            fc = func(fc, name="fc")
            fm = func(fm, name="fm")
            fs = func(fs, name="fm")
        return fc, fm, fs

    def sum_fc(self, x):
        return tf.reduce_sum(x, axis=-1, keepdim=True)

    def nn_fc(self, x, name="nn_fc"):
        return tf.layers.Dense(1, activation=tf.nn.relu, use_bias=True, name=name)(x)

    # def fm_fc(self, x):
    #     element_wise_product_list = []
    #     v = tf.Variable(tf.random_uniform(shape=[self.r, self.k]))
    #     vvt = tf.matmul(v, v, transpose_b=True) # [r, r]
    #     depth = x.shape[-1].value
    #     for l in range(self.sequence_length):
    #         current_index = 0.0
    #         for i in range(depth):
    #             for j in range(i + 1, depth):
    #                 current_index += x[:,l,i] * x[:,l,j] * vvt[i,j]
    #         element_wise_product_list.append(current_index)
    #     element_wise_product = tf.stack(element_wise_product_list, axis = 1) # [batch, d]
    #     element_wise_product = tf.expand_dims(element_wise_product, axis = -1) # [batch, d, 1]
    #     linear = tf.layers.Dense(1)(x)
    #
    #     return element_wise_product + linear

