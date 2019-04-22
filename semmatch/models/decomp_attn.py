import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn


#A Decomposable Attention Model for Natural Language Inference (EMNLP 2016)
#https://arxiv.org/abs/1606.01933
@register.register_subclass('model', 'decomp_attn')
class DecomposableAttention(Model):
    #Model diverged with loss = NaN.
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 200, keep_prob:float = 0.8,
                 model_name: str = 'bilstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_prob = 1-keep_prob

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
            if features.get('premise/elmo_characters', None) is not None:
                prem_mask = prem_mask[:, 1:-1]
                hyp_mask = hyp_mask[:, 1:-1]
            # prem_mask = tf.expand_dims(prem_mask, -1)
            # hyp_mask = tf.expand_dims(hyp_mask, -1)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            with tf.variable_scope("Attend"):
                F_a_bar = self._feedForwardBlock(premise_tokens, self._hidden_dim, 'F')
                F_b_bar = self._feedForwardBlock(hypothesis_tokens, self._hidden_dim, 'F', isReuse=True)

                # e_i,j = F'(a_hat, b_hat) = F(a_hat).T * F(b_hat) (1)
                #alignment_attention = Attention(self.hidden_size, self.hidden_size)
                #alpha = alignment_attention(F_b_bar, F_a_bar, keys_mask=self.query_mask)
                #beta = alignment_attention(F_a_bar, F_b_bar, keys_mask=self.doc_mask)
                alpha, beta = nn.bi_uni_attention(F_a_bar, F_b_bar, query_len=prem_seq_lengths, key_len=hyp_seq_lengths)

            with tf.variable_scope("Compare"):
                a_beta = tf.concat([premise_tokens, alpha], axis=2)
                b_alpha = tf.concat([hypothesis_tokens, beta], axis=2)

                # v_1,i = G([a_bar_i, beta_i])
                # v_2,j = G([b_bar_j, alpha_j]) (3)
                v_1 = self._feedForwardBlock(a_beta, self._hidden_dim, 'G')
                v_2 = self._feedForwardBlock(b_alpha, self._hidden_dim, 'G', isReuse=True)

            with tf.variable_scope("Aggregate"):
                # v1 = \sum_{i=1}^l_a v_{1,i}
                # v2 = \sum_{j=1}^l_b v_{2,j} (4)
                v1_sum = tf.reduce_sum(v_1, axis=1)
                v2_sum = tf.reduce_sum(v_2, axis=1)

                # y_hat = H([v1, v2]) (5)
                v = tf.concat([v1_sum, v2_sum], axis=1)

                ff_outputs = self._feedForwardBlock(v, self._hidden_dim, 'H')

                logits = tf.layers.dense(ff_outputs, self._num_classes)

            predictions = tf.argmax(logits, -1)
            output_dict = {'logits': logits, 'predictions': predictions}

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
                #metrics['auc'] = tf.metrics.auc(labels=labels, predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [tf.shape(hypothesis_tokens), tf.shape(premise_tokens),
                #                          tf.shape(alpha), tf.shape(beta)]
            return output_dict

    def _feedForwardBlock(self, inputs, num_units, scope, isReuse = False, initializer = None):
        """
        :param inputs: tensor with shape (batch_size, seq_length, embedding_size)
        :param num_units: dimensions of each feed forward layer
        :param scope: scope name
        :return: output: tensor with shape (batch_size, num_units)
        """
        with tf.variable_scope(scope, reuse=isReuse):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, 1 - self._dropout_prob)
                outputs = tf.layers.dense(inputs, num_units, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, 1 - self._dropout_prob)
                resluts = tf.layers.dense(outputs, num_units, tf.nn.relu, kernel_initializer=initializer)
                return resluts

