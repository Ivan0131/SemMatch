import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn


#A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations
#https://arxiv.org/abs/1511.08277
@register.register_subclass('model', 'mvlstm')
class MVLSTM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 50, dropout_rate: float = 0.5, top_k: int = 10, num_tensor_dim: int = 8,
                 sim_func: str = "cosine",
                 model_name: str = 'mvlstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._num_k = top_k
        self._num_tensor_dim = num_tensor_dim
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate
        self._sim_func = sim_func

    def forward(self, features, labels, mode, params):
        if self._sim_func != 'tensor' and self._num_tensor_dim != 1:
            self._num_tensor_dim = 1
            logger.warning("The similarity function is tensor layer. The number of tensor dim is not effective.")
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
            prem_hyp_mask = tf.matmul(prem_mask, hyp_mask, transpose_b=True)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            premise_outs, c1 = nn.bi_lstm(premise_tokens, self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = nn.bi_lstm(hypothesis_tokens, self._hidden_dim, seq_len=hyp_seq_lengths,
                                             name='hypothesis')
            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            max_premise_length = premise_tokens.shape[1].value
            max_hypothesis_length = hypothesis_tokens.shape[1].value

            if self._sim_func == 'tensor':
                M = tf.Variable(tf.random_normal([self._num_tensor_dim, 2 * self._hidden_dim, 2 * self._hidden_dim], stddev=0.1))
                W = tf.Variable(tf.random_normal([4 * self._hidden_dim, 1], stddev=0.1))
                bias = tf.Variable(tf.zeros([1]), name="tensor_bias")
                premise_ex = tf.tile(tf.expand_dims(premise_bi, axis=2), [1, 1, max_hypothesis_length, 1])
                hypothesis_ex = tf.tile(tf.expand_dims(hypothesis_bi, axis=1), [1, max_premise_length, 1, 1])
                tensor = []
                tmp2 = tf.einsum("abcd,df->abcf", tf.concat([premise_ex, hypothesis_ex], axis=3), W)  # [N, L1, L2, 1]
                tmp2 = tf.squeeze(tmp2, axis=3)
                for i in range(self._num_tensor_dim):
                    tmp1 = tf.einsum("abc,cd->abd", premise_bi, M[i])  # [N, L1, 2d]
                    tmp1 = tf.matmul(tmp1, hypothesis_bi, transpose_b=True)  # [N, L1, L2]
                    tensor.append(tf.nn.relu(tmp1 + tmp2+bias))
                tensor = tf.concat([tensor], axis=0)
            elif self._sim_func == 'cosine':
                tensor = tf.matmul(tf.nn.l2_normalize(premise_bi, axis=-1),
                                  tf.nn.l2_normalize(hypothesis_bi, axis=-1),
                                  transpose_b=True)  # [N, L1, L2]
            elif self._sim_func == 'bilinear':
                M = tf.Variable(tf.random_normal([2 * self._hidden_dim, 2 * self._hidden_dim], stddev=0.1))
                b = tf.Variable(tf.random_normal([max_premise_length, max_hypothesis_length], stddev=0.1))
                bilinear = tf.einsum("abc,cd->abd", premise_bi, M)  # [N, L1, 2d]
                tensor = tf.matmul(bilinear, hypothesis_bi, transpose_b=True) + b  # [N, L1, L2]
            else:
                raise ConfigureError("The simility function %s is not supported. "
                                     "The mvlstm only support simility function for [cosine, bilinear, tensor]." % self._sim_func)

            tensor *= prem_hyp_mask
            # 3.1 k-Max Pooling
            matrix_in = tf.reshape(tensor, [-1, max_premise_length * max_hypothesis_length])
            values, indices = tf.nn.top_k(matrix_in, k=self._num_k, sorted=False)
            kmax = tf.reshape(values, [-1, self._num_tensor_dim * self._num_k])

            # MLP layer
            h_mlp_1 = tf.contrib.layers.fully_connected(kmax, self._num_tensor_dim * self._num_k, scope='fc1')
            h_mlp_1_drop = tf.layers.dropout(h_mlp_1, self._dropout_rate, training=is_training)
            h_mlp_2 = tf.contrib.layers.fully_connected(h_mlp_1_drop, self._num_tensor_dim * self._num_k//2, scope='fc2')

            # Dropout applied to classifier
            h_drop = tf.layers.dropout(h_mlp_2, self._dropout_rate, training=is_training)
            # Get prediction
            output_dict = self._make_output(h_drop, params)

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

                    #tf.metrics.auc(labels=labels, predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict
