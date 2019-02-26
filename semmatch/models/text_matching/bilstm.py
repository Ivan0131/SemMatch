import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn


@register.register_subclass('model', 'text_matching_bilstm')
class BiLSTM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(), hidden_dim: int = 300, keep_prob:float = 0.5,
                 model_name: str = 'bilstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_prob = keep_prob

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            if 'premise/tokens' not in features:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens.")
            if 'hypothesis/tokens' not in features:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens.")
            prem_seq_lengths, prem_mask = nn.length(features['premise/tokens'])
            hyp_seq_lengths, hyp_mask = nn.length(features['hypothesis/tokens'])
            prem_mask = tf.expand_dims(prem_mask, -1)
            hyp_mask = tf.expand_dims(hyp_mask, -1)

            premise_tokens = features_embedding['premise/tokens']
            hypothesis_tokens = features_embedding['hypothesis/tokens']

            premise_outs, c1 = nn.bi_lstm(premise_tokens, self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = nn.bi_lstm(hypothesis_tokens, self._hidden_dim, seq_len=hyp_seq_lengths, name='hypothesis')

            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            premise_bi = premise_bi * prem_mask
            hypothesis_bi = hypothesis_bi * hyp_mask

            eps = 1e-11
            ### Mean pooling
            premise_sum = tf.reduce_sum(premise_bi, 1)
            premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1)+eps)

            hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
            hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1)+eps)

            ### Mou et al. concat layer ###
            diff = tf.subtract(premise_ave, hypothesis_ave)
            mul = tf.multiply(premise_ave, hypothesis_ave)
            h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

            # MLP layer
            h_mlp = tf.contrib.layers.fully_connected(h, self._hidden_dim, scope='fc1')
            # Dropout applied to classifier
            h_drop = tf.layers.dropout(h_mlp, self._dropout_prob, training=is_training)
            # Get prediction
            logits = tf.contrib.layers.fully_connected(h_drop, self._num_classes, activation_fn=None, scope='logits')

            predictions = tf.arg_max(logits, -1)
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
                metrics['auc'] = tf.metrics.auc(labels=labels, predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict
