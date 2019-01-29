import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.nn import layers


@register.register_subclass('model', 'text_matching_bilstm')
class BiLSTM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(), hidden_dim: int = 300, keep_prob:float = 0.5,
                 model_name: str = 'bilstm'):
        super().__init__(optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_prob = keep_prob
        self._reuse = False

    def forward(self, features, labels, mode, params):
        with tf.variable_scope(self._model_name) as scope:
            if self._reuse:
                scope.reuse_variables()
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            if 'premise/tokens' not in features:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens.")
            if 'hypothesis/tokens' not in features:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens.")
            prem_seq_lengths, prem_mask = layers.length(features['premise/tokens'])
            hyp_seq_lengths, hyp_mask = layers.length(features['hypothesis/tokens'])
            features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
            premise_tokens = features_embedding['premise/tokens']
            hypothesis_tokens = features_embedding['hypothesis/tokens']

            premise_outs, c1 = layers.biLSTM(premise_tokens, dim=self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = layers.biLSTM(hypothesis_tokens, dim=self._hidden_dim, seq_len=hyp_seq_lengths, name='hypothesis')

            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            ### Mean pooling
            premise_sum = tf.reduce_sum(premise_bi, 1)
            premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

            hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
            hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

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
            eps = 1e-10
            logits += eps
            predictions = tf.arg_max(logits, -1)
            output_dict = {'logits': logits, 'predictions': predictions}

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels = features_embedding['label/labels']
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=tf.argmax(labels, -1), predictions=predictions)
                output_dict['metrics'] = metrics
                output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi, h, h_mlp, logits]
            self._reuse = True
            return output_dict
