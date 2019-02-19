import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn


@register.register_subclass('model', 'text_matching_esim')
class ESIM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 300, keep_prob: float = 0.5, model_name: str = 'esim'):
        super().__init__(optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_prob = keep_prob

    def forward(self, features, labels, mode, params):
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            if 'premise/tokens' not in features:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens.")
            if 'hypothesis/tokens' not in features:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens.")
            prem_seq_lengths, prem_mask = nn.length(features['premise/tokens'])
            hyp_seq_lengths, hyp_mask = nn.length(features['hypothesis/tokens'])
            features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
            premise_tokens = features_embedding['premise/tokens']
            hypothesis_tokens = features_embedding['hypothesis/tokens']

            premise_outs, c1 = nn.bi_lstm(premise_tokens, self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = nn.bi_lstm(hypothesis_tokens, self._hidden_dim, seq_len=hyp_seq_lengths, name='hypothesis')

            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            ### Attention ###
            premise_attns, hypothesis_attns = nn.bi_uni_attention(premise_bi, hypothesis_bi, prem_seq_lengths,
                                                                  hyp_seq_lengths,
                                                                  similarity_function=nn.dot_similarity_function)

            # For making attention plots,
            prem_diff = tf.subtract(premise_bi, premise_attns)
            prem_mul = tf.multiply(premise_bi, premise_attns)
            hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
            hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

            m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2)
            m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2)

            ### Inference Composition ###

            v1_outs, c3 = nn.bi_lstm(m_a, self._hidden_dim, seq_len=prem_seq_lengths, name='v1')
            v2_outs, c4 = nn.bi_lstm(m_b, self._hidden_dim, seq_len=hyp_seq_lengths, name='v2')

            v1_bi = tf.concat(v1_outs, axis=2)
            v2_bi = tf.concat(v2_outs, axis=2)

            ### Pooling Layer ###
            eps = 1e-11
            v_1_sum = tf.reduce_sum(v1_bi, 1)
            v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1)+eps)

            v_2_sum = tf.reduce_sum(v2_bi, 1)
            v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1)+eps)

            v_1_max = tf.reduce_max(v1_bi, 1)
            v_2_max = tf.reduce_max(v2_bi, 1)

            v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)

            # MLP layer
            h_mlp = tf.contrib.layers.fully_connected(v, self._hidden_dim, activation_fn=tf.nn.tanh, scope='fc1')

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
                labels = features_embedding['label/labels']
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=tf.argmax(labels, -1), predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict
