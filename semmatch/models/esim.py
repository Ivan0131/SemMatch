import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn


#Enhanced LSTM for Natural Language Inference
#https://arxiv.org/abs/1609.06038
@register.register_subclass('model', 'esim')
class ESIM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 300, dropout_rate: float = 0.5, model_name: str = 'esim'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate

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
                prem_seq_lengths -= 2
            if features.get('hypothesis/elmo_characters', None) is not None:
                hyp_mask = hyp_mask[:, 1:-1]
                hyp_seq_lengths -= 2
            if isinstance(self._embedding_mapping.get_encoder('tokens'), Bert):
                prem_mask = prem_mask[:, 1:-1]
                prem_seq_lengths -= 2
                hyp_mask = hyp_mask[:, 1:-1]
                hyp_seq_lengths -= 2

            prem_mask = tf.expand_dims(prem_mask, -1)
            hyp_mask = tf.expand_dims(hyp_mask, -1)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            premise_outs, c1 = nn.bi_lstm(premise_tokens, self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = nn.bi_lstm(hypothesis_tokens, self._hidden_dim, seq_len=hyp_seq_lengths, name='hypothesis')

            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            premise_bi *= prem_mask
            hypothesis_bi *= hyp_mask

            ### Attention ###
            premise_attns, hypothesis_attns = nn.bi_uni_attention(premise_bi, hypothesis_bi, prem_seq_lengths,
                                                                  hyp_seq_lengths, func="dot")

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

            v1_bi = v1_bi * prem_mask
            v2_bi = v2_bi * hyp_mask

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
            h_drop = tf.layers.dropout(h_mlp, self._dropout_rate, training=is_training)

            # Get prediction
            logits = tf.contrib.layers.fully_connected(h_drop, self._num_classes, activation_fn=None, scope='logits')

            predictions = tf.argmax(logits, -1)

            output_dict = {'logits': logits, 'predictions': predictions}

            probs = tf.nn.softmax(logits, -1)
            output_score = tf.estimator.export.PredictOutput(probs)
            export_outputs = {"output_score": output_score}
            output_dict['export_outputs'] = export_outputs

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
               # metrics['auc'] = tf.metrics.auc(labels=labels, predictions=predictions)

                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          v_1_ave, v_2_ave, h_mlp, logits]
            return output_dict
