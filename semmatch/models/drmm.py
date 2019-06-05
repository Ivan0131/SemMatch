import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn


#A Deep Relevance Matching Model for Ad-hoc Retrieval
#https://arxiv.org/abs/1711.08611
@register.register_subclass('model', 'drmm')
class DRMM(Model):
    """
    DRMM Model
    Arguments:
        top_k: Size of top-k pooling layer.
        mlp_num_layers: The number of mlp layers.
        mlp_num_units: The hidden size of mlp.
        mlp_num_fan_out: The output size of mlp.
        mlp_activation_func: The activation function of mlp.
    """
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 top_k: int = 10, mlp_num_layers: int = 1, mlp_num_units: int = 5, mlp_num_fan_out: int = 1,
                 mlp_activation_func: str = 'tanh',
                 model_name: str = 'drmm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._top_k = top_k
        self._mlp_num_layers = mlp_num_layers
        self._mlp_num_units = mlp_num_units
        self._mlp_num_fan_out = mlp_num_fan_out
        self._mlp_activation_func = mlp_activation_func

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

            dense_output = tf.layers.dense(premise_tokens, 1, use_bias=False)
            dense_output += (1-prem_mask) * tf.float32.min
            attention_probs = tf.nn.softmax(dense_output, axis=1)

            # Matching histogram of top-k
            # shape = [B, M, N]
            matching_matrix = tf.matmul(tf.nn.l2_normalize(premise_tokens, axis=2), tf.nn.l2_normalize(hypothesis_tokens, axis=2),
                                        transpose_b=True)
            # shape = [B, M, K]
            matching_topk = tf.nn.top_k(matching_matrix, k=self._top_k, sorted=True)[0]

            # Feedforward matching topk
            # shape = [B, M, 1]
            dense_output = matching_topk
            for i in range(self._mlp_num_layers):
                dense_output = tf.layers.Dense(self._mlp_num_units, activation=self._mlp_activation_func, use_bias=True)(dense_output)
            dense_output = tf.layers.Dense(self._mlp_num_fan_out, activation=self._mlp_activation_func, use_bias=True)(dense_output)

            # shape = [B, 1, 1]
            dot_score = tf.matmul(attention_probs, dense_output, transpose_a=True)
            flatten_score = tf.reshape(dot_score, [-1, 1])
            # Get prediction
            output_dict = self._make_output(flatten_score, params)

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
                # metrics['map'] = tf.metrics.average_precision_at_k(labels=tf.cast(labels, tf.int64), predictions=output_dict['logits'],
                #                                                    k=2)
                # metrics['precision_1'] = tf.metrics.precision_at_k(labels=tf.cast(labels, tf.int64), predictions=output_dict['logits'],
                #                                                    k=1, class_id=1)

                    #tf.metrics.auc(labels=labels, predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict
