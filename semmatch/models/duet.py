from typing import List
import tensorflow as tf
from tensorflow import keras
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn


#Learning to Match Using Local and Distributed Representations of Text for Web Search
#https://arxiv.org/abs/1610.08136
@register.register_subclass('model', 'duet')
class Duet(Model):
    """
    Duet Model

    Arguments:
        lm_filters: Filter size of 1D convolution layer in the local model.
        lm_hidden_sizes: A list of hidden size of the MLP layer in the local model.
        dm_filters: Filter size of 1D convolution layer in the distributed model.
        dm_kernel_size: Kernel size of 1D convolution layer in the distributed model.
        dm_q_hidden_size: Hidden size of the MLP layer for the left text in the distributed model.
        dm_d_mpool: Max pooling size for the right text in the distributed model.
        dm_hidden_sizes: A list of hidden size of the MLP layer in the distributed model.
        activation_func: Activation function in the convolution layer.
        dropout_rate: The dropout rate.
    """
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 lm_filters: int = 32, lm_hidden_sizes: List[int] = [32], dropout_rate: float = 0.5,
                 dm_filters: int = 32, dm_kernel_size: int = 3, dm_d_mpool: int = 3,
                 dm_q_hidden_size: int = 32,
                 dm_hidden_sizes: List[int] = [32], activation_func: str = 'relu',
                 model_name: str = 'duet'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._lm_filters = lm_filters
        self._lm_hidden_sizes = lm_hidden_sizes
        self._dropout_rate = dropout_rate
        self._dm_filters = dm_filters
        self._dm_kernel_size = dm_kernel_size
        self._dm_d_mpool = dm_d_mpool
        self._dm_hidden_sizes = dm_hidden_sizes
        self._activation_func = activation_func
        self._dm_q_hidden_size = dm_q_hidden_size

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

            lm_xor = keras.layers.Lambda(self._xor_match)([premise_tokens_ids, hypothesis_tokens_ids])
            lm_conv = keras.layers.Conv1D(
                self._lm_filters,
                premise_tokens_ids.shape[1].value,
                padding='valid',
                activation=self._activation_func
            )(lm_xor)

            lm_conv = keras.layers.Dropout(self._dropout_rate)(
                lm_conv, training=is_training)
            lm_feat = keras.layers.Reshape((lm_conv.shape[2].value, ))(lm_conv)
            for hidden_size in self._lm_hidden_sizes:
                lm_feat = keras.layers.Dense(
                    hidden_size,
                    activation=self._activation_func
                )(lm_feat)
            lm_drop = keras.layers.Dropout(self._dropout_rate)(
                lm_feat, training=is_training)
            lm_score = keras.layers.Dense(1)(lm_drop)

            dm_q_conv = keras.layers.Conv1D(
                self._dm_filters,
                self._dm_kernel_size,
                padding='same',
                activation=self._activation_func
            )(premise_tokens)
            dm_q_conv = keras.layers.Dropout(self._dropout_rate)(
                dm_q_conv, training=is_training)
            dm_q_mp = keras.layers.MaxPooling1D(
                pool_size=premise_tokens_ids.shape[1].value)(dm_q_conv)
            dm_q_rep = keras.layers.Reshape((dm_q_mp.shape[2].value, ))(dm_q_mp)
            dm_q_rep = keras.layers.Dense(self._dm_q_hidden_size)(
                dm_q_rep)
            dm_q_rep = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(
                dm_q_rep)

            dm_d_conv1 = keras.layers.Conv1D(
                self._dm_filters,
                self._dm_kernel_size,
                padding='same',
                activation=self._activation_func
            )(hypothesis_tokens)
            dm_d_conv1 = keras.layers.Dropout(self._dropout_rate)(
                dm_d_conv1, training=is_training)
            dm_d_mp = keras.layers.MaxPooling1D(
                pool_size=self._dm_d_mpool)(dm_d_conv1)
            dm_d_conv2 = keras.layers.Conv1D(
                self._dm_filters, 1,
                padding='same',
                activation=self._activation_func
            )(dm_d_mp)
            dm_d_conv2 = keras.layers.Dropout(self._dropout_rate)(
                dm_d_conv2, training=is_training)

            h_dot = dm_q_rep * dm_d_conv2 #keras.layers.Lambda(self._hadamard_dot)([dm_q_rep, dm_d_conv2])
            dm_feat = keras.layers.Reshape((h_dot.shape[1].value*h_dot.shape[2].value, ))(h_dot)
            for hidden_size in self._dm_hidden_sizes:
                dm_feat = keras.layers.Dense(hidden_size)(dm_feat)
            dm_feat_drop = keras.layers.Dropout(self._dropout_rate)(
                dm_feat, training=is_training)
            dm_score = keras.layers.Dense(1)(dm_feat_drop)

            add = keras.layers.Add()([lm_score, dm_score])

            # Get prediction
            output_dict = self._make_output(add, params)

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

    @classmethod
    def _xor_match(cls, x):
        t1 = x[0]
        t2 = x[1]
        t1_shape = t1.get_shape()
        t2_shape = t2.get_shape()
        t1_expand = tf.stack([t1] * t2_shape[1].value, 2)
        t2_expand = tf.stack([t2] * t1_shape[1].value, 1)
        out_bool = tf.equal(t1_expand, t2_expand)
        out = tf.cast(out_bool, tf.float32)
        return out
