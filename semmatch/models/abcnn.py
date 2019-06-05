import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn
import numpy as np


#ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
#https://arxiv.org/abs/1512.05193
#to do 这论文是个坑 之后再调
@register.register_subclass('model', 'abcnn')
class ABCNN(Model):
    #debug NaN loss during training.
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 50, model_type: str = "ABCNN3", kernel_size: int = 7, max_length: int = 40,
                 num_layers: int = 2, model_name: str = 'abcnn'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._model_type = model_type
        self._kernel_size = kernel_size
        self._num_layers = num_layers
        self._max_length = max_length

    def forward(self, features, labels, mode, params):
        eps = 1e-12
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

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            s = self._max_length
            d0 = premise_tokens.get_shape()[2]
            # zero padding to inputs for wide convolution

            def pad_for_wide_conv(x):
                return tf.pad(x, np.array([[0, 0], [0, 0], [self._kernel_size - 1, self._kernel_size - 1], [0, 0]]), "CONSTANT",
                              name="pad_wide_conv")

            def cos_sim(v1, v2):
                norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
                norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
                dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

                return dot_products / (norm1 * norm2+eps)

            def make_attention_mat(x1, x2):
                # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
                # x2 => [batch, height, 1, width]
                # [batch, width, wdith] = [batch, s, s]
                euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1)+eps)
                return 1.0 / (1.0 + euclidean)

            def convolution(name_scope, x, d, reuse):
                with tf.name_scope(name_scope + "-conv"):
                    with tf.variable_scope("conv") as scope:
                        conv = tf.contrib.layers.conv2d(
                            inputs=x,
                            num_outputs=self._hidden_dim,
                            kernel_size=(d, self._kernel_size),
                            stride=1,
                            padding="VALID",
                            activation_fn=tf.nn.tanh,
                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            #weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                            biases_initializer=tf.constant_initializer(1e-04),
                            reuse=reuse,
                            trainable=True,
                            scope=scope
                        )
                        # Weight: [filter_height, filter_width, in_channels, out_channels]
                        # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                        # [batch, di, s+w-1, 1]
                        conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                        return conv_trans

            def w_pool(variable_scope, x, attention):
                # x: [batch, di, s+w-1, 1]
                # attention: [batch, s+w-1]
                with tf.variable_scope(variable_scope + "-w_pool"):
                    if self._model_type == "ABCNN2" or self._model_type == "ABCNN3":
                        pools = []
                        # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                        attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                        for i in range(s):
                            # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                            pools.append(tf.reduce_sum(x[:, :, i:i + self._kernel_size, :] * attention[:, :, i:i + self._kernel_size, :],
                                                       axis=2,
                                                       keep_dims=True))

                        # [batch, di, s, 1]
                        w_ap = tf.concat(pools, axis=2, name="w_ap")
                    else:
                        w_ap = tf.layers.average_pooling2d(
                            inputs=x,
                            # (pool_height, pool_width)
                            pool_size=(1, self._kernel_size),
                            strides=1,
                            padding="VALID",
                            name="w_ap"
                        )
                        # [batch, di, s, 1]

                    return w_ap

            def all_pool(variable_scope, x):
                with tf.variable_scope(variable_scope + "-all_pool"):
                    if variable_scope.startswith("input"):
                        pool_width = s
                        d = d0
                    else:
                        pool_width = s + self._kernel_size - 1
                        d = self._hidden_dim

                    all_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, pool_width),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )
                    # [batch, di, 1, 1]

                    # [batch, di]
                    all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                    # all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                    return all_ap_reshaped

            def CNN_layer(variable_scope, x1, x2, d):
                # x1, x2 = [batch, d, s, 1]
                with tf.variable_scope(variable_scope):
                    if self._model_type == "ABCNN1" or self._model_type == "ABCNN3":
                        with tf.name_scope("att_mat"):
                            aW = tf.get_variable(name="aW",
                                                 shape=(s, d),
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 #regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
                                                 )

                            # [batch, s, s]
                            att_mat = make_attention_mat(x1, x2)

                            # [batch, s, s] * [s,d] => [batch, s, d]
                            # matrix transpose => [batch, d, s]
                            # expand dims => [batch, d, s, 1]
                            x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                            x2_a = tf.expand_dims(tf.matrix_transpose(
                                tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)), -1)

                            # [batch, d, s, 2]
                            x1 = tf.concat([x1, x1_a], axis=3)
                            x2 = tf.concat([x2, x2_a], axis=3)

                    left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=False)
                    right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=True)

                    left_attention, right_attention = None, None

                    if self._model_type == "ABCNN2" or self._model_type == "ABCNN3":
                        # [batch, s+w-1, s+w-1]
                        att_mat = make_attention_mat(left_conv, right_conv)
                        # [batch, s+w-1], [batch, s+w-1]
                        left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat,
                                                                                                        axis=1)

                    left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)
                    left_ap = all_pool(variable_scope="left", x=left_conv)
                    right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
                    right_ap = all_pool(variable_scope="right", x=right_conv)

                    return left_wp, left_ap, right_wp, right_ap

            x1_expanded = tf.expand_dims(tf.transpose(premise_tokens, [0, 2, 1]), -1)
            x2_expanded = tf.expand_dims(tf.transpose(hypothesis_tokens, [0, 2, 1]), -1)

            LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)
            RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)

            LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
            sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)]

            #if self._num_layers > 1:
            for i in range(1, self._num_layers):
                _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", x1=LI_1, x2=RI_1, d=self._hidden_dim)
                # self.test = LO_2
                # self.test2 = RO_2
                sims.append(cos_sim(LO_2, RO_2))

            with tf.variable_scope("output-layer"):
                output_features = tf.concat([tf.stack(sims, axis=1)], axis=1,
                                                 name="output_features")

                output_dict = self._make_output(output_features, params)

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
                metrics['auc'] = tf.metrics.auc(labels=labels, predictions=output_dict['predictions'])
                output_dict['metrics'] = metrics
            return output_dict
