import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn


#Natural Language Inference over Interaction Space
#https://arxiv.org/abs/1709.04348
@register.register_subclass('model', 'diin')
class DIIN(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 300, dropout_rate: float = 0.0, dropout_decay_step: int = 10000,
                 l2_loss: bool = True, sigmoid_growing_l2loss: bool = True, l2_regularization_ratio: float = 9e-5,
                 weight_l2loss_step_full_reg: int = 100000,
                 char_filter_sizes: str = "5", char_filter_channel_dims: str = "100",
                 dropout_decay_rate: float = 0.977, highway_num_layers: int = 2, num_self_att_enc_layers: int = 1,
                 first_scale_down_layer_relu: bool = False, dense_net_first_scale_down_ratio: float = 0.3,
                 first_scale_down_kernel: int = 1, dense_net_growth_rate: int = 20, num_dense_net_layers: int = 8,
                 dense_net_kernel_size: int = 3, dense_net_transition_rate: float = 0.5, diff_penalty_loss_ratio: float = 1e-3,
                 model_name: str = 'diin'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._keep_prob = 1. - dropout_rate
        self._initializer_range = 0.02
        self._dropout_decay_step = dropout_decay_step
        self._dropout_decay_rate = dropout_decay_rate
        self._highway_num_layers = highway_num_layers
        self._num_self_att_enc_layers = num_self_att_enc_layers
        self._first_scale_down_layer_relu = first_scale_down_layer_relu
        self._dense_net_first_scale_down_ratio = dense_net_first_scale_down_ratio
        self._first_scale_down_kernel = first_scale_down_kernel
        self._dense_net_growth_rate = dense_net_growth_rate
        self._num_dense_net_layers = num_dense_net_layers
        self._dense_net_kernel_size = dense_net_kernel_size
        self._dense_net_transition_rate = dense_net_transition_rate
        self._char_filter_size = list(map(int, char_filter_sizes.split(',')))
        self._char_filter_channel_dims = list(map(int, char_filter_channel_dims.split(',')))
        self._diff_penalty_loss_ratio = diff_penalty_loss_ratio
        self._l2_loss = l2_loss
        self._sigmoid_growing_l2loss = sigmoid_growing_l2loss
        self._l2_regularization_ratio = l2_regularization_ratio
        self._weight_l2loss_step_full_reg = weight_l2loss_step_full_reg

    def forward(self, features, labels, mode, params):
        global_step = tf.train.get_or_create_global_step()
        dropout_keep_rate = tf.train.exponential_decay(self._keep_prob, global_step,
                                                       self._dropout_decay_step, self._dropout_decay_rate,
                                                       staircase=False, name='dropout_keep_rate')
        tf.summary.scalar('dropout_keep_rate', dropout_keep_rate)

        params.add_hparam('dropout_rate', 1 - dropout_keep_rate)
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

            premise_ins = []
            hypothesis_ins = []

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            premise_ins.append(premise_tokens)
            hypothesis_ins.append(hypothesis_tokens)

            premise_chars = features_embedding.get('premise/chars', None)
            hypothesis_chars = features_embedding.get('hypothesis/chars', None)

            if premise_chars is not None and hypothesis_chars is not None:

                with tf.variable_scope("conv") as scope:
                    conv_pre = nn.multi_conv1d_max(premise_chars, self._char_filter_size, self._char_filter_channel_dims,
                                                   "VALID", is_training, dropout_keep_rate, scope='conv')
                    scope.reuse_variables()
                    conv_hyp = nn.multi_conv1d_max(hypothesis_chars, self._char_filter_size, self._char_filter_channel_dims,
                                                   "VALID", is_training, dropout_keep_rate, scope='conv')
                    #conv_pre = tf.reshape(conv_pre, [-1, self.sequence_length, config.char_out_size])
                    #conv_hyp = tf.reshape(conv_hyp, [-1, self.sequence_length, config.char_out_size])

                    premise_ins.append(conv_pre)
                    hypothesis_ins.append(conv_hyp)

            premise_pos = features_embedding.get('premise/pos_tags', None)
            hypothesis_pos = features_embedding.get('hypothesis/pos_tags', None)

            if premise_pos is not None and hypothesis_pos is not None:
                premise_ins.append(premise_pos)
                hypothesis_ins.append(hypothesis_pos)

            premise_exact_match = features.get('premise/exact_match_labels', None)
            hypothesis_exact_match = features.get('hypothesis/exact_match_labels', None)

            if premise_exact_match is not None and hypothesis_exact_match is not None:
                premise_ins.append(tf.expand_dims(tf.cast(premise_exact_match, tf.float32), -1))
                hypothesis_ins.append(tf.expand_dims(tf.cast(hypothesis_exact_match, tf.float32), -1))

            premise_in = tf.concat(premise_ins, axis=2)
            hypothesis_in = tf.concat(hypothesis_ins, axis=2)

            with tf.variable_scope("highway") as scope:
                premise_in = nn.highway_network(premise_in, self._highway_num_layers)
                scope.reuse_variables()
                hypothesis_in = nn.highway_network(hypothesis_in, self._highway_num_layers)

            with tf.variable_scope("prepro") as scope:
                pre = premise_in
                hyp = hypothesis_in
                for i in range(self._num_self_att_enc_layers):
                    with tf.variable_scope("attention_encoder_%s" % i, reuse=False):
                        pre_att = nn.self_attention(pre, prem_seq_lengths, func='tri_linear',
                                                    scope="premise_self_attention")
                        p = nn.fuse_gate(pre, pre_att, scope="premise_fuse_gate")
                        hyp_att = nn.self_attention(hyp, hyp_seq_lengths, func='tri_linear',
                                                    scope="hypothesis_self_attention")
                        h = nn.fuse_gate(hyp, hyp_att, scope="hypothesis_fuse_gate")

                        pre = p
                        hyp = h
                        nn.variable_summaries(p, "p_self_enc_summary_layer_{}".format(i))
                        nn.variable_summaries(h, "h_self_enc_summary_layer_{}".format(i))

            with tf.variable_scope("main") as scope:
                pre = p
                hyp = h

                with tf.variable_scope("interaction"):
                    pre_length = tf.shape(pre)[1]
                    hyp_length = tf.shape(hyp)[1]
                    pre_new = tf.tile(tf.expand_dims(pre, 2), [1, 1, hyp_length, 1])
                    hyp_new = tf.tile(tf.expand_dims(hyp, 1), [1, pre_length, 1, 1])
                    bi_att_mx = pre_new * hyp_new

                    # mask = tf.expand_dims(tf.sequence_mask(query_len, tf.shape(query)[1], dtype=tf.float32),
                    #                       axis=2) * \
                    #        tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
                    bi_att_mx = tf.layers.dropout(bi_att_mx, 1-dropout_keep_rate, training=is_training)

                with tf.variable_scope("dense_net"):
                    dim = bi_att_mx.get_shape().as_list()[-1]
                    act = tf.nn.relu if self._first_scale_down_layer_relu else None
                    fm = tf.contrib.layers.convolution2d(bi_att_mx,
                                                         int(dim * self._dense_net_first_scale_down_ratio),
                                                         self._first_scale_down_kernel, padding="SAME",
                                                         activation_fn=act)

                    fm = nn.dense_net_block(fm, self._dense_net_growth_rate, self._num_dense_net_layers,
                                            self._dense_net_kernel_size, scope="first_dense_net_block")
                    fm = nn.dense_net_transition_layer(fm, self._dense_net_transition_rate,
                                                       scope='second_transition_layer')
                    fm = nn.dense_net_block(fm, self._dense_net_growth_rate, self._num_dense_net_layers,
                                            self._dense_net_kernel_size, scope="second_dense_net_block")
                    fm = nn.dense_net_transition_layer(fm, self._dense_net_transition_rate,
                                                       scope='third_transition_layer')
                    fm = nn.dense_net_block(fm, self._dense_net_growth_rate, self._num_dense_net_layers,
                                            self._dense_net_kernel_size, scope="third_dense_net_block")

                    fm = nn.dense_net_transition_layer(fm, self._dense_net_transition_rate,
                                                       scope='fourth_transition_layer')

                    shape_list = list(fm.get_shape())
                    #print(shape_list)
                    premise_final = tf.reshape(fm, [-1, shape_list[1] * shape_list[2] * shape_list[3]])

            logits = tf.layers.dense(premise_final, self._num_classes, activation=None, name="arg",
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=self._initializer_range))

            tf.summary.histogram('logit_histogram', logits)

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
                #######l2 loss#################
                if self._l2_loss:
                    if self._sigmoid_growing_l2loss:
                        weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if
                                                  tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0') or tensor.name.endswith('filter:0')])
                        full_l2_step = tf.constant(self._weight_l2loss_step_full_reg, dtype=tf.int32, shape=[],
                                                   name='full_l2reg_step')
                        full_l2_ratio = tf.constant(self._l2_regularization_ratio, dtype=tf.float32, shape=[],
                                                    name='l2_regularization_ratio')
                        gs_flt = tf.cast(global_step, tf.float32)
                        half_l2_step_flt = tf.cast(full_l2_step / 2, tf.float32)

                        # (self.global_step - full_l2_step / 2)
                        # tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)
                        # l2loss_ratio = tf.sigmoid( tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)) * full_l2_ratio
                        l2loss_ratio = tf.sigmoid(((gs_flt - half_l2_step_flt) * 8) / half_l2_step_flt) * full_l2_ratio
                        tf.summary.scalar('l2loss_ratio', l2loss_ratio)
                        l2loss = weights_added * l2loss_ratio
                    else:
                        l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if
                                           tensor.name.endswith("weights:0") or tensor.name.endswith(
                                               'kernel:0')]) * tf.constant(self._l2_regularization_ratio,
                                                                           dtype='float', shape=[],
                                                                           name='l2_regularization_ratio')
                    tf.summary.scalar('l2loss', l2loss)
                ######diff loss###############################
                diffs = []
                for i in range(self._num_self_att_enc_layers):
                    for tensor in tf.trainable_variables():
                        #print(tensor.name)
                        if tensor.name == "diin/prepro/attention_encoder_{}/premise_self_attention/similar_mat/similar_func/arg/kernel:0".format(
                                i):
                            l_lg = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_self_attention/similar_mat/similar_func/arg/kernel:0".format(
                                i):
                            r_lg = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/premise_fuse_gate/lhs_1/kernel:0".format(i):
                            l_fg_lhs_1 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_fuse_gate/lhs_1/kernel:0".format(
                                i):
                            r_fg_lhs_1 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/premise_fuse_gate/rhs_1/kernel:0".format(i):
                            l_fg_rhs_1 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_fuse_gate/rhs_1/kernel:0".format(
                                i):
                            r_fg_rhs_1 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/premise_fuse_gate/lhs_2/kernel:0".format(i):
                            l_fg_lhs_2 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_fuse_gate/lhs_2/kernel:0".format(
                                i):
                            r_fg_lhs_2 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/premise_fuse_gate/rhs_2/kernel:0".format(i):
                            l_fg_rhs_2 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_fuse_gate/rhs_2/kernel:0".format(
                                i):
                            r_fg_rhs_2 = tensor

                        if tensor.name == "diin/prepro/attention_encoder_{}/premise_fuse_gate/lhs_3/kernel:0".format(
                                i):
                            l_fg_lhs_3 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_fuse_gate/lhs_3/kernel:0".format(
                                i):
                            r_fg_lhs_3 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/premise_fuse_gate/rhs_3/kernel:0".format(
                                i):
                            l_fg_rhs_3 = tensor
                        elif tensor.name == "diin/prepro/attention_encoder_{}/hypothesis_fuse_gate/rhs_3/kernel:0".format(
                                i):
                            r_fg_rhs_3 = tensor

                    diffs += [l_lg - r_lg, l_fg_lhs_1 - r_fg_lhs_1, l_fg_rhs_1 - r_fg_rhs_1, l_fg_lhs_2 - r_fg_lhs_2,
                              l_fg_rhs_2 - r_fg_rhs_2]
                    diffs += [l_fg_lhs_3 - r_fg_lhs_3, l_fg_rhs_3 - r_fg_rhs_3]
                diff_loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in diffs]) * tf.constant(
                    self._diff_penalty_loss_ratio, dtype='float', shape=[], name='diff_penalty_loss_ratio')
                tf.summary.scalar('diff_loss', diff_loss)
                ###############################
                output_dict['loss'] = loss + l2loss + diff_loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predictions)
                metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predictions)
                metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predictions)

                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict



