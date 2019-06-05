import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.embeddings.encoders import Bert
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn

#DRr-Net: Dynamic Re-read Network for Sentence Semantic Matching
#http://staff.ustc.edu.cn/~cheneh/paper_pdf/2019/Kun-Zhang-AAAI.pdf
@register.register_subclass('model', 'drr_net')
class DRr_Net(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 256, dropout_rate: float = 0.5, num_rnn_layer: int = 3, reread_length: int = 6,
                 char_filter_sizes: str = "5", char_filter_channel_dims: str = "100", beta: float = 1e8,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 model_name: str = 'bilstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate
        self._char_filter_size = list(map(int, char_filter_sizes.split(',')))
        self._char_filter_channel_dims = list(map(int, char_filter_channel_dims.split(',')))
        self._num_rnn_layer= num_rnn_layer
        self._reread_length = reread_length
        self._beta = beta
        self._initializer = initializer

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            #########Word Embedding####################
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
                    conv_pre = nn.multi_conv1d_max(premise_chars, self._char_filter_size,
                                                   self._char_filter_channel_dims,
                                                   "VALID", is_training, self._dropout_rate, scope='conv')
                    scope.reuse_variables()
                    conv_hyp = nn.multi_conv1d_max(hypothesis_chars, self._char_filter_size,
                                                   self._char_filter_channel_dims,
                                                   "VALID", is_training, self._dropout_rate, scope='conv')
                    # conv_pre = tf.reshape(conv_pre, [-1, self.sequence_length, config.char_out_size])
                    # conv_hyp = tf.reshape(conv_hyp, [-1, self.sequence_length, config.char_out_size])

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

            premise_in = nn.highway_network(premise_in, 2, output_size=self._hidden_dim, dropout_rate=self._dropout_rate, is_trainging=is_training,
                                            scope="premise_highway")
            hypothesis_in = nn.highway_network(hypothesis_in, 2, output_size=self._hidden_dim, dropout_rate=self._dropout_rate, is_trainging=is_training,
                                            scope="hypothesis_highway")

            ########Attention Stack-GRU################
            def gru_network(input, input_len, name="gru_network"):
                with tf.variable_scope(name):
                    gru_input = input
                    for i in range(self._num_rnn_layer):
                        with tf.variable_scope("layer_%s" % i):
                            seq, c1 = nn.gru(gru_input, self._hidden_dim, seq_len=input_len,
                                             initializer=self._initializer)
                            gru_input = tf.concat([gru_input, seq], axis=2)
                return gru_input

            premise_gru = gru_network(premise_in, prem_seq_lengths, name='premise_gru_network')
            hypothesis_gru = gru_network(hypothesis_in, hyp_seq_lengths, name='hypothesis_gru_network')

            premise_gru = premise_gru * prem_mask
            hypothesis_gru = hypothesis_gru * hyp_mask
            #########
            premise_att = nn.attention_pool(premise_gru, self._hidden_dim, seq_len=prem_seq_lengths,
                                            initializer=self._initializer,
                                            name='premise_attention_pool')
            hypothesis_att = nn.attention_pool(hypothesis_gru, self._hidden_dim, seq_len=hyp_seq_lengths,
                                               initializer=self._initializer,
                                               name='hypothesis_attention_pool')
            ############Dynamic Re-read Mechanism################

            def dynamic_reread(h_seq_a, h_a, h_b, h_a_len, name="dymanic_reread"):
                with tf.variable_scope(name):
                    h_a_pre = h_a
                    # h_a_pre = nn.highway_layer(h_a, self._hidden_dim, initializer=self._initializer,
                    #                            scope="h_a_pre_highway")
                    # h_seq_a = nn.highway_layer(h_seq_a, self._hidden_dim, initializer=self._initializer,
                    #                            scope="h_seq_a_highway")
                    # h_b = nn.highway_layer(h_b, self._hidden_dim, initializer=self._initializer,
                    #                        scope="h_b_highway")
                    #####
                    w_d = tf.get_variable("w_d_weights", (h_seq_a.shape[-1].value, h_a_pre.shape[-1].value),
                                          initializer=self._initializer)
                    u_d = tf.get_variable("u_d_weights", (h_a_pre.shape[-1].value, h_a_pre.shape[-1].value),
                                          initializer=self._initializer)
                    m_d = tf.get_variable("m_d_weights", (h_b.shape[-1].value, h_a_pre.shape[-1].value),
                                          initializer=self._initializer)
                    omega_d = tf.get_variable("omega_d_weights", (h_a_pre.shape[-1].value, 1),
                                              initializer=self._initializer)
                    ##########
                    m_d_h_b = tf.tensordot(h_b, m_d, axes=[-1, 0])
                    h_seq_a_w_d = tf.tensordot(h_seq_a, w_d, axes=[-1, 0])

                    if h_a_len is not None:
                        mask = tf.expand_dims(tf.sequence_mask(h_a_len, tf.shape(h_seq_a)[1], dtype=tf.float32), axis=2)
                    else:
                        mask = None
                    gru_cell = tf.nn.rnn_cell.GRUCell(h_a_pre.shape[-1].value, kernel_initializer=self._initializer)

                    for i in range(self._reread_length):
                        u_d_h_a_pre = tf.tensordot(h_a_pre, u_d, axes=[-1, 0])
                        m_a = tf.nn.tanh(h_seq_a_w_d+tf.expand_dims(m_d_h_b + u_d_h_a_pre, 1))
                        m_a = tf.tensordot(m_a, omega_d, axes=[-1, 0])
                        if mask is not None:
                            m_a = m_a + (1. - mask) * tf.float32.min
                        alpha = tf.nn.softmax(self._beta*m_a, axis=1)
                        alpha = tf.reduce_sum(alpha*h_seq_a, axis=1)
                        gru_output, gru_state = gru_cell(alpha, h_a_pre)
                        h_a_pre = gru_state
                    return gru_output
            premise_v = dynamic_reread(premise_gru, premise_att, hypothesis_att, prem_seq_lengths,
                                       name='premise_dynamic_reread')
            hypothesis_v = dynamic_reread(hypothesis_gru, hypothesis_att, premise_att, hyp_seq_lengths,
                                       name='hypothesis_dynamic_reread')

            ########label prediction##############

            h = tf.concat([premise_att, hypothesis_att, hypothesis_att*premise_att, hypothesis_att-premise_att], axis=-1)
            v = tf.concat([premise_v, hypothesis_v, hypothesis_v*premise_v, hypothesis_v-premise_v], axis=-1)

            # h MLP layer
            h_mlp = tf.layers.dense(h, self._hidden_dim, activation=tf.nn.relu, kernel_initializer=self._initializer, name='h_fc1')
            # Dropout applied to classifier
            h_drop = tf.layers.dropout(h_mlp, self._dropout_rate, training=is_training)
            # Get prediction
            h_logits = tf.layers.dense(h_drop, self._num_classes, activation=None,
                                       kernel_initializer=self._initializer, name='h_logits')

            p_h = tf.nn.softmax(h_logits)

            # # MLP layer
            v_mlp = tf.layers.dense(v, self._hidden_dim, activation=tf.nn.relu, kernel_initializer=self._initializer, name='v_fc1')
            # Dropout applied to classifier
            v_drop = tf.layers.dropout(v_mlp, self._dropout_rate, training=is_training)
            # Get prediction
            v_logits = tf.layers.dense(v_drop, self._num_classes, activation=None,
                                       kernel_initializer=self._initializer, name='v_logits')

            p_v = tf.nn.softmax(v_logits)
            ####
            alpha_h = tf.layers.dense(h, 1, activation=tf.nn.sigmoid, kernel_initializer=self._initializer,
                                      bias_initializer=tf.zeros_initializer())
            alpha_v = tf.layers.dense(v, 1, activation=tf.nn.sigmoid, kernel_initializer=self._initializer,
                                      bias_initializer=tf.zeros_initializer())
            # # h MLP layer
            fuse_mlp = tf.layers.dense(alpha_h*h+alpha_v*v, self._hidden_dim, activation=tf.nn.relu,
                                       kernel_initializer=self._initializer, name='fuse_fc1')
            # Dropout applied to classifier
            fuse_drop = tf.layers.dropout(fuse_mlp, self._dropout_rate, training=is_training)
            #Get prediction
            output_dict = self._make_output(fuse_drop, params)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels_embedding = features_embedding['label/labels']
                labels = features['label/labels']

                h_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_embedding, logits=h_logits))
                v_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_embedding, logits=v_logits))
                fuse_loss = self._make_loss(labels=labels_embedding, logits=output_dict['logits'], params=params)

                output_dict['loss'] = v_loss + h_loss + fuse_loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=output_dict['predictions'])
                metrics['precision'] = tf.metrics.precision(labels=labels, predictions=output_dict['predictions'])
                metrics['recall'] = tf.metrics.recall(labels=labels, predictions=output_dict['predictions'])
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict
