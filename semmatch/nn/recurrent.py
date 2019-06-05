import tensorflow as tf


def _base_bi_rnn(fw_cell, bw_cell, sequence, sequence_length=None, initial_state=(None, None)):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sequence, sequence_length,
                                                      initial_state_fw=initial_state[0],
                                                      initial_state_bw=initial_state[1], dtype=tf.float32)
    if isinstance(states[0], tuple):
        return tf.concat(outputs, axis=-1), (tf.concat([s.c for s in states], axis=-1), tf.concat([s.h for s in states], axis=-1))
    else:
        return tf.concat(outputs, axis=-1), tf.concat(states, axis=-1)


def bi_lstm(seq, hidden_units, seq_len=None, initial_state=(None, None),
            initializer=tf.contrib.layers.xavier_initializer(), name='bi_lstm'):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, initializer=initializer)
        bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, initializer=initializer)
        return _base_bi_rnn(fw_cell, bw_cell, seq, seq_len, initial_state)


def bi_gru(seq, hidden_units, seq_len=None, initial_state=(None, None),
           initializer=tf.contrib.layers.xavier_initializer(), name="bi_grm"):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, kernel_initializer=initializer)
        bw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, kernel_initializer=initializer)
        return _base_bi_rnn(fw_cell, bw_cell, seq, seq_len, initial_state)


def _base_rnn(fw_cell, sequence, sequence_length=None, initial_state=None):
    outputs, states = tf.nn.dynamic_rnn(fw_cell, sequence, sequence_length,
                                        initial_state=initial_state, dtype=tf.float32)
    return outputs, states


def lstm(seq, hidden_units, seq_len=None, initial_state=None,
         initializer=tf.contrib.layers.xavier_initializer(), name="lstm"):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, initializer=initializer)
        return _base_rnn(fw_cell, seq, seq_len, initial_state)


def gru(seq, hidden_units, seq_len=None, initial_state=None,
        initializer=tf.contrib.layers.xavier_initializer(), name="gru"):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, kernel_initializer=initializer)
        return _base_rnn(fw_cell, seq, seq_len, initial_state)


def _base_rnn_network(seq, rnn_func, num_layer, hidden_units, seq_len=None, initial_state=None,
                      initializer=tf.contrib.layers.xavier_initializer(), name="base_rnn_network"):
    outputs = []
    states = []
    with tf.variable_scope(name):
        for i in range(num_layer):
            with tf.variable_scope("layer_%s" % i):
                seq, c1 = rnn_func(seq, hidden_units, seq_len=seq_len, initial_state=initial_state,
                                   initializer=initializer)
                outputs.append(seq)
                states.append(c1)
        return outputs, states


def bi_gru_network(seq, num_layer, hidden_units, seq_len=None, initial_state=(None, None),
                   initializer=tf.contrib.layers.xavier_initializer(), name="bi_gru_network"):
    return _base_rnn_network(seq, bi_gru, num_layer, hidden_units, seq_len, initial_state,
                             initializer, name)


def bi_lstm_network(seq, num_layer, hidden_units, seq_len=None, initial_state=(None, None),
                    initializer=tf.contrib.layers.xavier_initializer(), name="bi_lstm_network"):
    return _base_rnn_network(seq, bi_lstm, num_layer, hidden_units, seq_len, initial_state,
                             initializer, name)


def gru_network(seq, num_layer, hidden_units, seq_len=None, initial_state=None,
                initializer=tf.contrib.layers.xavier_initializer(), name="gru_network"):
    return _base_rnn_network(seq, gru, num_layer, hidden_units, seq_len, initial_state,
                             initializer, name)


def lstm_network(seq, num_layer, hidden_units, seq_len=None, initial_state=None,
                 initializer=tf.contrib.layers.xavier_initializer(), name="lstm_network"):
    return _base_rnn_network(seq, lstm, num_layer, hidden_units, seq_len, initial_state,
                             initializer, name)
