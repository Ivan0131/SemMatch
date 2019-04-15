import tensorflow as tf


def _base_bi_rnn(fw_cell, bw_cell, sequence, sequence_length=None):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sequence, sequence_length, dtype=tf.float32)
    if isinstance(states[0], tuple):
        return tf.concat(outputs, axis=-1), (tf.concat([s.c for s in states], axis=-1), tf.concat([s.h for s in states], axis=-1))
    else:
        return tf.concat(outputs, axis=-1), tf.concat(states, axis=-1)


def bi_lstm(seq, hidden_units, seq_len=None, name='bi_lstm'):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
        bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
        return _base_bi_rnn(fw_cell, bw_cell, seq, seq_len)


def bi_gru(seq, hidden_units, seq_len=None, name="bi_grm"):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        bw_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        return _base_bi_rnn(fw_cell, bw_cell, seq, seq_len)


def _base_rnn(fw_cell, sequence, sequence_length=None):
    outputs, states = tf.nn.dynamic_rnn(fw_cell, sequence, sequence_length, dtype=tf.float32)
    return outputs, states


def lstm(seq, hidden_units, seq_len=None, name="lstm"):
    with tf.variable_scope(name):
        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
        return _base_rnn(fw_cell, seq, seq_len)
