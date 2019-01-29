import tensorflow as tf



def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask


def biLSTM(inputs, dim, seq_len, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states