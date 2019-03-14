import tensorflow as tf


def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim).
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights


def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    if sequence.get_shape().ndims == 3:
        sequence = tf.reduce_sum(sequence, -1)
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(populated, tf.float32)
    return length, mask