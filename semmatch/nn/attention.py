import tensorflow as tf


def dot_similarity_function(query, key):
    sim_mat = tf.matmul(query, key, transpose_b=True)
    return sim_mat

def _get_similar_matrix(similarity_function, query, key, query_len=None, key_len=None):
    sim_mat = similarity_function(query, key)
    assert key_len is not None
    if query_len is not None:
        mask = tf.expand_dims(tf.sequence_mask(query_len, tf.shape(query)[1], dtype=tf.float32), axis=2) * \
                   tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
    else:
        mask = tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
    sim_mat = sim_mat + (1. - mask) * tf.float32.min
    return sim_mat


def bi_attention(query, key, query_len, key_len, similarity_function, name='bi_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(similarity_function, query, key, query_len, key_len)

        # Context-to-query Attention in the paper
        query_key_prob = tf.nn.softmax(sim_mat)
        query_key_attention = tf.matmul(query_key_prob, key)

        # Query-to-context Attention in the paper
        key_query_prob = tf.nn.softmax(tf.reduce_max(sim_mat, axis=-1))
        key_query_attention = tf.matmul(tf.expand_dims(key_query_prob, axis=1), query)
        key_query_attention = tf.tile(key_query_attention, [1, tf.shape(query)[1], 1])

        return query_key_attention, key_query_attention


def bi_uni_attention(query, key, query_len, key_len, similarity_function, name='bi_uni_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(similarity_function, query, key, query_len, key_len)

        query_key_prob = tf.nn.softmax(sim_mat)
        query_key_attention = tf.matmul(query_key_prob, key)

        key_query_prob = tf.nn.softmax(tf.transpose(sim_mat, perm=[0, 2, 1]))
        key_query_attention = tf.matmul(key_query_prob, query)

        return query_key_attention, key_query_attention


def uni_attention(query, key, key_len, similarity_function, value=None, name='uni_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(similarity_function, query, key, key_len=key_len)
        sim_prob = tf.nn.softmax(sim_mat)
        if value is not None:
            return tf.matmul(sim_prob, value)
        else:
            return tf.matmul(sim_prob, key)


def self_attention(query, query_len, similarity_function, name='self_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(similarity_function, query, query, key_len=query_len)
        sim_prob = tf.nn.softmax(sim_mat)
        return tf.matmul(sim_prob, query)
