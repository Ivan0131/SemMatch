import tensorflow as tf


def similarity_function(query, key, func=None, initializer_range=0.02, scope="similar_func"):
    with tf.variable_scope(scope):
        if func is None:
            func = "dot"
        if func == "dot":
            sim_mat = tf.matmul(query, key, transpose_b=True)
        elif func == "tri_linear":
            query_length = tf.shape(query)[1]
            key_length = tf.shape(key)[1]
            query_new = tf.tile(tf.expand_dims(query, 2), [1, 1, key_length, 1])
            key_new = tf.tile(tf.expand_dims(query, 1), [1, query_length, 1, 1])
            new_mat = tf.concat([query_new, key_new, query_new * key_new], axis=3)
            sim_mat = tf.layers.dense(new_mat, 1, activation=None, name="arg",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
            sim_mat = tf.squeeze(sim_mat, axis=3)
        else:
            raise Exception("%s similarity function is not found." % func)
        return sim_mat


def _get_similar_matrix(query, key, query_len=None, key_len=None, func='dot',
                        initializer_range=0.02, scope="similar_mat"):
    with tf.variable_scope(scope):
        sim_mat = similarity_function(query, key, func=func, initializer_range=initializer_range)
        assert key_len is not None
        if query_len is not None:
            mask = tf.expand_dims(tf.sequence_mask(query_len, tf.shape(query)[1], dtype=tf.float32), axis=2) * \
                       tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
        else:
            mask = tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
        sim_mat = sim_mat + (1. - mask) * tf.float32.min
        return sim_mat


def bi_attention(query, key, query_len, key_len, func='dot', name='bi_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(query, key, query_len, key_len, func=func)

        # Context-to-query Attention in the paper
        query_key_prob = tf.nn.softmax(sim_mat)
        query_key_attention = tf.matmul(query_key_prob, key)

        # Query-to-context Attention in the paper
        key_query_prob = tf.nn.softmax(tf.reduce_max(sim_mat, axis=-1))
        key_query_attention = tf.matmul(tf.expand_dims(key_query_prob, axis=1), query)
        key_query_attention = tf.tile(key_query_attention, [1, tf.shape(query)[1], 1])

        return query_key_attention, key_query_attention


def bi_uni_attention(query, key, query_len, key_len, func="dot", name='bi_uni_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(query, key, query_len, key_len, func=func)

        query_key_prob = tf.nn.softmax(sim_mat)
        query_key_attention = tf.matmul(query_key_prob, key)

        key_query_prob = tf.nn.softmax(tf.transpose(sim_mat, perm=[0, 2, 1]))
        key_query_attention = tf.matmul(key_query_prob, query)

        return query_key_attention, key_query_attention


def uni_attention(query, key, key_len, value=None, func="dot", name='uni_attention'):
    with tf.variable_scope(name):
        sim_mat = _get_similar_matrix(similarity_function, query, key, key_len=key_len, func=func)
        sim_prob = tf.nn.softmax(sim_mat)
        if value is not None:
            return tf.matmul(sim_prob, value)
        else:
            return tf.matmul(sim_prob, key)


def self_attention(query, query_len, func='dot', initializer_range=0.02, scope='self_attention'):
    with tf.variable_scope(scope):
        sim_mat = _get_similar_matrix(query, query, query_len=query_len, key_len=query_len, func=func,
                                      initializer_range=initializer_range)
        sim_prob = tf.nn.softmax(sim_mat)
        return tf.matmul(sim_prob, query)


def fuse_gate(lhs, rhs, initializer_range=0.02, self_att_fuse_gate_residual_conn=True, two_gate_fuse_gate=True,
              self_att_fuse_gate_relu_z=False, scope="fuse_gate"):
    with tf.variable_scope(scope):
        dim = lhs.get_shape().as_list()[-1]

        lhs_1 = tf.layers.dense(lhs, dim, activation=None, name="lhs_1",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        rhs_1 = tf.layers.dense(rhs, dim, activation=None, name="rhs_1",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))

        if self_att_fuse_gate_residual_conn and self_att_fuse_gate_relu_z:
            z = tf.nn.relu(lhs_1+rhs_1)
        else:
            z = tf.tanh(lhs_1+rhs_1)

        lhs_2 = tf.layers.dense(lhs, dim, activation=tf.nn.sigmoid, name="lhs_2",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        rhs_2 = tf.layers.dense(rhs, dim, activation=tf.nn.sigmoid, name="rhs_2",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        f = tf.sigmoid(lhs_2 + rhs_2)

        if two_gate_fuse_gate:
            lhs_3 = tf.layers.dense(lhs, dim, activation=tf.nn.sigmoid, name="lhs_3",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
            rhs_3 = tf.layers.dense(rhs, dim, activation=tf.nn.sigmoid, name="rhs_3",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
            f2 = tf.sigmoid(lhs_3 + rhs_3)
            out = f * lhs + f2 * z
        else:
            out = f * lhs + (1 - f) * z

        return out



if __name__ == "__main__":
    q = tf.ones((1, 4, 10))
    k = tf.ones((1, 6, 10))
    q_len = tf.constant([4])
    k_len = tf.constant([6])
    q2q = self_attention(q, q_len)
    q2k, k2q = bi_uni_attention(q, k, q_len, k_len)
    sess = tf.InteractiveSession()
    print(sess.run(q2q).shape)
    print(sess.run(q2k).shape)
    print(sess.run(k2q).shape)


