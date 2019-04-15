import tensorflow as tf


def highway_layer(arg, output_size=None, initializer_range=0.02, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        if output_size is not None:
            d = output_size
        else:
            d = arg.get_shape()[-1]
        trans = tf.layers.dense(arg, d, activation=tf.nn.relu, name="trans", kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        gate = tf.layers.dense(arg, d, activation=tf.nn.sigmoid, name="gate", kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        if d != arg.get_shape()[-1]:
            arg = tf.layers.dense(arg, d, activation=tf.nn.sigmoid, name="arg", kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, output_size=None, initializer_range=0.02, dropout_rate=None, is_trainging = True, scope=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, output_size=output_size, initializer_range=initializer_range,
                                scope="highway_laye_%s" % layer_idx)
            if dropout_rate:
                cur = tf.layers.dropout(cur, dropout_rate, training=is_trainging)
            prev = cur
        return cur


def dense_net_block(feature_map, growth_rate, num_layers, kernel_size, padding="SAME", act=tf.nn.relu,
                    scope=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        #dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(num_layers):
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)

        print("dense net block out shape")
        print(features.get_shape().as_list())
        return features


def dense_net_transition_layer(feature_map, transition_rate, scope=None):
    with tf.variable_scope(scope or "transition_layer"):
        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn=None)

        feature_map = tf.nn.max_pool(feature_map, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map