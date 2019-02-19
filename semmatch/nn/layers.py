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
