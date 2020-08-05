from semmatch.data.fields import Field
import tensorflow as tf


class NumericalField(Field):
    def __init__(self, value, namespace="numerical"):
        super().__init__()
        self._value = float(value)
        self._numerical_namespace = namespace

    def count_vocab(self, counter):
        pass

    def index(self, vocab):
        pass

    def to_example(self):
        if self._value is not None:
            features = dict()
            features[self._numerical_namespace] = tf.train.Feature(float_list=tf.train.FloatList(value=[self._value]))
            return features

    def get_example(self):
        features = dict()
        features[self._numerical_namespace] = tf.FixedLenFeature([], tf.float32)
        return features

    def get_padded_shapes(self):
        padded_shapes = dict()
        padded_shapes[self._numerical_namespace] = []
        return padded_shapes

    def get_padding_values(self):
        padding_values = dict()
        padding_values[self._numerical_namespace] = 0.0
        return padding_values

    def get_tf_shapes_and_dtypes(self):
        shapes_and_dtypes = dict()
        shapes_and_dtypes[self._numerical_namespace] = {'dtype': tf.float32, 'shape': (None)}
        return shapes_and_dtypes
