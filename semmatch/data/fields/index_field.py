from semmatch.data.fields import Field
import tensorflow as tf


class IndexField(Field):
    def __init__(self, index, index_namespace="index"):
        super().__init__()
        self._index_id = str(index)
        self._index_namespace = index_namespace

    def count_vocab(self, counter):
        pass

    def index(self, vocab):
        pass

    def to_example(self):
        if self._index_id is not None:
            features = dict()
            features[self._index_namespace] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(self._index_id, 'utf-8')]))
            return features

    def to_raw_data(self):
        if self._index_id is not None:
            features = dict()
            features[self._index_namespace] = [self._index_id]
            return features

    def get_example(self):
        features = dict()
        features[self._index_namespace] = tf.FixedLenFeature([], tf.string)
        return features

    def get_padded_shapes(self):
        padded_shapes = dict()
        padded_shapes[self._index_namespace] = []
        return padded_shapes

    def get_padding_values(self):
        padding_values = dict()
        padding_values[self._index_namespace] = ""
        return padding_values

    def get_tf_shapes_and_dtypes(self):
        shapes_and_dtypes = dict()
        shapes_and_dtypes[self._index_namespace] = {'dtype': tf.string, 'shape': (None)}
        return shapes_and_dtypes
