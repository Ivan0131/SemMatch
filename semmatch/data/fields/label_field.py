from semmatch.data.fields import Field
import tensorflow as tf


class LabelField(Field):
    def __init__(self, label, label_namespace="labels"):
        super().__init__()
        self.label = label
        self._label_namespace = label_namespace
        self._label_id = None

    def count_vocab(self, counter):
        if self._label_id is None:
            counter[self._label_namespace][self.label] += 1

    def index(self, vocab):
        if self._label_id is None:
            self._label_id = vocab.get_token_index(self.label, self._label_namespace)

    def to_example(self):
        if self._label_id is not None:
            features = dict()
            features[self._label_namespace] = tf.train.Feature(int64_list=tf.train.Int64List(value=[self._label_id]))
            return features

    def get_example(self):
        features = dict()
        features[self._label_namespace] = tf.FixedLenFeature([], tf.int64)
        return features
