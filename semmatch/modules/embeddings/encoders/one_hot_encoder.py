from semmatch.modules.embeddings.encoders import Encoder
from semmatch.utils import register
import tensorflow as tf


@register.register_subclass("encoder", 'one_hot')
class OneHotEncoder(Encoder):
    def __init__(self, n_values, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32, vocab_namespace='labels', encoder_name='one_hot_encoder'):
        super().__init__(encoder_name=encoder_name, vocab_namespace=vocab_namespace)
        self._n_values = n_values
        self._on_value = on_value
        self._off_value = off_value
        self._axis = axis
        self._dtype = dtype
        self._vocab_namespace = vocab_namespace

    def forward(self, features, labels, mode, params):
        outputs = dict()
        for (feature_key, feature) in features.items():
            if '/' not in feature_key:
                continue
            feature_namespace = feature_key.split("/")[1].strip()
            if feature_namespace == self._vocab_namespace:
                with tf.variable_scope(self._encoder_name):
                    output = tf.one_hot(feature, depth=self._n_values, on_value=self._on_value, off_value=self._off_value,
                                        axis=self._axis, dtype=self._dtype)
                outputs[feature_key] = output
        return outputs

    @classmethod
    def init_from_params(cls, params, vocab):
        vocab_namespace = params.pop('namespace', 'labels')
        n_values = params.pop_int('n_values', None)
        if n_values is None:
            n_values = vocab.get_vocab_size(vocab_namespace)
        on_value = params.pop_float("on_value", 1.0)
        off_value = params.pop_float("off_value", 0.0)
        axis = params.pop_int("axis", -1)
        encoder_name = params.pop("encoder_name", "one_hot_encoder")

        params.assert_empty(cls.__name__)
        return cls(n_values=n_values,
                   on_value=on_value,
                   off_value=off_value,
                   axis=axis, vocab_namespace=vocab_namespace,
                   encoder_name=encoder_name)

