from semmatch.modules.embeddings.encoders import Encoder
from semmatch.utils import register
import tensorflow as tf


@register.register_subclass("encoder", 'multilabel')
class MultiLabelEncoder(Encoder):
    def __init__(self, dtype=tf.float32, vocab_namespace='labels', encoder_name='multi_label_encoder'):
        super().__init__(encoder_name=encoder_name, vocab_namespace=vocab_namespace)
        self._dtype = dtype
        self._vocab_namespace = vocab_namespace

    def forward(self, features, labels, mode, params):
        outputs = dict()
        for (feature_key, feature) in features.items():
            feature_namespace = feature_key.split("/")[1].strip()
            if feature_namespace == self._vocab_namespace:
                with tf.variable_scope(self._encoder_name):
                    output = tf.cast(feature, dtype=self._dtype)
                outputs[feature_key] = output
        return outputs

    @classmethod
    def init_from_params(cls, params, vocab):
        vocab_namespace = params.pop('namespace', 'labels')
        encoder_name = params.pop("encoder_name", "multi_label_encoder")

        params.assert_empty(cls.__name__)
        return cls(vocab_namespace=vocab_namespace,
                   encoder_name=encoder_name)

