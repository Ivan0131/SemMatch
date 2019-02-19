from typing import Dict, MutableMapping, Mapping
from semmatch.data.fields import field
from semmatch.data import vocabulary
import tensorflow as tf


class Instance(Mapping[str, field.Field]):
    def __init__(self, fields: MutableMapping[str, field.Field]) -> None:
        self.fields = fields
        self.indexed = False

    def __getitem__(self, key: str) -> field.Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def index_fields(self, vocab) -> None:
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)

    def count_vocab(self, counter) -> None:
        for field in self.fields.values():
            field.count_vocab(counter)

    def to_example(self):
        """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
        instance_features = {}
        for field_name, field in self.fields.items():
            field_features = field.to_example()
            if not field_features:
                raise ValueError("%s index is not generated" % (field_name, ))
            if isinstance(field_features, Dict):
                for (feature_name, feature) in field_features.items():
                    instance_features[field_name+"/"+feature_name] = feature
            else:
                raise ValueError("The field %s to example is error." % (field_name, ))

        return tf.train.Example(features=tf.train.Features(feature=instance_features))

    def get_example_features(self):
        instance_features = {}
        for field_name, field in self.fields.items():
            field_features = field.get_example()
            if not field_features:
                raise ValueError("%s index is not generated" % (field_name,))
            if isinstance(field_features, Dict):
                for (feature_name, feature) in field_features.items():
                    instance_features[field_name + "/" + feature_name] = feature
            else:
                raise ValueError("The field %s get example features is error." % (field_name,))
                #instance_features[field_name] = field_features
        return instance_features

    def get_padded_shapes(self):
        instance_padded_shapes = {}
        for field_name, field in self.fields.items():
            field_padded_shapes = field.get_padded_shapes()
            if not field_padded_shapes:
                raise ValueError("%s index is not generated" % (field_name,))
            if isinstance(field_padded_shapes, Dict):
                for (feature_name, padded_shape) in field_padded_shapes.items():
                    instance_padded_shapes[field_name + "/" + feature_name] = padded_shape
            else:
                raise ValueError("The field %s get padded_shapes is error." % (field_name,))
                #instance_features[field_name] = field_features
        return instance_padded_shapes

    def get_padding_values(self):
        instance_padding_values = {}
        for field_name, field in self.fields.items():
            field_padding_values = field.get_padding_values()
            if not field_padding_values:
                raise ValueError("%s index is not generated" % (field_name,))
            if isinstance(field_padding_values, Dict):
                for (feature_name, padding_values) in field_padding_values.items():
                    instance_padding_values[field_name + "/" + feature_name] = padding_values
            else:
                raise ValueError("The field %s get padding values is error." % (field_name,))
                # instance_features[field_name] = field_features
        return instance_padding_values
