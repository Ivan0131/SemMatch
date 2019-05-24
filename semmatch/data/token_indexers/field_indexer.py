from typing import Dict, List
import tensorflow as tf
from semmatch.data.tokenizers import Token
from semmatch.data import Vocabulary
from semmatch.data.token_indexers.token_indexer import TokenIndexer
from semmatch.utils import register


@register.register_subclass("token_indexer", "fields")
class FieldIndexer(TokenIndexer):
    def __init__(self, namespace: str = '%s_tags', max_length: int = None, field_name: str = 'exact_match') -> None:
        super().__init__(namespace, max_length)
        if '%s' in namespace:
            self._namespace = namespace % field_name
        else:
            self._namespace = namespace
        self._field_name = field_name

    def get_field_name(self):
        return self._field_name

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        tag = getattr(token, self._field_name, None)
        if tag is None:
            tag = 'NONE'
        counter[self._namespace][tag] += 1

    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary):
        tags: List[str] = []

        for token in tokens:
            tag = getattr(token, self._field_name, None)
            if tag is None:
                tag = 'NONE'

            tags.append(tag)

        return {self._namespace: [vocabulary.get_token_index(tag, self._namespace) for tag in tags]}

    def get_padding_values(self) -> int:
        return 0

    def pad_token_sequence(self, tokens, max_length):
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            return tokens
        padding_token = self.get_padding_values()
        while len(tokens) < max_length:
            tokens.append(padding_token)
        return tokens

    def to_example(self, token_indexers):
        if self._max_length:
            token_indexers = self.pad_token_sequence(token_indexers, self._max_length)
        feature_list = [tf.train.Feature(int64_list=tf.train.Int64List(value=[token])) for token in token_indexers]
        return tf.train.FeatureList(feature=feature_list)

    def get_example(self):
        return tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)

    def get_padded_shapes(self):
        return [self._max_length]

    def get_tf_shapes_and_dtypes(self):
        return {'dtype': tf.int32, 'shape': (None, self._max_length)}



