from typing import Dict, List
import itertools
from semmatch.data.tokenizers import Token
from semmatch.data import Vocabulary
from semmatch.data.token_indexers.token_indexer import TokenIndexer
import tensorflow as tf
from semmatch.utils import register


@register.register_subclass("token_indexer", "single_id")
class SingleIdTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Parameters
    ----------
    namespace : ``str``, optional (default=``tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will call ``token.lower()`` before getting an index for the token from the
        vocabulary.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'tokens',
                 lowercase_tokens: bool = False,
                 max_length: int = None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        super().__init__(namespace, max_length)
        self.lowercase_tokens = lowercase_tokens
        self._max_length = max_length
        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If `text_id` is set on the token (e.g., if we're using some kind of hash-based word
        # encoding), we will not be using the vocab for this token.
        if getattr(token, 'text_id', None) is None:
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            counter[self._namespace][text] += 1

    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            if getattr(token, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead.
                indices.append(token.text_id)
            else:
                text = token.text
                if self.lowercase_tokens:
                    text = text.lower()
                indices.append(vocabulary.get_token_index(text, self._namespace))
        return {self._namespace: indices}

    def pad_token_sequence(self, tokens, max_length):
        if len(tokens) > max_length:
            if len(self._end_tokens):
                tokens = tokens[:max_length-len(self._end_tokens)] + tokens[-len(self._end_tokens):]
            else:
                tokens = tokens[:max_length]
            return tokens
        padding_token = self.get_padding_values()
        while len(tokens) < max_length:
            tokens.append(padding_token)
        return tokens

    def to_raw_data(self, token_indexers):
        if self._max_length:
            token_indexers = self.pad_token_sequence(token_indexers, self._max_length)
        feature_list = [[token] for token in token_indexers]
        return feature_list

    def to_example(self, token_indexers):
        if self._max_length:
            token_indexers = self.pad_token_sequence(token_indexers, self._max_length)
        feature_list = [tf.train.Feature(int64_list=tf.train.Int64List(value=[token])) for token in token_indexers]
        return tf.train.FeatureList(feature=feature_list)

    def get_example(self):
        return tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)

    def get_padded_shapes(self):
        return [self._max_length]

    def get_padding_values(self):
        return 0

    def get_tf_shapes_and_dtypes(self):
        return {'dtype': tf.int32, 'shape': (None, self._max_length)}


