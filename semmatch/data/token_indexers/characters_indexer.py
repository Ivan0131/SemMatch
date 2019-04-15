from typing import Dict, List
import itertools
from semmatch.data.tokenizers import Token
from semmatch.data import Vocabulary
from semmatch.data.token_indexers.token_indexer import TokenIndexer
from semmatch.data.tokenizers.character_tokenizer import CharacterTokenizer
import tensorflow as tf
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
import numpy as np


@register.register_subclass("token_indexer", "chars")
class CharactersIndexer(TokenIndexer):
    def __init__(self,
                 namespace: str = 'chars',
                 max_length: int = None,
                 character_tokenizer: CharacterTokenizer = CharacterTokenizer(),
                 max_word_length: int = 16,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        super().__init__(namespace)
        self._max_length = max_length
        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self._character_tokenizer = character_tokenizer
        self._max_word_length = max_word_length

    def set_max_length(self, max_length):
        self._max_length = max_length

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigureError('CharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            if getattr(character, 'text_id', None) is None:
                counter[self._namespace][character.text] += 1

    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary):
        indices = []
        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            token_indices = np.zeros(self._max_word_length, dtype=np.int64)
            if token.text is None:
                raise ConfigureError('TokenCharactersIndexer needs a tokenizer that retains text')
            for character_idx, character in enumerate(self._character_tokenizer.tokenize(token.text)):
                if character_idx >= self._max_word_length:
                    break
                else:
                    if getattr(character, 'text_id', None) is not None:
                        # `text_id` being set on the token means that we aren't using the vocab, we just
                        # use this id instead.
                        index = character.text_id
                    else:
                        index = vocabulary.get_token_index(character.text, self._namespace)
                    token_indices[character_idx] = index
            indices.append(token_indices)
        return {self._namespace: indices}

    def get_padding_values(self) -> int:
        return 0

    def pad_token_sequence(self, tokens, max_length):
        if len(tokens) >= max_length:
            tokens = tokens[:max_length]
            return tokens
        padding_tokens = [self.get_padding_values()] * self._max_word_length
        padding_tokens = [padding_tokens] * (max_length - len(tokens))
        padding_tokens = np.array(padding_tokens, dtype=np.int64)
        tokens.extend(padding_tokens)
        #tokens = np.concatenate(tokens, axis=0)
        return tokens

    def to_example(self, token_indexers):
        if self._max_length:
            token_indexers = self.pad_token_sequence(token_indexers, self._max_length)
        input_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=token_indexers[i]))
            for i in range(len(token_indexers))]
        return tf.train.FeatureList(feature=input_features)

    def get_example(self):
        return tf.FixedLenSequenceFeature([self._max_word_length], tf.int64, allow_missing=True)

    def get_padded_shapes(self):
        return [self._max_length, self._max_word_length]

    def get_tf_shapes_and_dtypes(self):
        return {'dtype': tf.int32, 'shape': (None, None, self._max_word_length)}



