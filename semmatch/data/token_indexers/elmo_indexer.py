from typing import Dict, List
import itertools
from semmatch.data.tokenizers import Token
from semmatch.data import Vocabulary
from semmatch.data.token_indexers.token_indexer import TokenIndexer
import tensorflow as tf
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
import numpy as np


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
            beginning_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
            end_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )

    bos_token = '<S>'
    eos_token = '</S>'

    @staticmethod
    def convert_word_to_char_ids(word: str) -> List[int]:
        if word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]


@register.register_subclass("token_indexer", "elmo_characters")
class ELMoTokenCharactersIndexer(TokenIndexer):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 max_length: int = None,
                 namespace: str = 'elmo_characters') -> None:
        super().__init__(namespace, max_length)

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary):
        # TODO(brendanr): Retain the token to index mappings in the vocabulary and remove this
        # pylint pragma. See:
        # https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/wordpiece_indexer.py#L113
        # pylint: disable=unused-argument

        texts = [token.text for token in tokens]
        texts = [ELMoCharacterMapper.bos_token] + texts + [ELMoCharacterMapper.eos_token]
        if any(text is None for text in texts):
            raise ConfigureError('ELMoTokenCharactersIndexer needs a tokenizer '
                                     'that retains text')
        return {self._namespace: [np.array(ELMoCharacterMapper.convert_word_to_char_ids(text), dtype=np.int64)
                                          for text in texts]}

    def pad_token_sequence(self, tokens, max_length):
        if len(tokens) >= max_length:
            tokens = tokens[:max_length-1]+[tokens[-1]]
            return tokens
        padding_tokens = [self.get_padding_values()]*ELMoCharacterMapper.max_word_length
        padding_tokens = [padding_tokens] * (max_length-len(tokens))
        padding_tokens = np.array(padding_tokens, dtype=np.int64)
        #while len(tokens) < max_length:
        tokens = np.concatenate((tokens, padding_tokens), axis=0)
            #tokens.append(padding_tokens)
        return tokens

    def get_padding_values(self):
        return 0

    def to_example(self, token_indexers):
        if self._max_length:
            token_indexers = self.pad_token_sequence(token_indexers, self._max_length)
        if len(token_indexers) == 0:
            token_indexers = np.array([[self.get_padding_values()]*ELMoCharacterMapper.max_word_length], dtype=np.int64)
        input_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=token_indexers[i]))
            for i in range(len(token_indexers))]
        return tf.train.FeatureList(feature=input_features)

    def get_example(self):
        return tf.FixedLenSequenceFeature([ELMoCharacterMapper.max_word_length], tf.int64, allow_missing=True)

    def get_padded_shapes(self):
        return [self._max_length, ELMoCharacterMapper.max_word_length]

    def get_tf_shapes_and_dtypes(self):
        return {'dtype': tf.int32, 'shape': (None, None, ELMoCharacterMapper.max_word_length)}



