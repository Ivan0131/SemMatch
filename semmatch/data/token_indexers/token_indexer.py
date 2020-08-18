from typing import Dict, List, TypeVar, Generic
from semmatch.data.tokenizers.token import Token
from semmatch.data.vocabulary import Vocabulary
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


@register.register("token_indexer")
class TokenIndexer(InitFromParams):
    def __init__(self, namespace: str, max_length: int = None):
        self._namespace = namespace
        self._max_length = max_length

    def set_max_length(self, max_length):
        self._max_length = max_length

    def get_namespace(self):
        return self._namespace

    def to_example(self, token_indexers):
        raise NotImplementedError

    def to_raw_data(self, token_indexers):
        raise NotImplementedError

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        """
        The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
        data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
        token).  This method takes a token and a dictionary of counts and increments counts for
        whatever vocabulary items are present in the token.  If this is a single token ID
        representation, the vocabulary item is likely the token itself.  If this is a token
        characters representation, the vocabulary items are all of the characters in the token.
        """
        raise NotImplementedError

    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary):
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        This could be just an ID for each token from the vocabulary.
        Or it could split each token into characters and return one ID per character.
        Or (for instance, in the case of byte-pair encoding) there might not be a clean
        mapping from individual tokens to indices.
        """
        raise NotImplementedError

    def get_padding_values(self):
        """
        When we need to add padding tokens, what should they look like?  This method returns a
        "blank" token of whatever type is returned by :func:`tokens_to_indices`.
        """
        raise NotImplementedError

    def get_padded_shapes(self):
        raise NotImplementedError

    def get_tf_shapes_and_dtypes(self):
        raise NotImplementedError