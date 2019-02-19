from typing import List
from semmatch.data.tokenizers import Token
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


@register.register("word_filter")
class WordFilter(InitFromParams):
    """
    A ``WordFilter`` removes words from a token list.  Typically, this is for stopword removal,
    though you could feasibly use it for more domain-specific removal if you want.

    Word removal happens `before` stemming, so keep that in mind if you're designing a list of
    words to be removed.
    """

    def filter_words(self, words: List[Token]) -> List[Token]:
        """
        Returns a filtered list of words.
        """
        raise NotImplementedError


@register.register_subclass("word_filter", "blank_word_filter")
class BlankWordFilter(WordFilter):
    def filter_words(self, words: List[Token]) -> List[Token]:
        return words
