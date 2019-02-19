from typing import List
from semmatch.data.tokenizers import Token
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


@register.register("word_stemmer")
class WordStemmer(InitFromParams):
    def stem_word(self, word: Token) -> Token:
        raise NotImplementedError


@register.register_subclass("word_stemmer", "blank_word_stemmer")
class BlankWordStemmer(WordStemmer):
    def stem_word(self, word: Token) -> Token:
        return word