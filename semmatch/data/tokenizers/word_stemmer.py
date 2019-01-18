from typing import List
from semmatch.data.tokenizers import Token
from semmatch.utils import register


@register.register("word_stemmer")
class WordStemmer(object):
    def stem_word(self, word: Token) -> Token:
        raise NotImplementedError


@register.register_subclass("word_stemmer", "blank_word_stemmer")
class BlankWordStemmer(WordStemmer):
    def stem_word(self, word: Token) -> Token:
        return word