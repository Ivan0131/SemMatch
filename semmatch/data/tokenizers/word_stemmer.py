from typing import List
from semmatch.data.tokenizers import Token


class WordStemmer(object):
    def stem_word(self, word: Token) -> Token:
        raise NotImplementedError


class BlankWordStemmer(WordStemmer):
    def stem_word(self, word: Token) -> Token:
        return word