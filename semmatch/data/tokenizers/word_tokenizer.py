from typing import List
from semmatch.data.tokenizers import Token
from semmatch.data.tokenizers import Tokenizer, RegexWordSplitter, BlankWordFilter


class WordTokenizer(Tokenizer):
    def __init__(self, word_splitter=RegexWordSplitter(), word_filter=BlankWordFilter(), word_stemmer=None,  start_tokens=None, end_tokens: List[str] = None):
        self._word_splitter = word_splitter
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer

    def tokenize(self, text: Token) -> List[Token]:
        tokens = self._word_splitter.split_words(text)
        if self._word_filter:
            tokens = self._word_filter.filter_words(tokens)
        if self._word_stemmer:
            tokens = [self._word_stemmer.stem_word(token) for token in tokens]
        return tokens