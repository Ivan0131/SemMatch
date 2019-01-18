from typing import List
from semmatch.data.tokenizers import Token
from semmatch.data.tokenizers import Tokenizer, RegexWordSplitter, BlankWordFilter, WordFilter, WordSplitter, WordStemmer, BlankWordStemmer
from semmatch.utils import register


@register.register_subclass("tokenizer", "word_tokenizer")
class WordTokenizer(Tokenizer):
    def __init__(self, word_splitter: WordSplitter = RegexWordSplitter(), word_filter: WordFilter = BlankWordFilter(),
                 word_stemmer: WordStemmer = BlankWordStemmer()) -> None:
        self._word_splitter = word_splitter
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer

    def tokenize(self, text: Token) -> List[Token]:
        tokens = self._word_splitter.split_words(text)
        tokens = self._word_filter.filter_words(tokens)
        tokens = [self._word_stemmer.stem_word(token) for token in tokens]
        return tokens