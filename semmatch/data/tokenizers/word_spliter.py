import re
from typing import List
from semmatch.data.tokenizers import Token
from semmatch.utils import register


@register.register("word_splitter")
class WordSplitter(object):
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [self.split_words(sentence) for sentence in sentences]

    def split_words(self, sentence: str) -> List[Token]:
        raise NotImplementedError


@register.register_subclass("word_splitter", "regex_word_splitter")
class RegexWordSplitter(WordSplitter):
    def split_words(self, sentence: str) -> List[Token]:
        words = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", sentence)
        return [Token(word) for word in words]

