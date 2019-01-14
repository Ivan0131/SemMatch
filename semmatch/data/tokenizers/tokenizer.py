from typing import List
from semmatch.data.tokenizers import Token


class Tokenizer(object):
    def tokenize(self, sentence: str) -> List[Token]:
        raise NotImplementedError

    def batch_tokenize(self, sentences) -> List[List[Token]]:
        return [self.tokenize(sentence) for sentence in sentences]