from typing import List
from semmatch.data.tokenizers import Token
from semmatch.config.init_from_params import InitFromParams
from semmatch.utils import register


@register.register("tokenizer")
class Tokenizer(InitFromParams):
    def tokenize(self, sentence: str) -> List[Token]:
        raise NotImplementedError

    def batch_tokenize(self, sentences) -> List[List[Token]]:
        return [self.tokenize(sentence) for sentence in sentences]