from typing import List
from semmatch.data.tokenizers import Token
from semmatch.data.tokenizers import Tokenizer
from semmatch.utils import register


@register.register_subclass("tokenizer", "char_tokenizer")
class CharacterTokenizer(Tokenizer):
    def __init__(self,
                 byte_encoding: str = None,
                 lowercase_characters: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        # TODO(brendanr): Add length truncation.
        self._byte_encoding = byte_encoding
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    def tokenize(self, text: str) -> List[Token]:
        if self._lowercase_characters:
            text = text.lower()
        if self._byte_encoding is not None:
            # We add 1 here so that we can still use 0 for masking, no matter what bytes we get out
            # of this.
            tokens = [Token(text_id=c + 1) for c in text.encode(self._byte_encoding)]
        else:
            tokens = [Token(t) for t in list(text)]
        for start_token in self._start_tokens:
            if isinstance(start_token, int):
                token = Token(text_id=start_token, idx=0)
            else:
                token = Token(text=start_token, idx=0)
            tokens.insert(0, token)
        for end_token in self._end_tokens:
            if isinstance(end_token, int):
                token = Token(text_id=end_token, idx=0)
            else:
                token = Token(text=end_token, idx=0)
            tokens.append(token)
        return tokens
