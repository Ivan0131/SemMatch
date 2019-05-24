import re
import unicodedata
import nltk
from typing import List
from semmatch.data.tokenizers import Token
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams
from semmatch.utils.tokenizer_utils import is_whitespace, is_control, is_punctuation, whitespace_tokenize
from semmatch.data.vocabulary import DEFAULT_OOV_TOKEN


@register.register("word_splitter")
class WordSplitter(InitFromParams):
    def __init__(self, do_lower_case: bool = True):
        self.do_lower_case = do_lower_case

    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [self.split_words(sentence) for sentence in sentences]

    def split_words(self, sentence: str) -> List[Token]:
        raise NotImplementedError


@register.register_subclass("word_splitter", "regex_word_splitter")
class RegexWordSplitter(WordSplitter):
    def split_words(self, sentence: str) -> List[Token]:
        words = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", sentence)
        return [Token(word.lower()) if self.do_lower_case else Token(word) for word in words]


@register.register_subclass("word_splitter", "white_word_splitter")
class WhiteWordSplitter(WordSplitter):
    def split_words(self, sentence: str) -> List[Token]:
        words = whitespace_tokenize(sentence)
        return [Token(word.lower()) if self.do_lower_case else Token(word) for word in words]


@register.register_subclass("word_splitter", "nltk_splitter")
class NLTKSplitter(WordSplitter):
    def split_words(self, sentence: str) -> List[Token]:
        words = nltk.word_tokenize(sentence)
        return [Token(word.lower()) if self.do_lower_case else Token(word) for word in words]


@register.register_subclass("word_splitter", "char_splitter")
class CharSplitter(WordSplitter):
    def split_words(self, sentence: str) -> List[Token]:
        words = [char for char in sentence]
        return [Token(word.lower()) if self.do_lower_case else Token(word) for word in words]


@register.register_subclass("word_splitter", "bert_basic_splitter")
class BertBasicSplitter(WordSplitter):
    def split_words(self, sentence: str)-> List[Token]:
        """Tokenizes a piece of text."""
        text = self._clean_text(sentence)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return [Token(word) for word in output_tokens]

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


@register.register_subclass("word_splitter", "bert_wordpiece_splitter")
class BertWordpieceSplitter(WordSplitter):
    def __init__(self, vocab_file, unk_token=DEFAULT_OOV_TOKEN, max_input_chars_per_word=200, do_lower_case=True):
        super().__init__(do_lower_case=do_lower_case)
        with open(vocab_file, 'r') as txt_file:
            vocab = txt_file.readlines()
            vocab = [v.strip() for v in vocab]
            vocab = set(vocab)
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def split_words(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            if self.do_lower_case:
                token = token.lower()
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return [Token(word) for word in output_tokens]