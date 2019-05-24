from typing import List
from semmatch.data.tokenizers import Token
from semmatch.data.tokenizers import Tokenizer, NLTKSplitter, BlankWordFilter, WordFilter, WordSplitter, WordStemmer, BlankWordStemmer
from semmatch.utils import register
import spacy


@register.register_subclass("tokenizer", "word_tokenizer")
class WordTokenizer(Tokenizer):
    def __init__(self, word_splitter: WordSplitter = NLTKSplitter(), word_filter: WordFilter = BlankWordFilter(),
                 word_stemmer: WordStemmer = BlankWordStemmer()) -> None:
        self._word_splitter = word_splitter
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer

    def tokenize(self, text: str) -> List[Token]:
        text = convert_to_unicode(text)
        tokens = self._word_splitter.split_words(text)
        tokens = self._word_filter.filter_words(tokens)
        tokens = [self._word_stemmer.stem_word(token) for token in tokens]
        return tokens


@register.register_subclass("tokenizer", "spacy_tokenizer")
class SpacyTokenizer(Tokenizer):
    def __init__(self, do_lower_case=True, lang: str = 'en',
                 disable: List[str] = None):
        self.do_lower_case = do_lower_case
        if disable is None:
            self._disable = ["ner", 'parser']
        else:
            self._disable = disable
        self.nlp = spacy.load(lang, disable=self._disable)

    def tokenize(self, text: str) -> List[Token]:
        text = convert_to_unicode(text)
        text = self.nlp(text)
        if 'tagger' in self._disable:
            lemmas = [None for w in text]
            tags = [None for w in text]
        else:
            lemmas = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in text]
            tags = [w.tag_ for w in text]
        if 'ner' in self._disable:
            ents = [None for w in text]
        else:
            ents = [w.ent_type_ for w in text]
        words = [w.text.lower() if self.do_lower_case else w.text for w in text]
        tokens = [Token(text=w, lemma=lemma, tag=tag, ent_type=ent) for w, tag, ent, lemma in zip(words, tags, ents, lemmas)]
        return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
