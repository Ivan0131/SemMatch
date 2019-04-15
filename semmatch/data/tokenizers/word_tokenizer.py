from typing import List
from semmatch.data.tokenizers import Token
from semmatch.data.tokenizers import Tokenizer, RegexWordSplitter, BlankWordFilter, WordFilter, WordSplitter, WordStemmer, BlankWordStemmer
from semmatch.utils import register
import spacy


@register.register_subclass("tokenizer", "word_tokenizer")
class WordTokenizer(Tokenizer):
    def __init__(self, word_splitter: WordSplitter = RegexWordSplitter(), word_filter: WordFilter = BlankWordFilter(),
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
    def __init__(self, lang: str = 'en'):
        self.nlp = spacy.load(lang, parse=False)

    def tokenize(self, text: str) -> List[Token]:
        text = convert_to_unicode(text)
        text = self.nlp(text)
        lemmas = [w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in text]
        tags = [w.tag_ for w in text]
        ents = [w.ent_type_ for w in text]
        words = [w for w in text]
        tokens = [Token(text=w.text, lemma=lemma, tag=tag, ent_type=ent) for w, tag, ent, lemma in zip(words, tags, ents, lemmas)]
        return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
