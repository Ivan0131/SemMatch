import os
from typing import Dict, List
import zipfile
import tensorflow as tf
from semmatch.data.data_readers import data_reader
from semmatch.data import data_utils
from semmatch.data.fields import Field, TextField, LabelField
from semmatch.data.tokenizers import WordTokenizer, Tokenizer
from semmatch.data import Instance
from semmatch.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from semmatch.utils import register
from semmatch.utils.logger import logger


@register.register_subclass('data', 'quora')
class QuoraDataReader(data_reader.DataReader):
    _QQP_URL = ("https://firebasestorage.googleapis.com/v0/b/"
                "mtl-sentence-representations.appspot.com/o/"
                "data%2FQQP.zip?alt=media&token=700c6acf-160d-"
                "4d89-81d1-de4191d02cb5")

    def __init__(self, data_name: str = "quora", data_path: str = None, batch_size: int = 32, train_filename="train.tsv",
                 valid_filename="dev.tsv", test_filename=None, max_length: int = 48, tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: List[Tokenizer] = None):
        super().__init__(data_name=data_name, data_path=data_path, batch_size=batch_size, train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or [SingleIdTokenIndexer(namespace='tokens')]
        self._max_length = max_length

    def _read(self, mode: str):
        qqp_dir = self._maybe_download_corpora(self._data_path)
        filename = self.get_filename_by_mode(mode)
        if filename:
            filename = os.path.join(qqp_dir, filename)
            for example in self.example_generator(filename):
                yield example
        else:
            return None

    def _process(self, example):
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(example['premise'])
        tokenized_hypothesis = self._tokenizer.tokenize(example['hypothesis'])
        fields["premise"] = TextField(tokenized_premise, self._token_indexers, max_length=self._max_length)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers, max_length=self._max_length)
        if 'label' in example:
            fields['label'] = LabelField(example['label'])
        return Instance(fields)

    def example_generator(self, filename):
        skipped = 0
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            if len(split_line) < 6:
                skipped += 1
                tf.logging.info("Skipping %d" % skipped)
                continue
            s1, s2, l = split_line[3:]
            # A neat data augmentation trick from Radford et al. (2018)
            # https://blog.openai.com/language-unsupervised/
            inputs = [[s1, s2], [s2, s1]]
            for inp in inputs:
                example = {
                    "premise": inp[0],
                    "hypothesis": inp[1],
                    "label": int(l)
                }
                yield self._process(example)

    def _maybe_download_corpora(self, tmp_dir):
        qqp_filename = "QQP.zip"
        qqp_finalpath = os.path.join(tmp_dir, "QQP")
        if not os.path.exists(qqp_finalpath):
            zip_filepath = data_utils.maybe_download(
                tmp_dir, qqp_filename, self._QQP_URL)
            zip_ref = zipfile.ZipFile(zip_filepath, "r")
            zip_ref.extractall(tmp_dir)
            zip_ref.close()

        return qqp_finalpath
