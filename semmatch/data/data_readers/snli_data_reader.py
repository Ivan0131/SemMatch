import os
from typing import Dict, List
import zipfile
import tensorflow as tf
from semmatch.data.data_readers import data_reader
from semmatch.data import data_utils
from semmatch.data.fields import Field, TextField, LabelField
from semmatch.data.tokenizers import WordTokenizer, Tokenizer
from semmatch.data import Instance
from semmatch.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, FieldIndexer
from semmatch.utils import register
from semmatch.utils.logger import logger
import simplejson as json


@register.register_subclass('data', 'snli')
class SnliDataReader(data_reader.DataReader):
    _snli_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'

    def __init__(self, data_name: str = "snli", data_path: str = None, tmp_path: str = None, batch_size: int = 32,
                 vocab_init_files: Dict[str, str] = None,
                 concat_sequence: bool = False,
                 emb_pretrained_files: Dict[str, str] = None, only_include_pretrained_words: bool = False,
                 train_filename="snli_1.0/snli_1.0_train.jsonl",
                 valid_filename="snli_1.0/snli_1.0_dev.jsonl",
                 test_filename='snli_1.0/snli_1.0_test.jsonl',
                 max_length: int = None, tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(data_name=data_name, data_path=data_path, tmp_path=tmp_path, batch_size=batch_size,
                         vocab_init_files=vocab_init_files,
                         emb_pretrained_files=emb_pretrained_files,
                         only_include_pretrained_words=only_include_pretrained_words, concat_sequence=concat_sequence,
                         train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename, max_length=max_length)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}

        self._cal_exact_match = False
        for token_indexer_name, token_indexer in self._token_indexers.items():
            if isinstance(token_indexer, FieldIndexer):
                if token_indexer.get_field_name() == 'exact_match':
                    self._cal_exact_match = True

    def _read(self, mode: str):
        self._maybe_download_corpora(self._data_path)
        filename = self.get_filename_by_mode(mode)
        if filename:
            file_path = os.path.join(self._data_path, filename)

            with open(file_path, 'r') as snli_file:
                logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
                for line in snli_file:
                    fields = json.loads(line)

                    label = fields["gold_label"]
                    if label == '-':
                        # These were cases where the annotators disagreed; we'll just skip them.  It's
                        # like 800 out of 500k examples in the training data.
                        continue

                    example = {
                        "premise": fields["sentence1"],
                        "hypothesis": fields["sentence2"],
                        "label": label
                                }

                    yield self._process(example)
        else:
            return None

    def _process(self, example):
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(example['premise'])
        tokenized_hypothesis = self._tokenizer.tokenize(example['hypothesis'])
        if self._cal_exact_match:
            premise_exact_match, hypothesis_exact_match = \
                data_utils.get_exact_match(tokenized_premise, tokenized_hypothesis)

            for token in tokenized_premise:
                token.exact_match = 0

            for token in tokenized_hypothesis:
                token.exact_match = 0

            for ind in premise_exact_match:
                tokenized_premise[ind].exact_match = 1

            for ind in hypothesis_exact_match:
                tokenized_hypothesis[ind].exact_match = 1

        fields["premise"] = TextField(tokenized_premise, self._token_indexers, max_length=self._max_length)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers, max_length=self._max_length)
        if 'label' in example:
            fields['label'] = LabelField(example['label'])
        return Instance(fields)

    def _maybe_download_corpora(self, tmp_dir):
        if not os.path.exists(tmp_dir):
            tf.gfile.MakeDirs(tmp_dir)
        snli_filename = os.path.basename(self._snli_url)
        snli_finalpath = os.path.join(tmp_dir, snli_filename)
        if not os.path.exists(snli_finalpath):
            data_utils.maybe_download(
                snli_finalpath, self._snli_url)
        data_utils.unzip(snli_finalpath, tmp_dir)


