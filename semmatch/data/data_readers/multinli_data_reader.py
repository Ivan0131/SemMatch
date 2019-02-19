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
import simplejson as json


@register.register_subclass('data', 'mnli')
class MultinliDataReader(data_reader.DataReader):
    #_snli_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    _mnli_url = 'https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
    _proprocessed_shared_file_id = "0B6CTyAhSHoJTa3ZSSE5QQUJrb3M"
    _multinli_matched_test_id = "1MddwhXZvsWSZZsvL_5oaOsJZFOshG6Dj"
    _multinli_mismatched_test_id = "1mzJFQ-BgkXJEr-fjv6L0t7SryeF546Cl"
    _proprocessed_shared_file_path = 'shared.jsonl'
    _multinli_matched_test_path = "multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl"
    _multinli_mismatched_test_path = "multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl"

    def __init__(self, data_name: str = "mnli", data_path: str = None, batch_size: int = 32,
                 train_filename="multinli_0.9/multinli_0.9_train.jsonl",
                 valid_filename="multinli_0.9/multinli_0.9_dev_matched.jsonl",
                 #valid_mnli_mismatched_filename="multinli_0.9/multinli_0.9_dev_mismatched.jsonl",
                 test_filename="multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl",
                 #train_snli_filename="snli_1.0/snli_1.0_train.jsonl",
                 #valid_snli_filename="snli_1.0/snli_1.0_dev.jsonl",
                 #test_snli_filename='snli_1.0/snli_1.0_test.jsonl',
                 max_length: int = None, tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: List[Tokenizer] = None):
        super().__init__(data_name=data_name, data_path=data_path, batch_size=batch_size, train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or [SingleIdTokenIndexer(namespace='tokens')]
        self._max_length = max_length

    def _read(self, mode: str):
        self._maybe_download_corpora(self._data_path)
        filename = self.get_filename_by_mode(mode)
        if filename:
            file_path = os.path.join(self._data_path, filename)

            with open(file_path, 'r') as snli_file:
                logger.info("Reading Multinli instances from jsonl dataset at: %s", file_path)
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
        fields["premise"] = TextField(tokenized_premise, self._token_indexers, max_length=self._max_length)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers, max_length=self._max_length)
        if 'label' in example:
            fields['label'] = LabelField(example['label'])
        return Instance(fields)

    def _maybe_download_corpora(self, tmp_dir):
        if not os.path.exists(tmp_dir):
            tf.gfile.MakeDirs(tmp_dir)
        # ###snli###
        # snli_filename = os.path.basename(self._snli_url)
        # snli_finalpath = os.path.join(tmp_dir, snli_filename)
        # if not os.path.exists(snli_finalpath):
        #     data_utils.maybe_download(
        #         snli_finalpath, self._snli_url)
        # data_utils.unzip(snli_finalpath, tmp_dir)
        # snli_finaldir = os.path.join(tmp_dir, os.path.splitext(snli_filename)[0])

        ###mnli####
        mnli_filename = os.path.basename(self._mnli_url)
        mnli_finalpath = os.path.join(tmp_dir, mnli_filename)
        if not os.path.exists(mnli_finalpath):
            data_utils.maybe_download(
                mnli_finalpath, self._mnli_url)
        data_utils.unzip(mnli_finalpath, tmp_dir)
        mnli_finaldir = os.path.join(tmp_dir, os.path.splitext(mnli_filename)[0])

        ###sharedfile#######
        data_utils.download_file_from_google_drive(self._proprocessed_shared_file_id,
                                                   os.path.join(tmp_dir, self._proprocessed_shared_file_path))

        ###mnli matched#####
        data_utils.download_file_from_google_drive(self._multinli_matched_test_id,
                                                   os.path.join(tmp_dir, self._multinli_matched_test_path))
        ###mnli mismatched#####
        data_utils.download_file_from_google_drive(self._multinli_mismatched_test_id,
                                                   os.path.join(tmp_dir, self._multinli_mismatched_test_path))
