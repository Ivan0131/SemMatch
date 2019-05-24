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
from semmatch.utils.exception import ConfigureError
import simplejson as json


@register.register_subclass('data', 'general')
class GeneralDataReader(data_reader.DataReader):
    def __init__(self, data_name: str = "general", data_path: str = None, tmp_path: str = None, batch_size: int = 32,
                 train_filename: str = None, valid_filename: str = None, test_filename: str = None,
                 field_mapping: Dict = None,
                 concat_sequence: bool = False,
                 max_length: int = None, tokenizer: Tokenizer = WordTokenizer(),
                 vocab_init_files: Dict[str, str] = None,
                 emb_pretrained_files: Dict[str, str] = None, only_include_pretrained_words: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(data_name=data_name, data_path=data_path, tmp_path=tmp_path, batch_size=batch_size,
                         emb_pretrained_files=emb_pretrained_files,
                         vocab_init_files=vocab_init_files,
                         only_include_pretrained_words=only_include_pretrained_words, concat_sequence=concat_sequence,
                         train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename, max_length=max_length)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._field_mapping = field_mapping

    def _read(self, mode: str):
        filename = self.get_filename_by_mode(mode)
        if filename:
            file_path = os.path.join(self._data_path, filename)
            if file_path.lower().endswith("jsonl"):
                if self._field_mapping is None:
                    raise ConfigureError("field mapping is not provided for jsonl file.")
                with open(file_path, 'r') as json_file:
                    logger.info("Reading instances from jsonl dataset at: %s", file_path)
                    for line in json_file:
                        fields = json.loads(line)
                        example = {}
                        for (field_tar, field_src) in self._field_mapping.items():
                            example[field_tar] = fields[field_src]
                        yield self._process(example)
            if file_path.lower().endswith("csv"):
                if self._field_mapping is None:
                    raise ConfigureError("field mapping is not provided for csv file.")
                with open(file_path, 'r') as csv_file:
                    logger.info("Reading instances from csv dataset at: %s", file_path)
                    for line in csv_file:
                        fields = line.strip().split("\t")
                        example = {}
                        for (field_tar, field_src) in self._field_mapping.items():
                            example[field_tar] = fields[int(field_src)]
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



