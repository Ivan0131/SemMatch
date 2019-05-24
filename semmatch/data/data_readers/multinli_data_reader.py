import os
from typing import Dict, List
import zipfile
import re
import tensorflow as tf
from semmatch.data.data_readers import data_reader
from semmatch.data.data_readers.data_reader import DataSplit
from semmatch.data import data_utils
from semmatch.data.fields import Field, TextField, LabelField, IndexField
from semmatch.data.tokenizers import WordTokenizer, Tokenizer, NLTKSplitter, Token, WhiteWordSplitter, NLTKWordStemmer
from semmatch.data import Instance
from semmatch.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PosTagIndexer, FieldIndexer, \
    CharactersIndexer
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

    def __init__(self, data_name: str = "mnli", data_path: str = None, tmp_path: str = None, batch_size: int = 32,
                 vocab_init_files: Dict[str, str] = None,
                 concat_sequence: bool = False,
                 emb_pretrained_files: Dict[str, str] = None, only_include_pretrained_words: bool = False,
                 train_filename="multinli_0.9/multinli_0.9_train.jsonl",
                 valid_filename="multinli_0.9/multinli_0.9_dev_matched.jsonl",
                 #valid_mnli_mismatched_filename="multinli_0.9/multinli_0.9_dev_mismatched.jsonl",
                 test_filename="multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl",
                 #train_snli_filename="snli_1.0/snli_1.0_train.jsonl",
                 #valid_snli_filename="snli_1.0/snli_1.0_dev.jsonl",
                 #test_snli_filename='snli_1.0/snli_1.0_test.jsonl',
                 max_length: int = None, tokenizer: Tokenizer = WordTokenizer(word_splitter = WhiteWordSplitter()),
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(data_name=data_name, data_path=data_path, tmp_path=tmp_path, batch_size=batch_size,
                         vocab_init_files=vocab_init_files,
                         emb_pretrained_files=emb_pretrained_files, concat_sequence=concat_sequence,
                         only_include_pretrained_words=only_include_pretrained_words,
                         train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename, max_length=max_length)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens'),
                                                  'chars': CharactersIndexer(namespace='chars'),
                                                  'pos_tags': PosTagIndexer(namespace='pos_tags'),
                                                  'exact_match_labels': FieldIndexer(namespace='exact_match_labels',
                                                                                     field_name='exact_match')}
        self._cal_exact_match = False
        for token_indexer_name, token_indexer in self._token_indexers.items():
            if isinstance(token_indexer, FieldIndexer):
                if token_indexer.get_field_name() == 'exact_match':
                    self._cal_exact_match = True

    def _parsing_parse(self, parse):
        base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
        pos = [pair.split(" ")[0] for pair in base_parse]
        return pos

    def _read(self, mode: str):
        self._maybe_download_corpora(self._data_path)

        extra_info = {}
        with open(os.path.join(self._data_path, "shared.jsonl"), 'r') as jsonl_file:
            for line in jsonl_file:
                fields = line.split("\t")
                id = fields[0]
                d = json.loads(fields[1])
                extra_info[id] = d

        filename = self.get_filename_by_mode(mode)
        if filename:
            file_path = os.path.join(self._data_path, filename)

            with open(file_path, 'r') as mnli_file:
                logger.info("Reading Multinli instances from jsonl dataset at: %s", file_path)
                for line in mnli_file:
                    fields = json.loads(line)

                    id = fields['pairID']
                    label = fields["gold_label"]
                    sent1 = fields["sentence1_binary_parse"]
                    sent2 = fields["sentence2_binary_parse"]
                    sent1 = re.sub(r'\(|\)', '', sent1)
                    sent2 = re.sub(r'\(|\)', '', sent2)
                    sent1 = sent1.replace(' ', '')
                    sent2 = sent2.replace(' ', '')
                    sent1_pos = fields['sentence1_parse']
                    sent2_pos = fields['sentence2_parse']

                    ant_feat_1 = extra_info[id]['sentence1_antonym_feature']
                    ant_feat_2 = extra_info[id]['sentence2_antonym_feature']
                    if len(ant_feat_1) or len(ant_feat_2):
                        print(ant_feat_1, ant_feat_2)

                    if label == '-':
                        # These were cases where the annotators disagreed; we'll just skip them.  It's
                        # like 800 out of 500k examples in the training data.
                        continue

                    if mode in [DataSplit.TRAIN, DataSplit.EVAL]:
                        example = {
                            "id": id,
                            "premise": sent1,
                            "hypothesis": sent2,
                            "premise_pos": sent1_pos,
                            "hypothesis_pos": sent2_pos,
                            "label": label
                        }
                    else:
                        example = {
                            "id": id,
                            "premise": sent1,
                            "hypothesis": sent2,
                            "premise_pos": sent1_pos,
                            "hypothesis_pos": sent2_pos,
                        }

                    yield self._process(example)
        else:
            return None

    def _process(self, example):
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(example['premise'])
        tokenized_hypothesis = self._tokenizer.tokenize(example['hypothesis'])
        premise_pos = self._parsing_parse(example['premise_pos'])
        hypothesis_pos = self._parsing_parse(example['hypothesis_pos'])

        if len(tokenized_premise) != len(premise_pos):
            print(tokenized_premise)
            print(premise_pos)
            print(example['premise'])
            print(example['premise_pos'])

        assert len(tokenized_premise) == len(premise_pos)

        if len(tokenized_hypothesis) != len(hypothesis_pos):
            print(tokenized_hypothesis)
            print(hypothesis_pos)
            print(example['hypothesis'])
            print(example['hypothesis_pos'])
        assert len(tokenized_hypothesis) == len(hypothesis_pos)

        for token, token_pos in zip(tokenized_premise, premise_pos):
            token.tag_ = token_pos

        for token, token_pos in zip(tokenized_hypothesis, hypothesis_pos):
            token.tag_ = token_pos

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

        fields['index'] = IndexField(example['id'])
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
