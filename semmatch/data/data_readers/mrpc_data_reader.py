import os
from typing import Dict, List
import zipfile
import tensorflow as tf
from semmatch.data.data_readers import data_reader, DataSplit
from semmatch.data import data_utils
from semmatch.data.fields import Field, TextField, LabelField
from semmatch.data.tokenizers import WordTokenizer, Tokenizer
from semmatch.data import Instance
from semmatch.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from semmatch.utils import register
from semmatch.utils.logger import logger
import simplejson as json


@register.register_subclass('data', 'mrpc')
class MRPCDataReader(data_reader.DataReader):
    _mrpc_train_url = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
    _mrpc_test_url = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
    _mrpc_dev_ids_url = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc'

    def __init__(self, data_name: str = "mrpc", data_path: str = None, tmp_path: str = None, batch_size: int = 32,
                 emb_pretrained_files: Dict[str, str] = None, only_include_pretrained_words: bool = False,
                 train_filename="msr_paraphrase_train.txt",
                 valid_filename="msr_paraphrase_train.txt",
                 test_filename="msr_paraphrase_test.txt",
                 max_length: int = None, tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(data_name=data_name, data_path=data_path, tmp_path=tmp_path, batch_size=batch_size,
                         emb_pretrained_files=emb_pretrained_files,
                         only_include_pretrained_words=only_include_pretrained_words,
                         train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename, max_length=max_length)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}

    def _read(self, mode: str):
        self._maybe_download_corpora(self._data_path)
        filename = self.get_filename_by_mode(mode)
        if mode in [DataSplit.TRAIN, DataSplit.EVAL]:
            dev_ids_filename = "mrpc_dev_ids.tsv"
            dev_ids_path = os.path.join(self._data_path, dev_ids_filename)
            dev_ids = []
            with open(dev_ids_path, 'r', encoding="utf-8") as ids_fh:
                for row in ids_fh:
                    dev_ids.append(row.strip().split('\t'))

        if filename:
            file_path = os.path.join(self._data_path, filename)
            with open(file_path, 'r', encoding='utf-8') as mrpc_file:
                logger.info("Reading Microsoft Research Paraphrase Corpus instances from txt dataset at: %s", file_path)
                for line in mrpc_file:
                    fields = line.strip().split("\t")
                    label, id1, id2, s1, s2 = fields
                    if label not in ['0', '1']:
                        #print(fields)
                        continue
                    if (mode == DataSplit.TRAIN and [id1, id2] not in dev_ids) \
                        or (mode == DataSplit.EVAL and [id1, id2] in dev_ids) \
                        or mode == DataSplit.PREDICT or mode == DataSplit.TEST:
                        inputs = [[s1, s2], [s2, s1]]
                        for inp in inputs:
                            example = {
                                "premise": inp[0],
                                "hypothesis": inp[1],
                                "label": label
                                        }
                            yield self._process(example)
                    # else:
                    #     print(mode, id1, id2, [id1, id2] in dev_ids)
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
        #mrpc_finalpath = os.path.join(tmp_dir, "MRPC")
        train_mrpc_finalpath = os.path.join(tmp_dir, os.path.basename(self._mrpc_train_url))
        data_utils.maybe_download(
                train_mrpc_finalpath, self._mrpc_train_url)

        test_mrpc_finalpath = os.path.join(tmp_dir, os.path.basename(self._mrpc_test_url))
        data_utils.maybe_download(
                test_mrpc_finalpath, self._mrpc_test_url)

        dev_mrpc_finalpath = os.path.join(tmp_dir, "mrpc_dev_ids.tsv")
        data_utils.maybe_download(
                dev_mrpc_finalpath, self._mrpc_dev_ids_url)


