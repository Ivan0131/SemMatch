import os
from typing import Dict, List
import zipfile
import tensorflow as tf
from semmatch.data.data_readers import data_reader, DataSplit
from semmatch.data import data_utils
from semmatch.data.fields import Field, TextField, LabelField, IndexField
from semmatch.data.tokenizers import WordTokenizer, Tokenizer
from semmatch.data import Instance
from semmatch.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from semmatch.utils import register
#from lxml import etree
from semmatch.utils.logger import logger
import bs4


@register.register_subclass('data', 'datarliving')
class DatarLivingDataReader(data_reader.DataReader):
    _datarliving_train_dev_url = 'http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-ql-traindev-v3.2.zip'
    _datarliving_test_dev_url = 'http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016_task3_test_input.zip'

    def __init__(self, data_name: str = "datarliving", data_path: str = None, tmp_path: str = None, batch_size: int = 32,
                 vocab_init_files: Dict[str, str] = None,
                 emb_pretrained_files: Dict[str, str] = None, only_include_pretrained_words: bool = False,
                 concat_sequence: bool = False,
                 train_filename=["v3.2/train/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml", "v3.2/train/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml"],
                 valid_filename="v3.2/dev/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml",
                 test_filename="SemEval2016_task3_test_input/English/SemEval2016-Task3-CQA-QL-test-subtaskA-input.xml",
                 max_length: int = None, tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(data_name=data_name, data_path=data_path, tmp_path=tmp_path, batch_size=batch_size,
                         emb_pretrained_files=emb_pretrained_files,
                         vocab_init_files=vocab_init_files, concat_sequence=concat_sequence,
                         only_include_pretrained_words=only_include_pretrained_words,
                         train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename, max_length=max_length)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}

    def _read(self, mode: str):
        self._maybe_download_corpora(self._data_path)
        filenames = self.get_filename_by_mode(mode)
        if filenames is None:
            return None

        if isinstance(filenames, str):
            filenames = [filenames]

        for filename in filenames:
            logger.info("reading QatarLiving data from %s" % filename)
            filepath = os.path.join(self._data_path, filename)
            threads = self.parse(filepath)
            for thread in threads:
                q_id = thread['RelQuestion']['RELQ_ID']
                q_category = thread['RelQuestion']['RELQ_CATEGORY']
                q_subject = thread['RelQuestion']['RelQSubject']
                q_text = thread['RelQuestion']['RelQBody']
                comments = thread['RelComments']
                for comment in comments:
                    c_id = comment["RELC_ID"]
                    c_rel = comment['RELC_RELEVANCE2RELQ'].lower()
                    c_test = comment['RelCText']
                    if c_rel != "?":
                        if c_rel != 'good':
                            c_rel = 'bad'
                        example = {
                            "index": c_id,
                            "premise": q_text,
                            "hypothesis": c_test,
                            "label": c_rel
                            }
                    else:
                        example = {
                            "index": c_id,
                            "premise": q_text,
                            "hypothesis": c_test,
                        }
                    yield self._process(example)

    def parse(self, xmlfilename):
        with open(xmlfilename, 'r', encoding='utf-8') as xmlfile:
            xml_str = xmlfile.read()
            soup = bs4.BeautifulSoup(xml_str, 'lxml')
            threads = []
            for thread in soup.find_all("thread"):
                relquestion = {}
                for attrib in ("RELQ_ID", "RELQ_CATEGORY", "RELQ_DATE", "RELQ_USERID", "RELQ_USERNAME"):
                    attrib = attrib.lower()
                    relquestion[attrib.upper()] = thread.find("relquestion")[attrib]
                relquestion["RelQSubject"] = thread.find("relquestion").relqsubject.text or ""
                relquestion["RelQBody"] = thread.find("relquestion").relqbody.text or ""

                relcomments = []
                for i, relcomment in enumerate(thread.find_all("relcomment")):
                    relcomments.append({})
                    for attrib in ("RELC_ID", "RELC_DATE", "RELC_USERID", "RELC_USERNAME", "RELC_RELEVANCE2RELQ"):
                        attrib = attrib.lower()
                        relcomments[i][attrib.upper()] = relcomment[attrib]
                    relcomments[i]["RelCText"] = relcomment.find("relctext").text or ""
                threads.append({
                    "THREAD_SEQUENCE": thread["thread_sequence"],
                    "RelQuestion": relquestion,
                    "RelComments": relcomments})
            return threads


    # def parse(self, xmlfilename):
    #     # with open(xmlfilename, 'r', encoding='utf-8') as xmlfile:
    #     #     xml_str = xmlfile.read()
    #     #     soup = bs4.BeautifulSoup(xml_str, 'lxml')
    #     #     print(soup.prettify())
    #     tree = etree.parse(xmlfilename)
    #     threads = []
    #     for thread in tree.findall("Thread"):
    #         relquestion = {}
    #         for attrib in ("RELQ_ID", "RELQ_CATEGORY", "RELQ_DATE", "RELQ_USERID", "RELQ_USERNAME"):
    #             relquestion[attrib] = thread.find("RelQuestion").attrib[attrib]
    #         relquestion["RelQSubject"] = thread.find("RelQuestion/RelQSubject").text or ""
    #         relquestion["RelQBody"] = thread.find("RelQuestion/RelQBody").text or ""
    #
    #         relcomments = []
    #         for i, relcomment in enumerate(thread.findall("RelComment")):
    #             relcomments.append({})
    #             for attrib in ("RELC_ID", "RELC_DATE", "RELC_USERID", "RELC_USERNAME", "RELC_RELEVANCE2RELQ"):
    #                 relcomments[i][attrib] = relcomment.attrib[attrib]
    #             relcomments[i]["RelCText"] = relcomment.find("RelCText").text or ""
    #
    #         threads.append({
    #             "THREAD_SEQUENCE": thread.attrib["THREAD_SEQUENCE"],
    #             "RelQuestion": relquestion,
    #             "RelComments": relcomments})
    #     return threads

    def _process(self, example):
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(example['premise'])
        tokenized_hypothesis = self._tokenizer.tokenize(example['hypothesis'])
        fields['index'] = IndexField(example['index'])
        fields["premise"] = TextField(tokenized_premise, self._token_indexers, max_length=self._max_length)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers, max_length=self._max_length)
        if 'label' in example:
            fields['label'] = LabelField(example['label'])
        return Instance(fields)

    def _maybe_download_corpora(self, tmp_dir):
        if not os.path.exists(tmp_dir):
            tf.gfile.MakeDirs(tmp_dir)

        train_mrpc_finalpath = os.path.join(tmp_dir, os.path.basename(self._datarliving_train_dev_url))
        data_utils.maybe_download(
                train_mrpc_finalpath, self._datarliving_train_dev_url)
        data_utils.unzip(train_mrpc_finalpath, tmp_dir)

        test_mrpc_finalpath = os.path.join(tmp_dir, os.path.basename(self._datarliving_test_dev_url))
        data_utils.maybe_download(
                test_mrpc_finalpath, self._datarliving_test_dev_url)
        data_utils.unzip(test_mrpc_finalpath, tmp_dir)




