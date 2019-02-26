import os
from typing import Dict
from semmatch.data import instance
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError
import collections
import tqdm
import simplejson as json
import re
DEFAULT_NON_PADDED_NAMESPACES = (".*tags", ".*labels")
DEFAULT_PADDING_TOKEN = "[PAD]"
DEFAULT_OOV_TOKEN = "[UNK]"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'
VOCAB_FILE = 'vocab_%s.txt'


def save_to_txt(items, filename):
    with open(filename, 'w', encoding='utf-8') as txt_file:
        for item in items:
            txt_file.write(item+'\n')


def load_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
        items = [line.strip() for line in lines]
        return items


def namespace_match(patterns, name):
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


class ConditionDefaultDict(collections.defaultdict):
    def __init__(self, conditions, default_factory_1, default_factory_2):
        super().__init__()
        self._conditions = conditions
        self._default_factory_1 = default_factory_1
        self._default_factory_2 = default_factory_2

    def __missing__(self, key):
        if namespace_match(self._conditions, key):
            value = self._default_factory_1()
        else:
            value = self._default_factory_2()
        collections.defaultdict.__setitem__(self, key, value)
        return value


class TokenToIndexDict(ConditionDefaultDict):
    def __init__(self, non_padded_namespaces, padding_token,  oov_token):
        super().__init__(non_padded_namespaces, lambda: {}, lambda: {padding_token: 0, oov_token: 1})


class IndexToTokenDict(ConditionDefaultDict):
    def __init__(self, non_padded_namespaces, padding_token,  oov_token):
        super().__init__(non_padded_namespaces, lambda: {}, lambda: {0: padding_token, 1: oov_token})


class Vocabulary(object):
    def __init__(self, counter=None, non_padded_namespaces=DEFAULT_NON_PADDED_NAMESPACES):
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._token_to_index = TokenToIndexDict(self._non_padded_namespaces, self._padding_token, self._oov_token)
        self._index_to_token = IndexToTokenDict(self._non_padded_namespaces, self._padding_token, self._oov_token)
        self._namespace_to_path = dict()
        self._extend(counter)

    def save_to_files(self, directory):
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logger.warning("Vocabulary directory %s is not empty", directory)

        save_to_txt(self._non_padded_namespaces, os.path.join(directory, NAMESPACE_PADDING_FILE))
        for namespace in self._token_to_index:
            vocab_namespace = [self._index_to_token[namespace][i] for i in range(len(self._index_to_token[namespace]))]
            vocab_namespace_file = os.path.join(directory, VOCAB_FILE % namespace)
            self._namespace_to_path[namespace] = vocab_namespace_file
            save_to_txt(vocab_namespace, vocab_namespace_file)

    def load_from_files(self, directory):
        if not os.path.exists(directory):
            logger.warning("Vocabulary directory %s does not exist.", directory)
            return False
        namespaces_file = os.path.join(directory, NAMESPACE_PADDING_FILE)
        if not os.path.exists(namespaces_file):
            logger.warning("Vocabulary namespaces file %s does not exist", namespaces_file)
            return False

        vocab_filenames = [filename for filename in os.listdir(directory)
                            if filename.startswith(VOCAB_FILE[:6]) and filename.endswith(VOCAB_FILE[-4:])]
        if len(vocab_filenames) == 0:
            logger.warning("Vocabulary file %s does not exist")

        self._non_padded_namespaces = load_from_txt(namespaces_file)

        for vocab_filename in vocab_filenames:
            namespace = vocab_filename[6:-4]
            vocab_namespace_file = os.path.join(directory, vocab_filename)
            self._namespace_to_path[namespace] = vocab_namespace_file
            vocab_namespace = load_from_txt(vocab_namespace_file)
            self._index_to_token[namespace] = dict((index, token) for index, token in enumerate(vocab_namespace))
            self._token_to_index[namespace] = dict((token, index) for index, token in enumerate(vocab_namespace))

        if self.valid():
            return True
        else:
            return False

    def valid(self):
        for namespace, mapping in self._index_to_token.items():
            for (index, token) in mapping.items():
                if self._token_to_index[namespace][token] != int(index):
                    logger.error("index/token in index_to_token : %s/%s not in token_to_index"%(index, token))
                    return False
        return True

    def get_index_token(self, token_index, namespace):
        token_index = int(token_index)
        if token_index in self._index_to_token[namespace]:
            return self._index_to_token[namespace][token_index]
        else:
            return self._oov_token

    def convert_indexes_to_tokens(self, token_indexes, namespace, ignore_padding=True):
        if ignore_padding and namespace not in self._non_padded_namespaces:
            new_token_indexes = []
            for token_index in token_indexes:
                if token_index != 0:
                    new_token_indexes.append(token_index)
            token_indexes = new_token_indexes

        tokens = [self.get_index_token(token_index, namespace) for token_index in token_indexes]
        return tokens

    def get_token_index(self, token, namespace):
        token = str(token)
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error('Namespace: %s', namespace)
                logger.error('Token: %s', token)
                raise

    def add_token_to_vocab(self, token, namespace):
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace].keys())
            token = str(token)
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def _extend(self, counter=None, min_count=5):
        counter = counter or {}
        for namespace in counter:
            if namespace in self._non_padded_namespaces:
                counter_keys = sorted(counter[namespace].keys())
            else:
                counter_keys = counter[namespace].keys()
            for token in counter_keys:
                if counter[namespace][token] > min_count:
                    self.add_token_to_vocab(token, namespace)

    def get_vocab_size(self, namespace='tokens'):
        if namespace not in self._token_to_index: raise ConfigureError("namespace %s not in vocabulary."%namespace)
        return len(self._token_to_index[namespace])

    def get_vocab_tokens(self, namespace='tokens'):
        if namespace not in self._token_to_index: raise ConfigureError("namespace %s not in vocabulary."%namespace)
        return set(self._token_to_index[namespace].keys())

    def get_vocab_index_to_token(self, namespace='tokens'):
        if namespace not in self._token_to_index: raise ConfigureError("namespace %s not in vocabulary."%namespace)
        return self._index_to_token[namespace]

    @classmethod
    def init_from_instances(cls, instances):
        logger.info("create vocab from instance")
        namespace_counter = collections.defaultdict(lambda: collections.defaultdict(int))
        try:
            for i, instance in enumerate(tqdm.tqdm(instances)):
                instance.count_vocab(namespace_counter)
        except StopIteration as e:
            logger.error("The data reader builds vocabulary error with StopIteration.")
        return cls(namespace_counter)

    def get_vocab_path(self, namespace):
        if namespace not in self._namespace_to_path:
            logger.error("%s vocab file does not exist." % namespace)
        return self._namespace_to_path.get(namespace, None)
