import os
from typing import Dict
from semmatch.data import instance
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError
import collections
import tqdm
import simplejson as json
import re
DEFAULT_NON_PADDED_NAMESPACES = (".*labels", )
DEFAULT_NON_UNK_NAMESPACES = (".*tags", )
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
        lines = [line.strip('\n') for line in lines]
        if len(lines[0].split()) == 2:
            lines = lines[1:]
        items = [line.split(" ", 1)[0] if line.strip() != "" else line for line in lines]
        return items


def namespace_match(patterns, name):
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


class ConditionDefaultDict(collections.defaultdict):
    def __init__(self, conditions_1, conditions_2, default_factory_1, default_factory_2, default_factory_3):
        super().__init__()
        self._conditions_1 = conditions_1
        self._conditions_2 = conditions_2
        self._default_factory_1 = default_factory_1
        self._default_factory_2 = default_factory_2
        self._default_factory_3 = default_factory_3

    def __missing__(self, key):
        if namespace_match(self._conditions_1, key):
            value = self._default_factory_1()
        elif namespace_match(self._conditions_2, key):
            value = self._default_factory_2()
        else:
            value = self._default_factory_3()
        collections.defaultdict.__setitem__(self, key, value)
        return value


class TokenToIndexDict(ConditionDefaultDict):
    def __init__(self, non_padded_namespaces, non_unk_namespaces, padding_token,  oov_token):
        super().__init__(non_padded_namespaces, non_unk_namespaces, lambda: {}, lambda: {padding_token: 0}, lambda: {padding_token: 0, oov_token: 1})


class IndexToTokenDict(ConditionDefaultDict):
    def __init__(self, non_padded_namespaces, non_unk_namespaces, padding_token,  oov_token):
        super().__init__(non_padded_namespaces, non_unk_namespaces, lambda: {}, lambda: {0: padding_token}, lambda: {0: padding_token, 1: oov_token})


class Vocabulary(object):
    def __init__(self, counter=None, non_padded_namespaces=DEFAULT_NON_PADDED_NAMESPACES,
                 non_unk_namespace=DEFAULT_NON_UNK_NAMESPACES,
                 vocab_init_files: Dict[str, str] = None,
                 pretrained_files=None, only_include_pretrained_words=False):
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._non_unk_namespace = set(non_unk_namespace)
        self._token_to_index = TokenToIndexDict(self._non_padded_namespaces, self._non_unk_namespace,
                                                self._padding_token, self._oov_token)
        self._index_to_token = IndexToTokenDict(self._non_padded_namespaces, self._non_unk_namespace,
                                                self._padding_token, self._oov_token)
        if vocab_init_files is not None:
            for namespace, vocab_path in vocab_init_files.items():
                vocab_namespace = load_from_txt(vocab_path)
                if not namespace_match(DEFAULT_NON_PADDED_NAMESPACES, namespace):
                    if namespace_match(DEFAULT_NON_UNK_NAMESPACES, namespace):
                        if vocab_namespace[0] != DEFAULT_PADDING_TOKEN:
                            vocab_namespace.insert(0, DEFAULT_PADDING_TOKEN)
                        vocab_namespace = vocab_namespace[:1] + list(set(vocab_namespace[1:]))
                    else:
                        if vocab_namespace[0] != DEFAULT_PADDING_TOKEN:
                            vocab_namespace.insert(0, DEFAULT_PADDING_TOKEN)
                        if vocab_namespace[1] != DEFAULT_OOV_TOKEN:
                            vocab_namespace.insert(1, DEFAULT_OOV_TOKEN)
                        vocab_namespace = vocab_namespace[:2] + list(set(vocab_namespace[2:]))
                else:
                    vocab_namespace = list(set(vocab_namespace))
                self._index_to_token[namespace] = dict((index, token) for index, token in enumerate(vocab_namespace))
                self._token_to_index[namespace] = dict((token, index) for index, token in enumerate(vocab_namespace))
        self._namespace_to_path = dict()
        self._extend(counter, pretrained_files=pretrained_files,
                     only_include_pretrained_words=only_include_pretrained_words)

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
            raise ConfigureError("Vocabulary valid error")

    def valid(self):
        for namespace, mapping in self._index_to_token.items():
            for (index, token) in mapping.items():
                if self._token_to_index[namespace][token] != int(index):
                    logger.error("index/token in index_to_token with namespace %s: %s/%s not in token_to_index" %
                                 (namespace, index, token))
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

    def _extend(self, counter=None, min_count=5, pretrained_files=None, only_include_pretrained_words=False):
        pretrained_files = pretrained_files or {}
        pretrained_files = dict(pretrained_files)
        counter = counter or {}
        for namespace in counter:
            if namespace in pretrained_files:
                pretrained_list = []
                pretrained_path = pretrained_files[namespace]
                with open(pretrained_path, 'r', encoding='utf-8') as embeddings_file:
                    for line in tqdm.tqdm(embeddings_file):
                        token = line.rstrip().split(" ", 1)[0]
                        fields = line.rstrip().split(" ")
                        if len(fields) != 2:
                            pretrained_list.append(token)
                pretrained_set = set(pretrained_list)
            else:
                pretrained_set = None

            if namespace_match(self._non_padded_namespaces, namespace):
                counter_keys = sorted(counter[namespace].keys())
            else:
                counter_keys = counter[namespace].keys()
            for token in counter_keys:
                if pretrained_set is None:
                    if counter[namespace][token] > min_count:
                        self.add_token_to_vocab(token, namespace)
                else:
                    if only_include_pretrained_words:
                        if counter[namespace][token] > min_count and token in pretrained_set:
                            self.add_token_to_vocab(token, namespace)
                    elif counter[namespace][token] > min_count or token in pretrained_set:
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
    def init_from_instances(cls, instances, vocab_init_files=None, pretrained_files=None, only_include_pretrained_words=False):
        logger.info("create vocab from instance")
        namespace_counter = collections.defaultdict(lambda: collections.defaultdict(int))
        try:
            for i, instance in enumerate(tqdm.tqdm(instances)):
                instance.count_vocab(namespace_counter)
        except StopIteration as e:
            logger.error("The data reader builds vocabulary error with StopIteration.")
        return cls(namespace_counter, pretrained_files=pretrained_files,
                   vocab_init_files=vocab_init_files,
                   only_include_pretrained_words=only_include_pretrained_words)

    def get_vocab_path(self, namespace):
        if namespace not in self._namespace_to_path:
            logger.error("%s vocab file does not exist." % namespace)
        return self._namespace_to_path.get(namespace, None)
