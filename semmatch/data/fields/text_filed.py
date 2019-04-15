from semmatch.data.fields import Field


class TextField(Field):
    def __init__(self, tokens, token_indexers, max_length=None):
        super().__init__()
        self.tokens = tokens
        self._token_indexers = {token_indexer.get_namespace(): token_indexer
                                for token_name, token_indexer in token_indexers.items()}
        self._indexed_tokens = None
        self._max_length = max_length
        if self._max_length:
            for token_indexer in self._token_indexers.values():
                token_indexer.set_max_length(self._max_length)

    def index(self, vocab):
        token_arrays = {}
        for token_indexer in self._token_indexers.values():
            token_indices = token_indexer.tokens_to_indices(self.tokens, vocab)
            token_arrays.update(token_indices)
        self._indexed_tokens = token_arrays

    def count_vocab(self, counter):
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    def to_example(self):
        if self._indexed_tokens is None:
            return
        features = {}
        for (namespace, token_indexer) in self._token_indexers.items():
            feature = token_indexer.to_example(self._indexed_tokens[namespace])
            features[namespace] = feature
        return features

    def get_example(self):
        features = {}
        for (namespace, token_indexer) in self._token_indexers.items():
            feature = token_indexer.get_example()
            features[namespace] = feature
        return features

    def get_padded_shapes(self):
        padded_shapes = dict()
        for (namespace, token_indexer) in self._token_indexers.items():
            padded_shape = token_indexer.get_padded_shapes()
            padded_shapes[namespace] = padded_shape
        return padded_shapes

    def get_padding_values(self):
        padding_values = dict()
        for (namespace, token_indexer) in self._token_indexers.items():
            padding_value = token_indexer.get_padding_values()
            padding_values[namespace] = padding_value
        return padding_values

    def get_tf_shapes_and_dtypes(self):
        shapes_and_dtypes_dict = dict()
        for (namespace, token_indexer) in self._token_indexers.items():
            shapes_and_dtypes = token_indexer.get_tf_shapes_and_dtypes()
            shapes_and_dtypes_dict[namespace] = shapes_and_dtypes
        return shapes_and_dtypes_dict
