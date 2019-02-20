class Field(object):
    def __init__(self):
        pass

    def count_vocab(self, counter):
        raise NotImplementedError

    def index(self, vocab):
        raise NotImplementedError

    def to_example(self):
        raise NotImplementedError

    def get_example(self):
        raise NotImplementedError

    def get_padded_shapes(self):
        raise NotImplementedError

    def get_padding_values(self):
        raise NotImplementedError

    def get_tf_shapes_and_dtypes(self):
        raise NotImplementedError
