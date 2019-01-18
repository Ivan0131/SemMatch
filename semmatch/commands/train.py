import collections
from semmatch.commands.command import Command
from semmatch.utils import register
from semmatch.config.parameters import Parameters
from semmatch.utils.logger import logger
from semmatch.utils.exception import NotFoundError
from semmatch.data.data_readers.data_reader import DataSplit, DataReader



@register.register_subclass('command', 'train')
class Train(Command):
    name = 'train'
    description = 'Train a specified model on a specified dataset'
    parser = None

    def __init__(self, data: DataReader):
        self._data = data
        self.train_input_fn = None
        self.valid_input_fn = None
        self.test_data_path = None
        self.set_input_fns(data)

    def make_input_fns(self, data: DataReader):
        train_input_fn = data.make_estimator_input_fn(DataSplit.TRAIN, force_repeat=True)
        valid_input_fn = data.make_estimator_input_fn(DataSplit.EVAL, force_repeat=True)
        test_data_path = data.make_estimator_input_fn(DataSplit.TEST, force_repeat=True)
        return train_input_fn, valid_input_fn, test_data_path

    def set_input_fns(self, data: DataReader):
        self.train_input_fn, self.valid_input_fn, self.test_data_path = self.make_input_fns(data)



    @classmethod
    def add_subparser(cls, parser):
        cls.parser = parser.add_parser(name=cls.name, description=cls.description, help='train a model')
        cls.parser.add_argument('--config_path', type=str,
                                help='the config path where store the params.')
        cls.parser.set_defaults(func=cls.init_train_from_args)
        return cls.parser

    @classmethod
    def init_train_from_args(cls, args):
        config_path = args.config_path
        cls.init_train_from_config_file(config_path)

    @classmethod
    def init_train_from_config_file(cls, config_path):
        params = Parameters.init_from_file(config_path)
        return cls.init_from_params(params)

