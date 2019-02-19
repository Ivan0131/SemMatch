import os
import tensorflow as tf
from semmatch.commands.command import Command
from semmatch.utils import register
from semmatch.config.parameters import Parameters
from semmatch.utils.exception import ConfigureError
from semmatch.data.data_readers.data_reader import DataSplit, DataReader
from semmatch.models import Model
from semmatch.config.run_config import RunConfig
from semmatch.config.hparams import HParams
from semmatch.utils.logger import logger


@register.register_subclass('command', 'train')
class Train(Command):
    name = 'train'
    description = 'Train a specified model on a specified dataset'
    parser = None

    def __init__(self, data_reader=None, train_input_fn=None, valid_input_fn=None, test_input_fn=None,
                 serving_feature_spec=None, model=None, warm_start_from=None, hparams=HParams(),
                 run_config: RunConfig = RunConfig()):
        if data_reader is not None and train_input_fn is None:
            self._train_input_fn, self._valid_input_fn, self._test_input_fn = self.make_input_fns(data_reader)
            self._serving_feature_spec = data_reader.get_features(DataSplit.EVAL)
        else:
            self._train_input_fn = train_input_fn
            self._valid_input_fn = valid_input_fn
            self._test_input_fn = test_input_fn
            self._serving_feature_spec = serving_feature_spec
        if self._train_input_fn is None:
            raise ConfigureError("The train dataset is not provided.")

        if model is None:
            raise ConfigureError("Please provide model for training.")
        self._model_fn = model.make_estimator_model_fn()

        self._estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            config=run_config, params=hparams, warm_start_from=warm_start_from)

        early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
            self._estimator,
            metric_name='loss',
            max_steps_without_decrease=1000,
            min_steps=100)
        exporters = None
        if self._serving_feature_spec:
            serving_input_receiver_fn = (
                tf.estimator.export.build_parsing_serving_input_receiver_fn(
                    self._serving_feature_spec))

            exporters = tf.estimator.BestExporter(
                name="best_exporter",
                serving_input_receiver_fn=serving_input_receiver_fn,
                exports_to_keep=5)

        self._train_spec = tf.estimator.TrainSpec(input_fn=self._train_input_fn, max_steps=hparams.train_steps,
                                                  hooks=[early_stopping])
        if self._valid_input_fn:
            self._valid_spec = tf.estimator.EvalSpec(input_fn=self._valid_input_fn, steps=hparams.eval_steps,
                                                     exporters=exporters)

        tf.estimator.train_and_evaluate(self._estimator, self._train_spec, self._valid_spec)
        # print(eval_results)
        # eval_results = [eval_result[0] for eval_result in eval_results]
        # eval_results = sorted(eval_results, key=lambda x: x['loss'])
        # logger.info("best evaluation result:")
        # logger.info(eval_results[0])
        # best_checkpoint_path = os.path.join(run_config.model_dir, "model.ckpt-%s" % eval_results[0]['global_step'])
        # if not os.path.exists(best_checkpoint_path+".index"):
        #     best_checkpoint_path = None
        # print(best_checkpoint_path)
        if self._test_input_fn:
            self._estimator.evaluate(self._test_input_fn, steps=hparams.test_steps, name=DataSplit.TEST)

    @classmethod
    def init_from_params(cls, params):
        # ####data reader##############
        data_reader = DataReader.init_from_params(params.pop('data'))
        vocab = data_reader.get_vocab()
        #####embedding mapping##########
        model = Model.init_from_params(params.pop('model'), vocab=vocab)
        run_config = RunConfig.init_from_params(params.pop('run_config'))
        hparams = HParams.init_from_params(params.pop('hparams'))
        params.assert_empty(cls.__name__)
        cls(data_reader=data_reader, model=model, hparams=hparams, run_config=run_config)

    def make_input_fns(self, data: DataReader):
        train_input_fn = data.make_estimator_input_fn(DataSplit.TRAIN, force_repeat=True)
        valid_input_fn = data.make_estimator_input_fn(DataSplit.EVAL, force_repeat=True)
        test_input_fn = data.make_estimator_input_fn(DataSplit.TEST, force_repeat=True)
        return train_input_fn, valid_input_fn, test_input_fn

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

