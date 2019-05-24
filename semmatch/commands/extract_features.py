import os
import tensorflow as tf
from semmatch.commands.command import Command
from semmatch.utils import register
from semmatch.config.parameters import Parameters
from semmatch.utils.exception import ConfigureError
from semmatch.data.data_readers.data_reader import DataSplit, DataReader
from semmatch.models import Model
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.config.run_config import RunConfig
from semmatch.config.hparams import HParams
from semmatch.utils.logger import logger
import simplejson as json


@register.register_subclass('command', 'extract_features')
class ExtractFeatures(Command):
    name = 'extract_features'
    description = 'Extract pre-computed feature vectors from pre-trained model.'
    parser = None

    def __init__(self, data_reader, embeddingmapping, output_file):
        vocab = data_reader.get_vocab()
        input_fn = data_reader.make_estimator_input_fn(DataSplit.PREDICT, force_repeat=False)

        hparams = HParams()
        run_config = RunConfig()
        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(embeddingmapping),
            config=run_config, params=hparams, warm_start_from=embeddingmapping.get_warm_start_setting())

        idx = 0
        with open(output_file, "w") as writer:
            for result in estimator.predict(input_fn, yield_single_examples=True):
                #print(result)
                output_json = dict()
                output_json["linex_index"] = idx
                all_features = []
                for i, (token_idx, token_vector) in enumerate(zip(result['premise/elmo_characters'], result['premise/elmo_characters_embedding'])):
                #for i, (token_idx, token_vector) in enumerate(zip(result['premise/tokens'], result['premise/tokens_embedding'])):
                    #token = vocab.get_index_token(token_idx, namespace='tokens')
                    token_vector = [
                            round(float(x), 6) for x in token_vector
                        ]
                    features = dict()
                    features['token'] = token_idx
                    features['values'] = token_vector
                    all_features.append(features)
                # for (i, token) in enumerate(feature.tokens):
                #     all_layers = []
                #     for (j, layer_index) in enumerate(layer_indexes):
                #         layer_output = result["layer_output_%d" % j]
                #         layers = collections.OrderedDict()
                #         layers["index"] = layer_index
                #         layers["values"] = [
                #             round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                #         ]
                #         all_layers.append(layers)
                #     features = collections.OrderedDict()
                #     features["token"] = token
                #     features["layers"] = all_layers
                #    all_features.append(features)
                #output_json["features"] = all_features
                #writer.write(json.dumps(output_json) + "\n")
                idx += 1

    def get_model_fn(self, embeddingmapping):
        def model_fn(features, labels, mode, params):
            logger.info("****Features****")
            for name in sorted(features.keys()):
                tf.logging.info("name = %s, shape = %s, data_split = %s" % (name, features[name].shape, mode))

            output_dict = embeddingmapping.forward(features, labels, mode, params)

            output_dict_new = {}
            for key, value in output_dict.items():
                output_dict_new[key+"_embedding"] = value

            output_dict_new.update(features)

            output_spec = tf.estimator.EstimatorSpec(mode, predictions=output_dict_new)

            return output_spec
        return model_fn

    @classmethod
    def init_from_params(cls, params):
        #####data reader###############
        data_reader = DataReader.init_from_params(params.pop('data'))
        vocab = data_reader.get_vocab()
        #######embedding###############
        embeddingmapping = EmbeddingMapping.init_from_params(params.pop('embedding_mapping'), vocab=vocab)
        output_file = params.pop('output_file')
        #####data reader###############
        params.assert_empty(cls.__name__)
        cls(data_reader=data_reader, embeddingmapping=embeddingmapping, output_file=output_file)

    @classmethod
    def add_subparser(cls, parser):
        cls.parser = parser.add_parser(name=cls.name, description=cls.description, help='extract features')
        cls.parser.add_argument('--config_path', type=str,
                                help='the config path where store the params.')
        cls.parser.set_defaults(func=cls.init_from_args)
        return cls.parser

    @classmethod
    def init_from_args(cls, args):
        config_path = args.config_path
        cls.init_from_config_file(config_path)

    @classmethod
    def init_from_config_file(cls, config_path):
        params = Parameters.init_from_file(config_path)
        return cls.init_from_params(params)

