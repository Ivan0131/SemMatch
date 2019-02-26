from typing import Dict
import tensorflow as tf
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams
from semmatch.modules.embeddings.encoders import Encoder
from semmatch.utils.exception import ConfigureError
from semmatch.utils.logger import logger


@register.register('embedding_mapping')
class EmbeddingMapping(InitFromParams):
    def forward(self, features, labels, mode, params):
        raise NotImplementedError


@register.register_subclass('embedding_mapping', 'base')
class BaseEmbeddingMapping(EmbeddingMapping):
    def __init__(self, encoders: Dict[str, Encoder]):
        self._encoders = encoders

    def get_warm_start_setting(self):
        warm_start_settings = None
        for (namespace, encoder) in self._encoders.items():
            warm_start_settings_namespace = self._encoders[namespace].get_warm_start_setting()
            if isinstance(warm_start_settings_namespace, tf.estimator.WarmStartSettings):
                if warm_start_settings is None:
                    warm_start_settings = warm_start_settings_namespace
                else:
                    logger.warning("There are two pretrained embedding, which is not supported int this toolkit now.")
        return warm_start_settings

    def forward(self, features, labels, mode, params):
        logger.debug("****Embeddings****")
        outputs = dict()
        feature_keys = features.keys()
        for feature_key in feature_keys:
            logger.debug("%s:" % feature_key)
            feature_vocab_namespace = feature_key.split("/")[1]
            if feature_vocab_namespace in self._encoders:
                outputs[feature_key] = self._encoders[feature_vocab_namespace].forward(features.get(feature_key, None),
                                                                                      labels,
                                                                                      mode, params)
            else:
                logger.warning("The embedding mapping of feature %s is not assigned, so the outputs of embedding will "
                               "not contain this feature" % feature_key)
        return outputs

    @classmethod
    def init_from_params(cls, params, vocab):
        token_embedder_params = params.pop('encoders', None)

        if token_embedder_params is not None:
            token_embedders = {
                name: Encoder.init_from_params(subparams, vocab=vocab, vocab_namespace=name)
                for name, subparams in token_embedder_params.items()
            }
        else:
            raise ConfigureError("The parameters of embeddings is not provided.")

        params.assert_empty(cls.__name__)
        return cls(token_embedders)
