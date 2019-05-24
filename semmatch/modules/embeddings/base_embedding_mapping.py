from typing import Dict, List
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

    def get_encoder(self, name):
        raise NotImplementedError


@register.register_subclass('embedding_mapping', 'base')
class BaseEmbeddingMapping(EmbeddingMapping):
    def __init__(self, encoders: List[Encoder]):
        self._encoders = {encoder.get_namespace(): encoder for encoder in encoders}

    def get_encoder(self, name):
        return self._encoders[name]

    def get_warm_start_setting(self):
        warm_start_settings = None
        for encoder in self._encoders.values():
            warm_start_settings_namespace = encoder.get_warm_start_setting()
            if isinstance(warm_start_settings_namespace, tf.estimator.WarmStartSettings):
                if warm_start_settings is None:
                    warm_start_settings = warm_start_settings_namespace
                else:
                    logger.warning("There are two pretrained embedding, which is not supported int this toolkit now.")
        return warm_start_settings

    def forward(self, features, labels, mode, params):
        logger.debug("****Embeddings****")
        feature_keys = features.keys()
        for feature_key in feature_keys:
            logger.debug("%s:" % feature_key)

        outputs = dict()
        for encoder in self._encoders.values():
            outputs_namespace = encoder.forward(features, labels, mode, params)
            outputs.update(outputs_namespace)
        return outputs

    @classmethod
    def init_from_params(cls, params, vocab):
        token_embedder_params = params.pop('encoders', None)

        if token_embedder_params is not None:
            token_embedders = [
                Encoder.init_from_params(subparams, vocab=vocab)
                for name, subparams in token_embedder_params.items()
            ]
            # if isinstance(token_embedder_params, Dict):
            #
            # else:
            #     token_embedders = [
            #         Encoder.init_from_params(subparams, vocab=vocab)
            #         for subparams in token_embedder_params
            #     ]
        else:
            raise ConfigureError("The parameters of embeddings is not provided.")

        params.assert_empty(cls.__name__)
        return cls(token_embedders)
