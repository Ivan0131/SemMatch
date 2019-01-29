from typing import Dict
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

    def forward(self, features, labels, mode, params):
        for vocab_namespace in self._encoders:
            self._encoders[vocab_namespace].reset_status()
        outputs = dict()
        feature_keys = features.keys()
        for feature_key in feature_keys:
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
