from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


@register.register("encoder")
class Encoder(InitFromParams):
    def __init__(self, vocab_namespace, encoder_name='encoder'):
        self._vocab_namespace = vocab_namespace
        self._encoder_name = encoder_name

    def get_namespace(self):
        return self._vocab_namespace

    def forward(self, features, labels, mode, params):
        raise NotImplementedError

    def get_warm_start_setting(self):
        return None

