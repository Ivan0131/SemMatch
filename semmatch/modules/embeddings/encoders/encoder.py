from semmatch.config.init_from_params import InitFromParams
from semmatch.utils import register


@register.register("encoder")
class Encoder(InitFromParams):
    def __init__(self, encoder_name='encoder'):
        self._encoder_name = encoder_name

    def forward(self, features, labels, mode, params):
        raise NotImplementedError
