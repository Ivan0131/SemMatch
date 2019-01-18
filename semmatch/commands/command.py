from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


@register.register('command')
class Command(InitFromParams):
    name = "command"
    description = ""
    parser = None

    @classmethod
    def add_subparser(cls, parser):
        raise NotImplementedError
