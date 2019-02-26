from semmatch.utils.logger import logger
from semmatch.config.parameters import Parameters
from semmatch.utils import register
from semmatch.config.config_utils import takes_arg, create_kwargs


def init_from_params(cls, params, **extras):
    logger.info(f"instantiating class {cls} from params {getattr(params, 'params', params)} "
                f"and extras {extras}")
    if params is None:
        return

    if isinstance(params, str):
        params = Parameters({"name": params})

    try:
        subclasses = register.get_by_name(cls)
    except LookupError as e:
        logger.debug(e)
        subclasses = None

    if subclasses is not None:
        subclass_name = params.pop_choice("name", subclasses.keys())
        subclass = subclasses[subclass_name]

        if not takes_arg(subclass.init_from_params, 'extras'):
            extras = {k: v for k, v in extras.items() if takes_arg(subclass.init_from_params, k)}

        return subclass.init_from_params(params=params, **extras)
    else:
        if cls.__init__ == object.__init__:
            kwargs = {}
        else:
            kwargs = create_kwargs(cls, params, **extras)
        return cls(**kwargs)


class InitFromParams:
    @classmethod
    def init_from_params(cls, params, **extras):
        return init_from_params(cls, params, **extras)
