"""
This is basic utils function for instantiating class from parameters. It is implemented by AllenNLP
"""

import inspect
from typing import TypeVar, Type, Dict, Union, Any, cast, List, Tuple, Set
from semmatch.utils.exception import NotFoundError, ConfigureError
from semmatch.config.parameters import Parameters
from semmatch.utils import register

_NO_DEFAULT = inspect.Parameter.empty


def takes_arg(obj, arg: str) -> bool:
    """
Checks whether the provided obj takes a certain arg.
If it's a class, we're really checking whether its constructor does.
If it's a function or method, we're checking the object itself.
Otherwise, we raise an error.
"""
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigureError(f"object {obj} is not callable")
    return arg in signature.parameters


def remove_optional(annotation: type):
    """
Optional[X] annotations are actually represented as Union[X, NoneType].
For our purposes, the "Optional" part is not interesting, so here we
throw it away.
"""
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        return annotation


def create_kwargs(cls, params, **extras):
    """
Given some class, a `Params` object, and potentially other keyword arguments,
create a dict of keyword args suitable for passing to the class's constructor.

The function does this by finding the class's constructor, matching the constructor
arguments to entries in the `params` object, and instantiating values for the parameters
using the type annotation and possibly a init_from_params method.

Any values that are provided in the `extras` will just be used as is.
For instance, you might provide an existing `Vocabulary` this way.
"""
    # Get the signature of the constructor.
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}

    # Iterate over all the constructor parameters and their annotations.
    for name, param in signature.parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if name == "self":
            continue

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])

        # The parameter is optional if its default value is not the "no default" sentinel.
        default = param.default
        optional = default != _NO_DEFAULT

        # Some constructors expect extra non-parameter items, e.g. vocab: Vocabulary.
        # We check the provided `extras` for these and just use them if they exist.
        if name in extras:
            kwargs[name] = extras[name]

        # The next case is when the parameter type is itself constructible init_from_params.
        elif hasattr(annotation, 'init_from_params'):
            if name in params:
                # Our params have an entry for this, so we use that.
                subparams = params.pop(name)

                if takes_arg(annotation.init_from_params, 'extras'):
                    # If annotation.params accepts **extras, we need to pass them all along.
                    # For example, `BasicTextFieldEmbedder.init_from_params` requires a Vocabulary
                    # object, but `TextFieldEmbedder.init_from_params` does not.
                    subextras = extras
                else:
                    # Otherwise, only supply the ones that are actual args; any additional ones
                    # will cause a TypeError.
                    subextras = {k: v for k, v in extras.items() if takes_arg(annotation.init_from_params, k)}

                # In some cases we allow a string instead of a param dict, so
                # we need to handle that case separately.
                if isinstance(subparams, str):
                    kwargs[name] = register.get_by_name(annotation, subparams)()
                else:
                    kwargs_value = annotation.init_from_params(params=subparams, **subextras)
                    if kwargs_value:
                        kwargs[name] = kwargs_value
            elif not optional:
                # Not optional and not supplied, that's an error!
                raise NotFoundError(f"expected key {name} for {cls.__name__}")
            else:
                kwargs[name] = default

        # If the parameter type is a Python primitive, just pop it off
        # using the correct casting pop_xyz operation.
        elif annotation == str:
            kwargs[name] = (params.pop(name, default)
                            if optional
                            else params.pop(name))
        elif annotation == int:
            kwargs[name] = (params.pop_int(name, default)
                            if optional
                            else params.pop_int(name))
        elif annotation == bool:
            kwargs[name] = (params.pop_bool(name, default)
                            if optional
                            else params.pop_bool(name))
        elif annotation == float:
            kwargs[name] = (params.pop_float(name, default)
                            if optional
                            else params.pop_float(name))

        # This is special logic for handling types like Dict[str, TokenIndexer],
        # List[TokenIndexer], Tuple[TokenIndexer, Tokenizer], and Set[TokenIndexer],
        # which it creates by instantiating each value init_from_params and returning the resulting structure.
        elif origin in (Dict, dict) and len(args) == 2 and hasattr(args[-1], 'init_from_params'):
            value_cls = annotation.__args__[-1]

            value_dict = {}

            for key, value_params in params.pop(name, Parameters({})).items():
                value_dict[key] = value_cls.init_from_params(params=value_params, **extras)

            kwargs[name] = value_dict

        elif origin in (List, list) and len(args) == 1 and hasattr(args[0], 'init_from_params'):
            value_cls = annotation.__args__[0]

            value_list = []

            for value_params in params.pop(name, Parameters({})):
                value_list.append(value_cls.init_from_params(params=value_params, **extras))

            kwargs[name] = value_list

        elif origin in (Tuple, tuple) and all(hasattr(arg, 'init_from_params') for arg in args):
            value_list = []

            for value_cls, value_params in zip(annotation.__args__, params.pop(name, Parameters({}))):
                value_list.append(value_cls.init_from_params(params=value_params, **extras))

            kwargs[name] = tuple(value_list)

        elif origin in (Set, set) and len(args) == 1 and hasattr(args[0], 'init_from_params'):
            value_cls = annotation.__args__[0]

            value_set = set()

            for value_params in params.pop(name, Parameters({})):
                value_set.add(value_cls.init_from_params(params=value_params, **extras))

            kwargs[name] = value_set

        else:
            # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
            if optional:
                kwargs[name] = params.pop(name, default)
            else:
                kwargs[name] = params.pop(name)

    params.assert_empty(cls.__name__)
    return kwargs
