import collections
from semmatch.utils import misc_utils

_NAME_TO_SUBCLASSES = collections.defaultdict()
_CLASS_TO_SUBCLASSES = collections.defaultdict()


def default_name(obj_class):
    """Convert a class name to the registry's default name for the class.

    Args:
      obj_class: the name of a class

    Returns:
      The registry's default name for the class.
    """
    return misc_utils.camel_to_snake(obj_class.__name__)


def default_object_name(obj):
    """Convert an object to the registry's default name for the object class.
    Args:
      obj: an object instance
    Returns:
      The registry's default name for the class of the object.
    """
    return default_name(obj.__class__)


def register(name):
    """Register a base class. """
    def decorator(base_cls, registration_name):
        base_cls_name = registration_name or default_name(base_cls)
        if base_cls_name in _NAME_TO_SUBCLASSES:
            raise LookupError("Base class %s already registered" % base_cls_name)
        base_cls.REGISTERED_NAME = base_cls_name
        base_cls_subclasses = {}
        _NAME_TO_SUBCLASSES[base_cls_name] = base_cls_subclasses
        _CLASS_TO_SUBCLASSES[base_cls] = base_cls_subclasses
        return base_cls
    if callable(name):
        base_cls = name
        return decorator(base_cls, registration_name=default_name(base_cls))
    return lambda base_cls: decorator(base_cls, name)


def register_subclass(base_cls, name):
    def decorator(sub_cls, base_cls, registration_name):
        sub_cls_name = registration_name or default_name(sub_cls)
        if isinstance(base_cls, str):
            if base_cls not in _NAME_TO_SUBCLASSES:
                raise LookupError("Base class %s is not registered" % base_cls)
            if sub_cls_name in _NAME_TO_SUBCLASSES[base_cls]:
                raise LookupError("Subclass %s is already registered" % sub_cls_name)
        else:
            if base_cls not in _CLASS_TO_SUBCLASSES:
                raise LookupError("Base class %s is not registered" % default_name(base_cls))
            if sub_cls_name in _CLASS_TO_SUBCLASSES[base_cls]:
                raise LookupError("Subclass %s is already registered" % sub_cls_name)
        sub_cls.REGISTERED_NAME = sub_cls_name
        if isinstance(base_cls, str):
            _NAME_TO_SUBCLASSES[base_cls][sub_cls_name] = sub_cls
        else:
            _CLASS_TO_SUBCLASSES[base_cls][sub_cls_name] = sub_cls
        return sub_cls

    return lambda sub_cls: decorator(sub_cls, base_cls, name)


def get_by_name(base_cls=None, name=None):
    if base_cls is None and name is not None:
        base_cls_name = base_cls if isinstance(base_cls, str) else default_name(base_cls)
        raise LookupError("Sub class %s is provided, but base class name %s is not assigned" % (name, base_cls_name))
    if isinstance(base_cls, str):
        if base_cls not in _NAME_TO_SUBCLASSES:
            raise LookupError("Base class %s is not registered" % base_cls)
        if name:
            if name not in _NAME_TO_SUBCLASSES[base_cls]:
                raise LookupError("Subclass %s is not registered in base class %s" % (name, base_cls))
            return _NAME_TO_SUBCLASSES[base_cls][name]
        else:
            return _NAME_TO_SUBCLASSES[base_cls]
    else:
        if base_cls not in _CLASS_TO_SUBCLASSES:
            raise LookupError("Base class %s is not registered" % default_name(base_cls))
        if name:
            if name not in _CLASS_TO_SUBCLASSES[base_cls]:
                raise LookupError("Subclass %s is not registered in base class %s" % (name, default_name(base_cls)))
            return _CLASS_TO_SUBCLASSES[base_cls][name]
        else:
            return _CLASS_TO_SUBCLASSES[base_cls]


def list_available(base_cls=None, name=None):
    if base_cls is None and name is not None:
        base_cls_name = base_cls if isinstance(base_cls, str) else default_name(base_cls)
        raise LookupError("Sub class %s is provided, but base class name %s is not assigned" % (name, base_cls_name))
    if name is None:
        if isinstance(base_cls, str):
            return list(sorted(_NAME_TO_SUBCLASSES[base_cls]))
        else:
            return list(sorted(_CLASS_TO_SUBCLASSES[base_cls]))
    else:
        if isinstance(base_cls, str):
            return _NAME_TO_SUBCLASSES[base_cls][name]
        else:
            return _CLASS_TO_SUBCLASSES[base_cls][name]

#
# def register_data(name=None):
#     """Register a data reader. The default name is snake-cased"""
#
#     def decorator(data_cls, registration_name=None):
#         """Registers & returns model_cls with registration_name or default name."""
#         data_name = registration_name or default_name(data_cls)
#         if data_name in _DATA_READERS:
#             raise LookupError("Data reader %s already registered." % data_name)
#         data_cls.REGISTERED_NAME = data_name
#         _DATA_READERS[data_name] = data_cls
#         if data_name.endswith("_data_reader"):
#             data_name = data_name[:len(data_name)-len("_data_reader")]
#             _DATA_READERS[data_name] = data_cls
#         return data_cls
#     if callable(name):
#         data_cls = name
#         return decorator(data_cls, registration_name=default_name(data_cls))
#     return lambda data_cls: decorator(data_cls, name)
#
#
# def data_reader(name):
#     if name not in _DATA_READERS:
#         raise LookupError("Data reader %s never registered.  Available data readers:\n %s" %
#                           (name, "\n".join(list_data_readers())))
#
#     return _DATA_READERS[name]
#
#
# def list_data_readers():
#     return list(sorted(_DATA_READERS))
#
# #
# # def register_command(name=None):
# #     """Register a data reader. The default name is snake-cased"""
# #
# #     def decorator(command_cls, registration_name=None):
# #         """Registers & returns model_cls with registration_name or default name."""
# #         command_name = registration_name or default_name(command_cls)
# #         if command_name in _COMMANDS:
# #             raise LookupError("Command %s already registered." % command_name)
# #         command_cls.REGISTERED_NAME = command_name
# #         _COMMANDS[command_name] = command_cls
# #         return command_cls
# #
# #     if callable(name):
# #         command_cls = name
# #         return decorator(command_cls, registration_name=default_name(command_cls))
# #     return lambda command_cls: decorator(command_cls, name)
# #
# #
# # def command(name):
# #     if name not in _COMMANDS:
# #         raise LookupError("Command %s never registered.  Available commands:\n %s" %
# #                           (name, "\n".join(list_commands())))
# #
# #     return _COMMANDS[name]
# #
# #
# # def list_commands():
# #     return list(sorted(_COMMANDS))
# #
# #
# # def register_token_indexer(name=None):
# #     """Register a token indexer. The default name is snake-cased"""
# #
# #     def decorator(token_indexer_cls, registration_name=None):
# #         """Registers & returns model_cls with registration_name or default name."""
# #         token_indexer_name = registration_name or default_name(token_indexer_cls)
# #         if token_indexer_name in _TOKEN_INDEXERS:
# #             raise LookupError("Token indexer %s already registered." % token_indexer_name)
# #         token_indexer_cls.REGISTERED_NAME = token_indexer_name
# #         _TOKEN_INDEXERS[token_indexer_name] = token_indexer_cls
# #         if token_indexer_name.endswith("_token_indexer"):
# #             token_indexer_name = token_indexer_name[:len(token_indexer_name)-len("_token_indexer")]
# #             _TOKEN_INDEXERS[token_indexer_name] = token_indexer_cls
# #         return token_indexer_cls
# #     if callable(name):
# #         token_indexer_cls = name
# #         return decorator(token_indexer_cls, registration_name=default_name(token_indexer_cls))
# #     return lambda token_indexer_cls: decorator(token_indexer_cls, name)
# #
# #
# # def token_indexer(name):
# #     if name not in _DATA_READERS:
# #         raise LookupError("Token index %s never registered.  Available token indexers:\n %s" %
# #                           (name, "\n".join(list_token_indexers())))
# #
# #     return _TOKEN_INDEXERS[name]
# #
# #
# # def list_token_indexers():
# #     return list(sorted(_TOKEN_INDEXERS))