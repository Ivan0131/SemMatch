from semmatch.utils import misc_utils

_DATA_READERS = {}


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


def register_data(name=None):
    """Register a data reader. The default name is snake-cased"""

    def decorator(data_cls, registration_name=None):
        """Registers & returns model_cls with registration_name or default name."""
        data_name = registration_name or default_name(data_cls)
        if data_name in _DATA_READERS:
            raise LookupError("Data reader %s already registered." % data_name)
        data_cls.REGISTERED_NAME = data_name
        _DATA_READERS[data_name] = data_cls
        return data_cls

    return lambda data_cls: decorator(data_cls, name)


def data_reader(name):
    if name not in _DATA_READERS:
        raise LookupError("Data reader %s never registered.  Available data readers:\n %s" %
                          (name, "\n".join(list_data_readers())))

    return _DATA_READERS[name]


def list_data_readers():
    return list(sorted(_DATA_READERS))

