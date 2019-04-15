import collections
import yaml
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError
from tensorflow.contrib.training import HParams

# SPECIFIC_FIELDS = set(["name"])


def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Parameters(collections.MutableMapping):
    """
    The parameters of in this library. The params contains hparams for tensorflow model and other params for our library
    class. hparams is hte harams for tensorflow model, remove the parameters for our library class from params.
    """
    def __init__(self, params=None, hparams=None):
        self._params = params or collections.defaultdict(dict)
        # self._flatten_params = flatten(self._params, "", "/")
        # self._hparams = hparams or collections.defaultdict(dict)

    @classmethod
    def init_from_file(cls, params_filename):
        logger.info("loading config file from %s"%params_filename)
        with open(params_filename, 'r', encoding='utf-8') as yaml_file:
            params = yaml.load(yaml_file)
        #cls.replace_special_names(params)
        #cls.solve_conflict(params)
        logger.info(params)
        return cls(params)

    def get(self, path, default=None):
        keys = path.strip().split("/")
        node = self._params
        for key in keys:
            node = self._params.get(key, None)
            if node:
                continue
            else:
                return default
        if isinstance(node, dict):
            return Parameters(node)
        else:
            return node

    def pop(self, path, default=None):
        keys = path.strip().split("/")
        node = self._params
        for key in keys:
            node = self._params.pop(key, None)
            if node:
                continue
            else:
                return default
        if isinstance(node, dict):
            return Parameters(node)
        else:
            return node

    def pop_int(self, path, default=None):
        value = self.pop(path, default)
        if value and not isinstance(value, Parameters):
            return int(value)
        else:
            return value

    def pop_float(self, path, default=None):
        value = self.pop(path, default)
        if value and not isinstance(value, Parameters):
            return float(value)
        else:
            return value

    def pop_bool(self, path, default=None):
        value = self.pop(path, default)
        if value and not isinstance(value, Parameters):
            if isinstance(value, str):
                if value.strip().lower() == "false":
                    return False
                elif value.strip().lower() == "true":
                    return True
                else:
                    return bool(value)
            else:
                return bool(value)
        else:
            return value

    def pop_choice(self, path, choice, default=None):
        value = self.pop(path, default)
        if value not in choice:
            raise ConfigureError("value %s get by key %s is not in acceptable choices %s" % (value, path, str(choice)))
        return value

    def assert_empty(self, class_name):
        if self._params:
            raise ConfigureError("Extra parameters are provided %s for class %s" % (str(self._params), class_name))

    def __getitem__(self, key):
        value = self.get(key, None)
        if value:
            return value
        else:
            return KeyError

    def __setitem__(self, key, value):
        self._params[key] = value

    def __delitem__(self, key):
        del self._params[key]

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    # def get_params(self):
    #     params = collections.defaultdict()
    #     for namespace_params in self._params:
    #         params.update(namespace_params)
    #     return HParams(params)


    # def get_params_by_namespace(self, namespace):
    #     return HParams(self._params[namespace])

    # @staticmethod
    # def solve_conflict(params):
    #     params_set = set()
    #     params_to_namespace = dict()
    #     for (namespace, namespace_params) in params.items():
    #         new_namespace_params = collections.defaultdict()
    #         for (param_name, param_value) in namespace_params.items():
    #             if param_name not in params_set:
    #                 params_set.add(param_name)
    #                 params_to_namespace[param_name] = namespace
    #                 new_namespace_params[param_name] = param_value
    #             else:
    #                 new_param_name = namespace + "_" + param_name
    #                 new_namespace_params[new_param_name] = param_value
    #                 logger.warning("parameter %s from %s is the same name as papameter from %s. We relace it with %s."
    #                                %(param_name, namespace, params_to_namespace[param_name], new_param_name))
    #         params[namespace] = new_namespace_params
    #     return params

    # @staticmethod
    # def replace_special_names(params):
    #     for (namespace, namespace_params) in params.items():
    #         new_namespace_params = collections.defaultdict()
    #         for (param_name, param_value) in namespace_params.items():
    #             if param_name in SPECIFIC_FIELDS:
    #                 new_namespace_params[namespace+"_"+param_name] = param_value
    #             else:
    #                 new_namespace_params[param_name] = param_value
    #         params[namespace] = new_namespace_params
    #     return params



    # @classmethod
    # def init_from_args(cls, prog="semmatch"):
    #