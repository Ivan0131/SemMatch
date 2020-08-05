import os
import collections
import six
import re
import tensorflow as tf
from semmatch.utils.logger import logger
from tensorflow.python.estimator import model_fn
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import tensor_shape, dtypes


def generate_input_map(signature_def, features, labels=None):
    features_mapping = {"input_query": "premise/tokens", "input_title": "hypothesis/tokens"}
    inputs = signature_def.inputs
    input_map = {}
    for (key, tensor_info) in inputs.items():
        input_name = tensor_info.name
        if ':' in input_name:
            input_name = input_name[:input_name.find(':')]
        control_dependency_name = '^' + input_name
        if features_mapping is not None and key in features_mapping:
            feature_key = features_mapping[key]
        else:
            feature_key = key
        if feature_key in features:
            check_same_dtype_and_shape(features[feature_key], tensor_info, key)
            input_map[input_name] = input_map[control_dependency_name] = features[feature_key]
        elif labels is not None and feature_key in labels:
            check_same_dtype_and_shape(labels[feature_key], tensor_info, key)
            input_map[input_name] = input_map[control_dependency_name] = labels[feature_key]
        else:
            raise ValueError(
                'Key \"%s\" not found in features or labels passed in to the model '
                'function. All required keys: %s' % (feature_key, inputs.keys()))
    return input_map


def check_same_dtype_and_shape(tensor, tensor_info, name):
  """Validate that tensor has the same properties as the TensorInfo proto.
  Args:
    tensor: a `Tensor` object.
    tensor_info: a `TensorInfo` proto.
    name: Name of the input (to identify Tensor if an error is raised).
  Raises:
    ValueError: If the tensor shape or dtype don't match the TensorInfo
  """
  dtype_error = (tensor.dtype != dtypes.DType(tensor_info.dtype))
  shape_error = not tensor.shape.is_compatible_with(tensor_info.tensor_shape)

  if dtype_error or shape_error:
    msg = 'Tensor shape and/or dtype validation failed for input %s:' % name
    if dtype_error:
      msg += ('\n\tExpected dtype: %s, Got: %s'
              % (dtypes.DType(tensor_info.dtype), tensor.dtype))
    if shape_error:
      msg += ('\n\tExpected shape: %s, Got: %s'
              % (tensor_shape.TensorShape(tensor_info.tensor_shape),
                 tensor.shape))

    raise ValueError(msg)


def extract_available_modes(saved_model_loader):
    """Return list of modes found in SavedModel."""
    available_modes = []
    logger.info('Checking available modes.')
    for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
                 tf.estimator.ModeKeys.PREDICT]:
        try:
            get_meta_graph_def_for_mode(saved_model_loader, mode)
        except RuntimeError:
            logger.warning('%s mode not found in SavedModel.' % mode)
            continue

        if get_signature_def_for_mode(saved_model_loader, mode) is not None:
            available_modes.append(mode)

    logger.info('Available modes: %s' % available_modes)
    return available_modes


def get_meta_graph_def_for_mode(saved_model_loader, mode):
    try:
        tags = model_fn.EXPORT_TAG_MAP[mode]
    except AttributeError as e:
        tags = ['serve']
    return saved_model_loader.get_meta_graph_def_from_tags(tags)


def get_signature_def_for_mode(saved_model_loader, mode):
    meta_graph_def = get_meta_graph_def_for_mode(saved_model_loader, mode)
    sig_def_key = (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                   if mode == model_fn.ModeKeys.PREDICT else mode)
    if sig_def_key not in meta_graph_def.signature_def:
        logger.warning('Metagraph for mode %s was found, but SignatureDef with'
                       ' key \"%s\" is missing.' % (mode, sig_def_key))
        return None
    return meta_graph_def.signature_def[sig_def_key]


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, num_of_group=0):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)
    init_vars_name = [name for (name, _) in init_vars]

    if num_of_group > 0:
        assignment_map = []
        for gid in range(num_of_group):
            assignment_map.append(collections.OrderedDict())
    else:
        assignment_map = collections.OrderedDict()

    for name in name_to_variable:
        if name in init_vars_name:
            tvar_name = name
        elif (re.sub(r"/group_\d+/", "/group_0/",
                     six.ensure_str(name)) in init_vars_name and
              num_of_group > 1):
            tvar_name = re.sub(r"/group_\d+/", "/group_0/", six.ensure_str(name))
        elif (re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/ffn_\d+/", "/ffn_1/", six.ensure_str(name))
        elif (re.sub(r"/attention_\d+/", "/attention_1/", six.ensure_str(name))
              in init_vars_name and num_of_group > 1):
            tvar_name = re.sub(r"/attention_\d+/", "/attention_1/",
                               six.ensure_str(name))
        else:
            tf.logging.info("name %s does not get matched", name)
            continue
        tf.logging.info("name %s match to %s", name, tvar_name)
        if num_of_group > 0:
            group_matched = False
            for gid in range(1, num_of_group):
                if (("/group_" + str(gid) + "/" in name) or
                        ("/ffn_" + str(gid) + "/" in name) or
                        ("/attention_" + str(gid) + "/" in name)):
                    group_matched = True
                    tf.logging.info("%s belongs to %dth", name, gid)
                    assignment_map[gid][tvar_name] = name
            if not group_matched:
                assignment_map[0][tvar_name] = name
        else:
            assignment_map[tvar_name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[six.ensure_str(name) + ":0"] = 1

    return (assignment_map, initialized_variable_names)

#
# def get_variables_path(export_dir):
#   """Return the variables path, used as the prefix for checkpoint files."""
#   return os.path.join(
#       compat.as_text(get_variables_dir(export_dir)),
#       compat.as_text(constants.VARIABLES_FILENAME))
#
#
# def get_variables_dir(export_dir):
#   """Return variables sub-directory in the SavedModel."""
#   return os.path.join(
#       compat.as_text(export_dir),
#       compat.as_text(constants.VARIABLES_DIRECTORY))
#
#
# class SavedModelLoader(object):
#   """Load graphs and restore variable values from a `SavedModel`."""
#
#   def __init__(self, export_dir):
#     """Creates a `SavedModelLoader`.
#
#     Args:
#       export_dir: Directory in which the SavedModel protocol buffer and
#         variables to be loaded are located.
#     """
#     self._export_dir = export_dir
#     self._variables_path = get_variables_path(export_dir)
#     self._saved_model = _parse_saved_model(export_dir)
#
#   @property
#   def export_dir(self):
#     """Directory containing the SavedModel."""
#     return self._export_dir
#
#   @property
#   def variables_path(self):
#     """Path to variable checkpoint files."""
#     return self._variables_path
#
#   @property
#   def saved_model(self):
#     """SavedModel object parsed from the export directory."""
#     return self._saved_model
#
#   def get_meta_graph_def_from_tags(self, tags):
#     """Return MetaGraphDef with the exact specified tags.
#
#     Args:
#       tags: A list or set of string tags that identify the MetaGraphDef.
#
#     Returns:
#       MetaGraphDef with the same tags.
#
#     Raises:
#       RuntimeError: if no metagraphs were found with the associated tags.
#     """
#     found_match = False
#     for meta_graph_def in self._saved_model.meta_graphs:
#       if set(meta_graph_def.meta_info_def.tags) == set(tags):
#         meta_graph_def_to_load = meta_graph_def
#         found_match = True
#         break
#
#     if not found_match:
#       raise RuntimeError(
#           "MetaGraphDef associated with tags " + str(tags).strip("[]") +
#           " could not be found in SavedModel. To inspect available tag-sets in"
#           " the SavedModel, please use the SavedModel CLI: `saved_model_cli`"
#       )
#     return meta_graph_def_to_load
#
#   def load_graph(self, graph, tags, import_scope=None, **saver_kwargs):
#     """Load ops and nodes from SavedModel MetaGraph into graph.
#
#     Args:
#       graph: tf.Graph object.
#       tags: a set of string tags identifying a MetaGraphDef.
#       import_scope: Optional `string` -- if specified, prepend this string
#         followed by '/' to all loaded tensor names. This scope is applied to
#         tensor instances loaded into the passed session, but it is *not* written
#         through to the static `MetaGraphDef` protocol buffer that is returned.
#       **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.
#
#     Returns:
#       A tuple of
#         * Saver defined by the MetaGraph, which can be used to restore the
#           variable values.
#         * List of `Operation`/`Tensor` objects returned from
#           `tf.import_graph_def` (may be `None`).
#     """
#     meta_graph_def = self.get_meta_graph_def_from_tags(tags)
#     with graph.as_default():
#       return tf_saver._import_meta_graph_with_return_elements(  # pylint: disable=protected-access
#           meta_graph_def, import_scope=import_scope, **saver_kwargs)
#
#   def restore_variables(self, sess, saver, import_scope=None):
#     """Restore SavedModel variable values into the session.
#
#     Args:
#       sess: tf.Session to restore variable values.
#       saver: a tf.train.Saver object. Can be None if there are no variables in
#         graph. This may be the saver returned by the load_graph() function, or a
#         default `tf.train.Saver()`.
#       import_scope: Optional `string` -- if specified, prepend this string
#         followed by '/' to all loaded tensor names. This scope is applied to
#         tensor instances loaded into the passed session, but it is *not* written
#         through to the static `MetaGraphDef` protocol buffer that is returned.
#
#     Raises:
#       ValueError: if no saver was passed to the saver argument, and there are
#         variables in the graph.
#     """
#     with sess.graph.as_default():
#       if (saver is None and
#           not variables._all_saveable_objects(scope=import_scope)):  # pylint: disable=protected-access
#         tf_logging.info("The specified SavedModel has no variables; no "
#                         "checkpoints were restored.")
#       elif isinstance(saver, tf_saver.Saver):
#         saver.restore(sess, self._variables_path)
#       else:
#         raise ValueError(
#             "No tf.train.Saver object was passed to the function "
#             "SavedModelLoader.restore_variables. Since there are variables in "
#             "the graph, a saver is required.")
#
#   def run_init_ops(self, sess, tags, import_scope=None):
#     """Run initialization ops defined in the `MetaGraphDef`.
#
#     Args:
#       sess: tf.Session to restore variable values.
#       tags: a set of string tags identifying a MetaGraphDef.
#       import_scope: Optional `string` -- if specified, prepend this string
#         followed by '/' to all loaded tensor names. This scope is applied to
#         tensor instances loaded into the passed session, but it is *not* written
#         through to the static `MetaGraphDef` protocol buffer that is returned.
#     """
#     meta_graph_def = self.get_meta_graph_def_from_tags(tags)
#     with sess.graph.as_default():
#       # Get asset tensors, if any.
#       asset_tensors_dictionary = _get_asset_tensors(
#           self._export_dir, meta_graph_def, import_scope=import_scope)
#
#       main_op_tensor = (
#           _get_main_op_tensor(meta_graph_def, constants.MAIN_OP_KEY) or
#           _get_main_op_tensor(meta_graph_def, constants.LEGACY_INIT_OP_KEY))
#       if main_op_tensor is not None:
#         sess.run(fetches=[main_op_tensor], feed_dict=asset_tensors_dictionary)
#
#   def load(self, sess, tags, import_scope=None, **saver_kwargs):
#     """Load the MetaGraphDef graph and restore variable values into the session.
#
#     Args:
#       sess: tf.Session to restore variable values.
#       tags: a set of string tags identifying a MetaGraphDef.
#       import_scope: Optional `string` -- if specified, prepend this string
#         followed by '/' to all loaded tensor names. This scope is applied to
#         tensor instances loaded into the passed session, but it is *not* written
#         through to the static `MetaGraphDef` protocol buffer that is returned.
#       **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.
#
#     Returns:
#       `MetagraphDef` proto of the graph that was loaded.
#     """
#     with sess.graph.as_default():
#       saver, _ = self.load_graph(sess.graph, tags, import_scope,
#                                  **saver_kwargs)
#       self.restore_variables(sess, saver, import_scope)
#       self.run_init_ops(sess, tags, import_scope)
#     return self.get_meta_graph_def_from_tags(tags)
