import os
import tensorflow as tf
from semmatch.utils.logger import logger
from tensorflow.python.estimator import model_fn
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.saved_model.loader_impl import _parse_saved_model, _get_asset_tensors, _get_main_op_tensor
from tensorflow.python.util import compat


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
