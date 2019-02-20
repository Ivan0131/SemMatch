from semmatch.utils.logger import logger
from tensorflow.python.estimator import model_fn
from tensorflow.python.saved_model import signature_constants
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, dtypes


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
    tags = model_fn.EXPORT_TAG_MAP[mode]
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
