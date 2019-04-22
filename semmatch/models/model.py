import sys
import tensorflow as tf
from semmatch.utils import register
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError, ModelError
from semmatch.config.init_from_params import InitFromParams
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings import EmbeddingMapping
#from tensorflow.python import debug as tf_debug


@register.register("model")
class Model(InitFromParams):
    def __init__(self, embedding_mapping: EmbeddingMapping, optimizer: Optimizer=AdamOptimizer(), model_name: str="model"):
        self._model_name = model_name
        self._optimizer = optimizer
        self._embedding_mapping = embedding_mapping

    def get_warm_start_setting(self):
        return self._embedding_mapping.get_warm_start_setting()

    def forward(self, features, labels, mode, params):
        raise NotImplementedError

    def make_estimator_model_fn(self):
        def model_fn(features, labels, mode, params):
            logger.info("****Features****")
            for name in sorted(features.keys()):
                tf.logging.info("name = %s, shape = %s, data_split = %s" % (name, features[name].shape, mode))

            output_dict = self.forward(features, labels, mode, params)

            if mode == tf.estimator.ModeKeys.TRAIN:
                if 'loss' not in output_dict:
                    raise ModelError("Please provide loss in the model outputs for %s dataset."%mode)
                train_op, optimizer_hooks = self._optimizer.get_train_op(output_dict['loss'], params)
                #optimizer_hooks.append(tf_debug.LocalCLIDebugHook())
                ##########
                if 'debugs' in output_dict:
                    tvars = output_dict['debugs'] #tf.trainable_variables()
                    print_ops = []
                    for op in tvars:
                        op_name = op.name
                        #op = tf.debugging.is_nan(tf.reduce_mean(op))
                        print_ops.append(tf.print(op.name, op, output_stream=sys.stdout))

                    print_op = tf.group(*print_ops)
                    train_op = tf.group(print_op, train_op)
                ########
                output_spec = tf.estimator.EstimatorSpec(mode, loss=output_dict['loss'], train_op=train_op,
                                                         export_outputs=output_dict.get("export_outputs", None),
                                                         predictions=output_dict.get('predictions', None),
                                                         training_hooks=optimizer_hooks,
                                                         eval_metric_ops=output_dict.get('metrics', None))

            elif mode == tf.estimator.ModeKeys.EVAL:
                output_spec = tf.estimator.EstimatorSpec(mode, export_outputs=output_dict.get("export_outputs", None),
                                                         loss=output_dict.get('loss', None),
                                                         predictions=output_dict.get('predictions', None),
                                                         eval_metric_ops=output_dict.get('metrics', None))
            elif mode == tf.estimator.ModeKeys.PREDICT:
                output_spec = tf.estimator.EstimatorSpec(mode, export_outputs=output_dict.get("export_outputs", None),
                                                         predictions=output_dict.get('predictions', None))
            else:
                raise ValueError("Mode %s are not supported."%mode)
            return output_spec
        return model_fn
