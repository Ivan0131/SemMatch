import sys
import tensorflow as tf
from semmatch.utils import register
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError, ModelError
from semmatch.config.init_from_params import InitFromParams
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.utils.saved_model import get_assignment_map_from_checkpoint
from semmatch.nn.loss import rank_hinge_loss, focal_loss, multilabel_categorical_crossentropy, \
    multilabel_categorical_crossentropy_topk, GHM_Loss, GHM_Loss2
from tensorflow.python import debug as tf_debug


@register.register("model")
class Model(InitFromParams):
    def __init__(self, embedding_mapping: EmbeddingMapping, optimizer: Optimizer = AdamOptimizer(), init_checkpoint=None,
                 model_name: str = "model"):
        self._model_name = model_name
        self._optimizer = optimizer
        self._embedding_mapping = embedding_mapping
        self._init_checkpoint = init_checkpoint
        self._bool_init_checkpoint = False

    def get_warm_start_setting(self):
        return self._embedding_mapping.get_warm_start_setting()

    def forward(self, features, labels, mode, params):
        raise NotImplementedError

    def _make_output(self, inputs, params):
        task = params.get('task', 'classification')
        task_type = params.get('task_type', 'multiclass')

        if task == 'classification':
            logits = tf.contrib.layers.fully_connected(inputs, self._num_classes, activation_fn=None, scope='logits')
            if task_type == 'multiclass':
                predictions = tf.cast(tf.argmax(logits, -1), tf.int32)
                output_score = tf.nn.softmax(logits, -1)
            elif task_type == 'multilabel':
                threshold = params.get('threshold', 0.5)
                output_score = tf.sigmoid(logits)
                predictions = tf.cast(tf.greater(output_score, threshold), tf.int32)
            elif task_type == 'topk':
                output_score = logits #tf.nn.softmax(logits, -1)
                predictions = tf.cast(tf.greater(logits, 0), tf.int32) #tf.cast(tf.argmax(logits, -1), tf.int32) #
               # predictions = tf.one_hot(predictions, depth=self._num_classes, axis=-1, dtype=tf.int32)
            else:
                raise ConfigureError("Task type %s is not support for task %s. "
                                     "Only multiclass and multilabel is support for task %s" % (task_type, task, task))
        elif task == 'rank':
            logits = tf.contrib.layers.fully_connected(inputs, 1, activation_fn=None, scope='logits')
            predictions = logits
            output_score = logits
        else:
            raise ConfigureError(
                "Task %s is not support. Only task and classification tasks are supported" % task)

        output_dict = {'logits': logits, 'predictions': predictions}
        output_score = tf.estimator.export.PredictOutput(output_score)
        output_predictions = tf.estimator.export.PredictOutput(predictions)
        export_outputs = {"output_score": output_score}
        output_dict['export_outputs'] = export_outputs
        return output_dict

    def _make_loss(self, logits, labels, params):
        task = params.get('task', 'classification')
        task_type = params.get('task_type', 'multiclass')
        if task == 'classification':
            if task_type == 'multiclass':
                #loss = GHM_Loss().ghm_class_loss(logits=logits, targets=labels)
                loss = focal_loss(logits=logits, labels=labels)
                # loss = tf.reduce_mean(
                #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
            elif task_type == 'multilabel':
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            elif task_type == 'topk':
                loss = multilabel_categorical_crossentropy(labels=labels, logits=logits)
            else:
                raise ConfigureError("Task type %s is not support for task %s. "
                                     "Only multiclass and multilabel is support for task %s" % (task_type, task, task))

        elif task == 'rank':
            loss = rank_hinge_loss(labels=labels, logits=logits, params=params)
        else:
            raise ConfigureError(
                "Task %s is not support. Only task and classification tasks are supported" % task)
        return loss

    def make_estimator_model_fn(self):
        def model_fn(features, labels, mode, params):
            logger.info("****Features****")
            for name in sorted(features.keys()):
                tf.logging.info("name = %s, shape = %s, data_split = %s" % (name, features[name].shape, mode))

            output_dict = self.forward(features, labels, mode, params)
            ########################################################################################
            if not self._bool_init_checkpoint:
                initialized_variable_names = {}
                tvars = tf.trainable_variables()
                if self._init_checkpoint:
                    (assignment_map, initialized_variable_names
                     ) = get_assignment_map_from_checkpoint(tvars,  self._init_checkpoint)

                    tf.train.init_from_checkpoint( self._init_checkpoint, assignment_map)
                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)
                self._bool_init_checkpoint = True
            ########################################################################################
            if mode == tf.estimator.ModeKeys.TRAIN:
                if 'loss' not in output_dict:
                    raise ModelError("Please provide loss in the model outputs for %s dataset." % mode)
                train_op, optimizer_hooks = self._optimizer.get_train_op(output_dict['loss'], params)
                # optimizer_hooks.append(tf_debug.LocalCLIDebugHook())
                ##########
                if 'debugs' in output_dict:
                    tvars = output_dict['debugs']  # tf.trainable_variables()
                    print_ops = []
                    for op in tvars:
                        op_name = op.name
                        # op = tf.debugging.is_nan(tf.reduce_mean(op))
                        print_ops.append(tf.print(op.name, op, output_stream=sys.stdout))

                    print_op = tf.group(*print_ops)
                    train_op = tf.group(print_op, train_op)
                # ########
                # tvars = tf.trainable_variables()
                # tgrads = tf.gradients(output_dict['loss'], tvars)
                # lambda_est = []
                # for grad, var in zip(tgrads, tvars):
                #     tmp = tf.reduce_mean(tf.maximum(-grad*var / 2 * var ** 2, 0))
                #     lambda_est.append(tmp)
                # lambda_est = tf.stack(lambda_est, axis=0)
                # lambda_est = tf.reduce_mean(lambda_est)
                # lambda_mean, lambda_update = tf.metrics.mean(lambda_est)
                # print_op = tf.print("l2 weight decay lambda value", lambda_mean, output_stream=sys.stdout)
                # train_op = tf.group(train_op, print_op, lambda_update)
                # #########
                # ########
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
                raise ValueError("Mode %s are not supported." % mode)
            return output_spec

        return model_fn
