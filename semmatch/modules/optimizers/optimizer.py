from typing import List
import re
import tensorflow as tf
from semmatch.config.init_from_params import InitFromParams
from semmatch.utils import register
from semmatch.utils.logger import logger
from tensorflow.contrib.opt import DecoupledWeightDecayExtension


@register.register("optimizer")
class Optimizer(InitFromParams):
    def __init__(self,
                 learning_rate: float = 0.001,
                 embedding_learning_rate: float = None,
                 embedding_trainable: bool = True,
                 exclude_from_weight_decay: List[str] = None,
                 weight_decay_rate: float = 0.0,
                 optimizer_name='optimizer'):
        self._optimizer_name = optimizer_name
        self._embedding_trainable = embedding_trainable
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self._weight_decay_rate = weight_decay_rate
        if embedding_learning_rate is None:
            self._embedding_learning_rate = learning_rate
        else:
            self._embedding_learning_rate = embedding_learning_rate
        self._optimizer = None

    def get_or_create_optimizer(self, params):
        raise NotImplementedError

    def get_schedule(self, train_steps=None, warmup_proportion=None, decay_steps=None,
                     decay_rate=None, decay_type='polynomial'):
        global_step = tf.train.get_or_create_global_step()
        rate = tf.constant(value=1.0, shape=[], dtype=tf.float32)
        num_train_steps = train_steps
        if decay_type == 'polynomial':
            if num_train_steps:
                rate = tf.train.polynomial_decay(
                    rate,
                    global_step,
                    num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
        elif decay_type == 'exponential':
            if decay_steps and decay_rate:
                rate = tf.train.exponential_decay(
                    rate,
                    global_step,
                    decay_steps,
                    decay_rate,
                    staircase=False, name=None)
        else:
            logger.error("The decay type %s is not supported." % decay_type)

        if warmup_proportion and num_train_steps:
            num_warmup_steps = int(num_train_steps * self._warmup_proportion)
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_rate = rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            rate = (
                    (1.0 - is_warmup) * rate + is_warmup * warmup_rate)
        return rate

    def get_train_op(self, loss, params):
        self.get_or_create_optimizer(params)
        global_step = tf.train.get_or_create_global_step()
        tvars = tf.trainable_variables()
        embedding_tvars = tf.trainable_variables("embedding")
        model_tvars = list(set(tvars) - set(embedding_tvars))
        model_grads = tf.gradients(loss, model_tvars)
        # model_grads, _ = tf.clip_by_global_norm(model_grads, 10.0)
        train_ops = []
        optimizer_hooks = []

        # This is how the model was pre-trained.
        embedding_grads = tf.gradients(loss, embedding_tvars)
        # embedding_grads, _ = tf.clip_by_global_norm(embedding_grads, 10.0)
        if self._weight_decay_rate:
            model_decay_var_list = [var for var in model_tvars
                                    if self._do_use_weight_decay(self._get_variable_name(var.name))]
        else:
            model_decay_var_list = None

        model_train_op = self._model_optimizer.apply_gradients(
            zip(model_grads, model_tvars), global_step=global_step, decay_var_list=model_decay_var_list)

        train_ops.append(model_train_op)
        if self._embedding_trainable and len(embedding_grads) and embedding_grads[0] is not None:
            if self._weight_decay_rate:
                embedding_decay_var_list = [var for var in embedding_tvars
                                            if self._do_use_weight_decay(self._get_variable_name(var.name))]
            else:
                embedding_decay_var_list = None

            embedding_train_op = self._embedding_optimizer.apply_gradients(
                zip(embedding_grads, embedding_tvars), global_step=global_step, decay_var_list=embedding_decay_var_list)
            train_ops.append(embedding_train_op)

        new_global_step = global_step + 1
        train_op = tf.group(train_ops, [global_step.assign(new_global_step)])
        # with tf.control_dependencies([train_op]):
        #     ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        #     if self._embedding_trainable:
        #         train_op = ema.apply(tvars)
        #     else:
        #         train_op = ema.apply(model_tvars)
        return train_op, optimizer_hooks

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay_rate:
            return False
        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class MyDecoupledWeightDecayExtension(DecoupledWeightDecayExtension):
    def __init__(self, weight_decay, learning_rate, **kwargs):
        weight_decay *= learning_rate
        super(MyDecoupledWeightDecayExtension, self).__init__(learning_rate=learning_rate, weight_decay=weight_decay,
                                                              **kwargs)