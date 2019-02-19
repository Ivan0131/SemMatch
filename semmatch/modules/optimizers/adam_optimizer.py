from typing import List
import tensorflow as tf
import re
from semmatch.utils import register
from semmatch.modules.optimizers import Optimizer


@register.register_subclass('optimizer', 'adam')
class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.001, warmup_proportion: float = None,
                 weight_decay_rate: float = 0.0,
                 exclude_from_weight_decay: List[str] = None,
                 optimizer_name: str = 'adam_optimizer'):
        super().__init__(optimizer_name=optimizer_name)
        self._learning_rate = learning_rate
        self._warmup_proportion = warmup_proportion
        self._weight_decay_rate = weight_decay_rate
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self._optimizer = None

    def get_or_create_optimizer(self, params):
        if self._optimizer:
            return self._optimizer
        else:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.constant(value=self._learning_rate, shape=[], dtype=tf.float32)
            num_train_steps = params.train_steps
            if num_train_steps:
                learning_rate = tf.train.polynomial_decay(
                    learning_rate,
                    global_step,
                    num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
            if self._warmup_proportion and num_train_steps:
                num_warmup_steps = int(num_train_steps * self._warmup_proportion)
                global_steps_int = tf.cast(global_step, tf.int32)
                warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

                warmup_percent_done = global_steps_float / warmup_steps_float
                warmup_learning_rate = self._learning_rate * warmup_percent_done

                is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
                learning_rate = (
                        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

            self._optimizer = AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                weight_decay_rate=self._weight_decay_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=self._exclude_from_weight_decay)
            return self._optimizer


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name





