from typing import List
import tensorflow as tf
import re
from semmatch.utils import register
from semmatch.modules.optimizers import Optimizer, MyDecoupledWeightDecayExtension


@register.register_subclass('optimizer', 'adam')
class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-08, warmup_proportion: float = None,
                 weight_decay_rate: float = 0.0, embedding_learning_rate: float = None,
                 decay_steps: int = None, decay_rate: float = None, decay_type='polynomial',
                 embedding_trainable: bool = True,
                 exclude_from_weight_decay: List[str] = None,
                 optimizer_name: str = 'adam_optimizer'):
        super().__init__(learning_rate=learning_rate, embedding_learning_rate=embedding_learning_rate,
                         optimizer_name=optimizer_name, embedding_trainable=embedding_trainable,
                         weight_decay_rate=weight_decay_rate, exclude_from_weight_decay=exclude_from_weight_decay)
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._warmup_proportion = warmup_proportion
        self._model_optimizer = None
        self._embedding_optimizer = None
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._decay_type = decay_type

    def get_or_create_optimizer(self, params):
        if self._optimizer:
            return self._optimizer
        else:
            schedule = self.get_schedule(params.train_steps, self._warmup_proportion,
                                         self._decay_steps, self._decay_rate, self._decay_type)
            learning_rate = self._learning_rate * schedule

            embedding_learning_rate = self._embedding_learning_rate * schedule

            self._model_optimizer = AdamWOptimizer(
                learning_rate=learning_rate,
                weight_decay=self._weight_decay_rate,
                beta1=self._beta1,
                beta2=self._beta2,
                epsilon=self._epsilon,
                name="adamw_model_optimizer")

            self._embedding_optimizer = AdamWOptimizer(
                learning_rate=embedding_learning_rate,
                weight_decay=self._weight_decay_rate,
                beta1=self._beta1,
                beta2=self._beta2,
                epsilon=self._epsilon,
                name="adamw_embedding_optimizer")
        return self._optimizer, self._embedding_optimizer


class AdamWOptimizer(MyDecoupledWeightDecayExtension, tf.train.AdamOptimizer):
    def __init__(self, weight_decay, learning_rate: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-08,
                 name='adadelta_weight_decay_optimizer'):
        super(AdamWOptimizer, self).__init__(weight_decay, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                             epsilon=epsilon, name=name)


# class AdamWeightDecayOptimizer(tf.train.Optimizer):
#   """A basic Adam optimizer that includes "correct" L2 weight decay."""
#
#   def __init__(self,
#                learning_rate,
#                weight_decay_rate=0.0,
#                beta_1=0.9,
#                beta_2=0.999,
#                epsilon=1e-6,
#                name="AdamWeightDecayOptimizer"):
#     """Constructs a AdamWeightDecayOptimizer."""
#     super(AdamWeightDecayOptimizer, self).__init__(False, name)
#
#     self.learning_rate = learning_rate
#     self.weight_decay_rate = weight_decay_rate
#     self.beta_1 = beta_1
#     self.beta_2 = beta_2
#     self.epsilon = epsilon
#     self._decay_var_list = None
#
#   def apply_gradients(self, grads_and_vars, global_step=None, decay_var_list=None, name=None):
#     """See base class."""
#     self._decay_var_list = set(decay_var_list) if decay_var_list else None
#
#     assignments = []
#     for (grad, param) in grads_and_vars:
#       if grad is None or param is None:
#         continue
#
#       param_name = self._get_variable_name(param.name)
#
#       m = tf.get_variable(
#           name=param_name + "/adam_m",
#           shape=param.shape.as_list(),
#           dtype=tf.float32,
#           trainable=False,
#           initializer=tf.zeros_initializer())
#       v = tf.get_variable(
#           name=param_name + "/adam_v",
#           shape=param.shape.as_list(),
#           dtype=tf.float32,
#           trainable=False,
#           initializer=tf.zeros_initializer())
#
#       # Standard Adam update.
#       next_m = (
#           tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
#       next_v = (
#           tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
#                                                     tf.square(grad)))
#
#       update = next_m / (tf.sqrt(next_v) + self.epsilon)
#
#       # Just adding the square of the weights to the loss function is *not*
#       # the correct way of using L2 regularization/weight decay with Adam,
#       # since that will interact with the m and v parameters in strange ways.
#       #
#       # Instead we want ot decay the weights in a manner that doesn't interact
#       # with the m/v parameters. This is equivalent to adding the square
#       # of the weights to the loss with plain (non-momentum) SGD.
#       if not self._decay_var_list or param in self._decay_var_list:
#       #if self._do_use_weight_decay(param_name):
#         update += self.weight_decay_rate * param
#
#       update_with_lr = self.learning_rate * update
#
#       next_param = param - update_with_lr
#
#       assignments.extend(
#           [param.assign(next_param),
#            m.assign(next_m),
#            v.assign(next_v)])
#     return tf.group(*assignments, name=name)







