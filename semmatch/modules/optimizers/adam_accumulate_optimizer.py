from typing import List
import tensorflow as tf
import re
from semmatch.utils import register
from semmatch.modules.optimizers import Optimizer


@register.register_subclass('optimizer', 'adam_accum')
class AdamAccumulateOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-08, accum_iters: int = 1,
                 amsgrad: bool = False,
                 warmup_proportion: float = None,
                 weight_decay_rate: float = 0.0, embedding_learning_rate: float = None,
                 embedding_trainable: bool = True,
                 exclude_from_weight_decay: List[str] = None,
                 optimizer_name: str = 'adam_optimizer'):
        super().__init__(optimizer_name=optimizer_name, embedding_trainable=embedding_trainable)
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._accum_iters = accum_iters
        self._amsgrad = amsgrad
        self._warmup_proportion = warmup_proportion
        self._weight_decay_rate = weight_decay_rate
        self._exclude_from_weight_decay = exclude_from_weight_decay
        if embedding_learning_rate is None:
            self._embedding_learning_rate = learning_rate
        else:
            self._embedding_learning_rate = embedding_learning_rate
        self._model_optimizer = None
        self._embedding_optimizer = None

    def get_or_create_optimizer(self, params):
        if self._optimizer:
            return self._optimizer
        else:
            learning_rate = self.get_learning_rate(self._learning_rate, params.train_steps, self._warmup_proportion)
            embedding_learning_rate = self.get_learning_rate(self._embedding_learning_rate, params.train_steps,
                                                             self._warmup_proportion)
            if self._weight_decay_rate == 0:
                self._model_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=self._beta1,
                                                               beta2=self._beta2, epsilon=self._epsilon,
                                                               name="adam_model_optimizer")

                self._embedding_optimizer = tf.train.AdamOptimizer(learning_rate=embedding_learning_rate,
                                                                   beta1=self._beta1,
                                                                   beta2=self._beta2, epsilon=self._epsilon,
                                                                   name="adam_embedding_optimizer")
            else:
                self._model_optimizer = AdamAccumulateWeightDecayOptimizer(
                    learning_rate=learning_rate,
                    weight_decay_rate=self._weight_decay_rate,
                    beta_1=self._beta_1,
                    beta_2=self._beta_2,
                    epsilon=self._beta_3,
                    amsgrad=self._amsgrad,
                    accum_iters=self._accum_iters,
                    exclude_from_weight_decay=self._exclude_from_weight_decay, name="adamw_model_optimizer")

                self._embedding_optimizer = AdamAccumulateWeightDecayOptimizer(
                    learning_rate=embedding_learning_rate,
                    weight_decay_rate=self._weight_decay_rate,
                    beta_1=self._beta_1,
                    beta_2=self._beta_2,
                    epsilon=self._beta_3,
                    amsgrad=self._amsgrad,
                    accum_iters=self._accum_iters,
                    exclude_from_weight_decay=self._exclude_from_weight_decay, name="adamw_embedding_optimizer")
            return self._optimizer, self._embedding_optimizer


class AdamAccumulateWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 accum_iters=1,
                 amsgrad=False,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamAccumulateWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.accum_iters = accum_iters
        self.amsgrad = amsgrad

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()

        ms = [tf.get_variable(
                name=self._get_variable_name(param.name) + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()) for (grad, param) in grads_and_vars]
        vs = [tf.get_variable(
                name=self._get_variable_name(param.name) + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()) for (grad, param) in grads_and_vars]
        gs = [tf.get_variable(
                name=self._get_variable_name(param.name) + "/adam_g",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()) for (grad, param) in grads_and_vars]

        if self.amsgrad:
            vhats = [tf.get_variable(
                name=self._get_variable_name(param.name) + "/adam_vhat",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()) for (grad, param) in grads_and_vars]
        else:
            vhats = [tf.get_variable(
                name=self._get_variable_name(param.name) + "/adam_vhat",
                shape=1,
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()) for (grad, param) in grads_and_vars]

        flag = tf.equal(self.iterations % self.accum_iters, 0)
        flag = tf.cast(flag, dtype='float32')

        for (grad, param), m, v, vhat, gg in zip(grads_and_vars, ms, vs, vhats, gs):
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            sum_grad = gg + grad
            avg_grad = sum_grad / self.accum_iters

            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * tf.square(avg_grad)

            if self.amsgrad:
                vhat_t = tf.maximum(vhat, v_t)
                update = m_t / (tf.sqrt(vhat_t) + self.epsilon)
                assignments.extend(vhat.assign((1-flag)*vhat+flag*vhat_t))
            else:
                update = m_t / (tf.sqrt(v_t) + self.epsilon)
            #p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

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

            next_param = param - flag * update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(flag * m_t + (1 - flag) * m),
                 v.assign(flag * v_t + (1 - flag) * v),
                 gg.assign((1-flag) * sum_grad)])
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
