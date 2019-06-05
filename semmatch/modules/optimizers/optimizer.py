from typing import List
import re
import tensorflow as tf
from semmatch.config.init_from_params import InitFromParams
from semmatch.utils import register
from semmatch.utils.logger import logger
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops


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


class DecoupledWeightDecayExtension(object):
    """This class allows to extend optimizers with decoupled weight decay.
    It implements the decoupled weight decay described by Loshchilov & Hutter
    (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
    decoupled from the optimization steps w.r.t. to the loss function.
    For SGD variants, this simplifies hyperparameter search since it decouples
    the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield better
    training loss and generalization error in the paper above.
    This class alone is not an optimizer but rather extends existing
    optimizers with decoupled weight decay. We explicitly define the two examples
    used in the above paper (SGDW and AdamW), but in general this can extend
    any OptimizerX by using
    `extend_with_weight_decay(OptimizerX, weight_decay=weight_decay)`.
    In order for it to work, it must be the first class the Optimizer with
    weight decay inherits from, e.g.
    ```python
    class AdamWOptimizer(DecoupledWeightDecayExtension, adam.AdamOptimizer):
      def __init__(self, weight_decay, *args, **kwargs):
        super(AdamWOptimizer, self).__init__(weight_decay, *args, **kwargs).
    ```
    Note that this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of'var' in the update step!

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:
    ```python
      schedule = tf.train.piecewise_constant(tf.train.get_global_step(),
                                             [10000, 15000], [1e-0, 1e-1, 1e-2])
      lr = 1e-1 * schedule()
      wd = lambda: 1e-4 * schedule()
      # ...
      optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=lr,
                                                    weight_decay=wd,
                                                    momentum=0.9,
                                                    use_nesterov=True)
    ```
    """

    def __init__(self, weight_decay, **kwargs):
        """Construct the extension class that adds weight decay to an optimizer.
        Args:
          weight_decay: A `Tensor` or a floating point value, the factor by which
            a variable is decayed in the update step.
          **kwargs: Optional list or tuple or set of `Variable` objects to
            decay.
        """
        self._decay_var_list = None  # is set in minimize or apply_gradients
        self._weight_decay = weight_decay
        # The tensors are initialized in call to _prepare
        self._weight_decay_tensor = None
        super(DecoupledWeightDecayExtension, self).__init__(**kwargs)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=optimizer.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False,
                 name=None, grad_loss=None, decay_var_list=None):
        """Add operations to minimize `loss` by updating `var_list` with decay.
        This function is the same as Optimizer.minimize except that it allows to
        specify the variables that should be decayed using decay_var_list.
        If decay_var_list is None, all variables in var_list are decayed.
        For more information see the documentation of Optimizer.minimize.
        Args:
          loss: A `Tensor` containing the value to minimize.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          var_list: Optional list or tuple of `Variable` objects to update to
            minimize `loss`.  Defaults to the list of variables collected in
            the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
          gate_gradients: How to gate the computation of gradients.  Can be
            `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
          aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
          colocate_gradients_with_ops: If True, try colocating gradients with
            the corresponding op.
          name: Optional name for the returned operation.
          grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
          decay_var_list: Optional list of decay variables.
        Returns:
          An Operation that updates the variables in `var_list`.  If `global_step`
          was not `None`, that operation also increments `global_step`.
        """
        self._decay_var_list = set(decay_var_list) if decay_var_list else False
        return super(DecoupledWeightDecayExtension, self).minimize(
            loss, global_step=global_step, var_list=var_list,
            gate_gradients=gate_gradients, aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
            grad_loss=grad_loss)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None,
                        decay_var_list=None):
        """Apply gradients to variables and decay the variables.
        This function is the same as Optimizer.apply_gradients except that it
        allows to specify the variables that should be decayed using
        decay_var_list. If decay_var_list is None, all variables in var_list
        are decayed.
        For more information see the documentation of Optimizer.apply_gradients.
        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.
          decay_var_list: Optional list of decay variables.
        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.
        """
        self._decay_var_list = set(decay_var_list) if decay_var_list else False
        return super(DecoupledWeightDecayExtension, self).apply_gradients(
            grads_and_vars, global_step=global_step, name=name)

    def _prepare(self):
        weight_decay = self._weight_decay
        if callable(weight_decay):
            weight_decay = weight_decay()
        self._weight_decay_tensor = ops.convert_to_tensor(
            weight_decay, name="weight_decay")
        # Call the optimizers _prepare function.
        super(DecoupledWeightDecayExtension, self)._prepare()

    def _decay_weights_op(self, var):
        if not self._decay_var_list or var in self._decay_var_list:
            return var.assign_sub(self._weight_decay * var, self._use_locking)
        return control_flow_ops.no_op()

    def _decay_weights_sparse_op(self, var, indices, scatter_add):
        if not self._decay_var_list or var in self._decay_var_list:
            update = -self._weight_decay * array_ops.gather(var, indices)
            return scatter_add(var, indices, update, self._use_locking)
        return control_flow_ops.no_op()

    # Here, we overwrite the apply functions that the base optimizer calls.
    # super().apply_x resolves to the apply_x function of the BaseOptimizer.
    def _apply_dense(self, grad, var):
        with ops.control_dependencies([self._decay_weights_op(var)]):
            return super(DecoupledWeightDecayExtension, self)._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        with ops.control_dependencies([self._decay_weights_op(var)]):
            return super(DecoupledWeightDecayExtension, self)._resource_apply_dense(
                grad, var)

    def _apply_sparse(self, grad, var):
        scatter_add = state_ops.scatter_add
        decay_op = self._decay_weights_sparse_op(var, grad.indices, scatter_add)
        with ops.control_dependencies([decay_op]):
            return super(DecoupledWeightDecayExtension, self)._apply_sparse(
                grad, var)

    def _resource_scatter_add(self, x, i, v, _=None):
        # last argument allows for one overflow argument, to have the same function
        # signature as state_ops.scatter_add
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        scatter_add = self._resource_scatter_add
        decay_op = self._decay_weights_sparse_op(var, indices, scatter_add)
        with ops.control_dependencies([decay_op]):
            return super(DecoupledWeightDecayExtension, self)._resource_apply_sparse(
                grad, var, indices)


class MyDecoupledWeightDecayExtension(DecoupledWeightDecayExtension):
    def __init__(self, weight_decay, learning_rate, **kwargs):
        weight_decay *= learning_rate
        super(MyDecoupledWeightDecayExtension, self).__init__(learning_rate=learning_rate, weight_decay=weight_decay,
                                                              **kwargs)
