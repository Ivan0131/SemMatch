import tensorflow as tf
from semmatch.config.init_from_params import InitFromParams
from semmatch.utils import register


@register.register("optimizer")
class Optimizer(InitFromParams):
    def __init__(self, optimizer_name='optimizer'):
        self._optimizer_name = optimizer_name
        self._optimizer = None

    def get_or_create_optimizer(self, params):
        raise NotImplementedError

    def get_train_op(self, loss, params):
        self.get_or_create_optimizer(params)
        global_step = tf.train.get_or_create_global_step()
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        # This is how the model was pre-trained.
        #(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = self._optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)
        new_global_step = global_step + 1
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])
        return train_op
