import tensorflow as tf
from semmatch.config.init_from_params import InitFromParams
from semmatch.utils import register


@register.register("optimizer")
class Optimizer(InitFromParams):
    def __init__(self, embedding_trainable:bool = True, optimizer_name='optimizer'):
        self._optimizer_name = optimizer_name
        self._embedding_trainable = embedding_trainable
        self._optimizer = None

    def get_or_create_optimizer(self, params):
        raise NotImplementedError

    def get_learning_rate(self, init_learning_rate, train_steps=None, warmup_proportion=None):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=init_learning_rate, shape=[], dtype=tf.float32)
        num_train_steps = train_steps
        if num_train_steps:
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)

        if warmup_proportion and num_train_steps:
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
        return learning_rate

    def get_train_op(self, loss, params):
        self.get_or_create_optimizer(params)
        global_step = tf.train.get_or_create_global_step()
        tvars = tf.trainable_variables()
        embedding_tvars = tf.trainable_variables("embedding")
        model_tvars = list(set(tvars) - set(embedding_tvars))
        model_grads = tf.gradients(loss, model_tvars)
        #model_grads, _ = tf.clip_by_global_norm(model_grads, 10.0)
        train_ops = []
        optimizer_hooks = []

        # This is how the model was pre-trained.
        embedding_grads = tf.gradients(loss, embedding_tvars)
        #embedding_grads, _ = tf.clip_by_global_norm(embedding_grads, 10.0)
        model_train_op = self._model_optimizer.apply_gradients(
            zip(model_grads, model_tvars), global_step=global_step)
        train_ops.append(model_train_op)
        if self._embedding_trainable and len(embedding_grads) and embedding_grads[0] is not None:
            embedding_train_op = self._embedding_optimizer.apply_gradients(
                zip(embedding_grads, embedding_tvars), global_step=global_step)
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
