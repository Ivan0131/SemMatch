import tensorflow as tf
from semmatch.utils import register
from semmatch.modules.optimizers import Optimizer


@register.register_subclass('optimizer', 'adagrad')
class AdagradOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.05,
                 initial_accumulator_value=0.1,
                 warmup_proportion: float = None,
                 embedding_learning_rate: float = None,
                 embedding_trainable: bool = True,
                 decay_steps: int = None, decay_rate: float = None, decay_type='polynomial',
                 optimizer_name: str = 'adagrad_optimizer'):
        super().__init__(optimizer_name=optimizer_name, embedding_trainable=embedding_trainable)
        self._learning_rate = learning_rate
        self._initial_accumulator_value = initial_accumulator_value
        self._warmup_proportion = warmup_proportion
        if embedding_learning_rate is None:
            self._embedding_learning_rate = learning_rate
        else:
            self._embedding_learning_rate = embedding_learning_rate
        self._model_optimizer = None
        self._embedding_optimizer = None
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._decay_type = decay_type

    def get_or_create_optimizer(self, params):
        if self._optimizer:
            return self._optimizer
        else:
            learning_rate = self.get_learning_rate(self._learning_rate, params.train_steps, self._warmup_proportion,
                                                   self._decay_steps, self._decay_rate, self._decay_type)
            embedding_learning_rate = self.get_learning_rate(self._embedding_learning_rate, params.train_steps,
                                                             self._warmup_proportion, self._decay_steps,
                                                             self._decay_rate, self._decay_type)

            self._model_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                                              initial_accumulator_value=self._initial_accumulator_value,
                                                               name="adagrad_model_optimizer")

            self._embedding_optimizer = tf.train.AdagradOptimizer(learning_rate=embedding_learning_rate,
                                                                  initial_accumulator_value=self._initial_accumulator_value,
                                                                  name="adagrad_embedding_optimizer")
            return self._optimizer, self._embedding_optimizer