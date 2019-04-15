from typing import List
import tensorflow as tf
import re
from semmatch.utils import register
from semmatch.modules.optimizers import Optimizer


@register.register_subclass('optimizer', 'momentum')
class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 0.0001, momentum: float = 0.9, use_nesterov: bool = False,
                 warmup_proportion: float = None,
                 embedding_learning_rate: float = None,
                 embedding_trainable: bool = True,
                 optimizer_name: str = 'adam_optimizer'):
        super().__init__(optimizer_name=optimizer_name, embedding_trainable=embedding_trainable)
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._use_nesterov = use_nesterov
        self._warmup_proportion = warmup_proportion
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

            self._model_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self._momentum,
                                                               use_nesterov=self._use_nesterov,
                                                               name="momentum_model_optimizer")

            self._embedding_optimizer = tf.train.MomentumOptimizer(learning_rate=embedding_learning_rate,
                                                                   momentum=self._momentum,
                                                                   use_nesterov=self._use_nesterov,
                                                                   name="momentum_embedding_optimizer")
            return self._optimizer, self._embedding_optimizer
