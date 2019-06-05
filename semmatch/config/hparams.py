from typing import Dict
import tensorflow as tf
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


class HParams(InitFromParams, tf.contrib.training.HParams):
    def __init__(self, task: str = 'classification', rank_loss_margin: float = 1.0, train_steps: int = None, eval_steps: int = 100, test_steps: int = None,
                 early_stopping_max_steps_without_decrease: int = 1000, early_stopping_min_steps: int = 100):
        tf.contrib.training.HParams.__init__(self, task=task, train_steps=train_steps, eval_steps=eval_steps,
                                             rank_loss_margin=rank_loss_margin,
                                             test_steps=test_steps, early_stopping_min_steps=early_stopping_min_steps,
                                             early_stopping_max_steps_without_decrease=early_stopping_max_steps_without_decrease)
