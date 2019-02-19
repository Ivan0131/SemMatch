from typing import Dict
import tensorflow as tf
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


class HParams(InitFromParams, tf.contrib.training.HParams):
    def __init__(self, train_steps: int = None, eval_steps: int = 100, test_steps: int = None):
        tf.contrib.training.HParams.__init__(self, train_steps=train_steps, eval_steps=eval_steps,
                                             test_steps=test_steps)
