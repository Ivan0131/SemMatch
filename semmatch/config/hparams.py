from typing import Dict
import tensorflow as tf
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


class HParams(InitFromParams, tf.contrib.training.HParams):
    def __init__(self, task: str = 'classification', task_type: str = 'mulitclass', rank_loss_margin: float = 1.0, train_steps: int = None,
                 eval_steps: int = 100, test_steps: int = None, throttle_secs: int = 600,
                 per_process_gpu_memory_fraction: float = None, threshold: float = 0.5,
                 early_stopping_max_steps_without_decrease: int = 1000, early_stopping_min_steps: int = 100):
        tf.contrib.training.HParams.__init__(self, task=task, task_type=task_type, train_steps=train_steps, eval_steps=eval_steps,
                                             rank_loss_margin=rank_loss_margin, throttle_secs=throttle_secs,
                                             test_steps=test_steps, early_stopping_min_steps=early_stopping_min_steps,
                                             per_process_gpu_memory_fraction = per_process_gpu_memory_fraction, threshold=threshold,
                                             early_stopping_max_steps_without_decrease=early_stopping_max_steps_without_decrease)
