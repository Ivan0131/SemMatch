from typing import Dict
import tensorflow as tf
from tensorflow.python.estimator.run_config import _USE_DEFAULT
from semmatch.utils import register
from semmatch.config.init_from_params import InitFromParams


class RunConfig(InitFromParams, tf.estimator.RunConfig):
    def __init__(self, model_dir: str = None, random_seed: int = None, save_summary_steps: int = 100,
                 save_checkpoints_steps: int = None,
                 save_checkpoints_secs: int = None,
                 session_config: Dict = None, keep_checkpoint_max: int = 5,
                 keep_checkpoint_every_n_hours: int = 10000, log_step_count_steps: int = 100,
                 train_distribute=None, device_fn=None, protocol=None,
                 eval_distribute=None, experimental_distribute=None):
        if save_checkpoints_steps is None:
            save_checkpoints_steps = _USE_DEFAULT
        if save_checkpoints_secs is None:
            save_checkpoints_secs = _USE_DEFAULT
        tf.estimator.RunConfig.__init__(self, model_dir=model_dir,
                                        tf_random_seed=random_seed,
                                        save_summary_steps=save_summary_steps,
                                        save_checkpoints_steps=save_checkpoints_steps,
                                        save_checkpoints_secs=save_checkpoints_secs,
                                        session_config=session_config,
                                        keep_checkpoint_max=keep_checkpoint_max,
                                        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                                        log_step_count_steps=log_step_count_steps,
                                        train_distribute=train_distribute,
                                        device_fn=device_fn,
                                        protocol=protocol,
                                        eval_distribute=eval_distribute,
                                        experimental_distribute=experimental_distribute)
