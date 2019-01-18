from tensorflow.contrib.training import HParams


def basic_data_hparams(HParams):
    def __init__(**kwargs):
        super().__init__(batch_size=1024,
                         batch_shuffle_size=512,







                         )