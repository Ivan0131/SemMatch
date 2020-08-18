import os
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
from semmatch.modules.embeddings.encoders import Encoder
from semmatch.utils import register
from semmatch.utils.path import get_file_extension
from semmatch.utils.logger import logger
from semmatch.utils.exception import ConfigureError
import tensorflow as tf
from semmatch.data import Vocabulary
import tqdm
import pickle
import gzip


@register.register_subclass("encoder", 'embedding')
class Embedding(Encoder):
    def __init__(self, embedding_dim: int, num_embeddings: int = None, projection_dim: int = None,
                 vocab: Vocabulary = None, vocab_namespace: str = None, dropout_rate: float = 0.0, padding_zero: bool = False,
                 trainable: bool = True, pretrained_file: str = None, tmp_dir: str = None, encoder_name: str ="embedding"):
        super().__init__(encoder_name=encoder_name, vocab_namespace=vocab_namespace)
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._projection_dim = projection_dim
        self._dropout_rate = dropout_rate
        if pretrained_file:
            weight = _read_pretrained_embeddings(pretrained_file, tmp_dir, embedding_dim, vocab, vocab_namespace)
        else:
            weight = None
        self._weight = weight
        self._trainable = trainable
        self._embeddings = None
        self._padding_zero = padding_zero
        self._vocab_namespace = vocab_namespace

    def forward(self, features, labels, mode, params):
        outputs = dict()
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        for (feature_key, feature) in features.items():
            if '/' not in feature_key:
                continue
            feature_namespace = feature_key.split("/")[1].strip()
            if feature_namespace == self._vocab_namespace:
                with tf.variable_scope("embedding/"+self._vocab_namespace, reuse=tf.AUTO_REUSE):
                    if self._weight is None:
                        if not self._trainable:
                            logger.warning("No pretrained embedding is assigned. The embedding should be trainable.")
                        logger.debug("loading random embedding.")
                        if self._padding_zero:
                            word_embeddings = tf.get_variable("embedding_weight", shape=(self._num_embeddings-1, self._embedding_dim),
                                                       initializer=initializers.xavier_initializer(), trainable=self._trainable)
                            pad_embeddings = tf.constant(np.zeros([1, self._embedding_dim]), dtype=tf.float32)
                            self._embeddings = tf.concat([pad_embeddings, word_embeddings], axis=0)
                        else:
                            self._embeddings = tf.get_variable("embedding_weight", shape=(self._num_embeddings, self._embedding_dim),
                                                           initializer=initializers.xavier_initializer(), trainable=self._trainable)
                    else:
                        if self._weight.shape != (self._num_embeddings, self._embedding_dim):
                            raise ConfigureError("The parameter of embedding with shape (%s, %s), "
                                                 "but the pretrained embedding with shape %s."
                                                 %(self._num_embeddings, self._embedding_dim, self._weight.shape))
                        logger.debug("loading pretrained embedding with trainable %s." % self._trainable)
                        if self._padding_zero:
                            word_embeddings = tf.get_variable("embedding_weight",
                                                              initializer=self._weight[1:, :],
                                                              trainable=self._trainable)
                            pad_embeddings = tf.constant(np.zeros([1, self._embedding_dim]), dtype=tf.float32)
                            self._embeddings = tf.concat([pad_embeddings, word_embeddings], axis=0)
                        else:
                            self._embeddings = tf.get_variable("embedding_weight",
                                                           initializer=self._weight, trainable=self._trainable)
                            # tf.Variable(self._weight, trainable=self._trainable, name='embedding_weight')
                    emb = tf.nn.embedding_lookup(self._embeddings, feature)

                    dropout_rate = params.get('dropout_rate')
                    if dropout_rate is None:
                        dropout_rate = self._dropout_rate
                    emb_drop = tf.layers.dropout(emb, dropout_rate, training=is_training)
                    if self._projection_dim:
                        emb_drop = tf.layers.dense(emb_drop, self._projection_dim, use_bias=False, kernel_initializer=initializers.xavier_initializer())
                    outputs[feature_key] = emb_drop
        return outputs

    @classmethod
    def init_from_params(cls, params, vocab):
        num_embeddings = params.pop_int('num_embeddings', None)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop("pretrained_file", None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        dropout_rate = params.pop_float("dropout_rate", 0.0)
        encoder_name = params.pop("encoder_name", "embedding")
        tmp_dir = params.pop("tmp_dir", None)
        vocab_namespace = params.pop('namespace', 'tokens')
        padding_zero = params.pop_bool('padding_zero', 'False')

        params.assert_empty(cls.__name__)
        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   dropout_rate=dropout_rate, padding_zero=padding_zero,
                   pretrained_file=pretrained_file, vocab=vocab, vocab_namespace=vocab_namespace,
                   trainable=trainable, encoder_name=encoder_name, tmp_dir=tmp_dir)


def _read_pretrained_embeddings(pretrained_file, tmp_dir, embedding_dim, vocab, vocab_namespace):
    if not os.path.exists(pretrained_file):
        logger.error("Pretrained embedding file is not existing")
        return None
    if tmp_dir:
        if not os.path.exists(tmp_dir):
            tf.gfile.MakeDirs(tmp_dir)
        cache_embedding_file = os.path.join(tmp_dir, "embedding.pkl.gz")
    else:
        cache_embedding_file = None
    if tmp_dir and os.path.exists(cache_embedding_file):
        logger.info("loading cache embedding from %s." % cache_embedding_file)
        with gzip.open(cache_embedding_file, 'rb') as pkl_file:
            embeddings = pickle.load(pkl_file)
    else:
        file_ext = get_file_extension(pretrained_file)
        if file_ext in ['.txt']:
            embeddings = _read_pretrained_embeddings_text(pretrained_file, embedding_dim, vocab, vocab_namespace)
        else:
            logger.error("Do not support this embedding file type.")
            return None
        if cache_embedding_file:
            with gzip.open(cache_embedding_file, 'wb') as pkl_file:
                pickle.dump(embeddings, pkl_file)
    return embeddings


def _read_pretrained_embeddings_text(pretrained_file, embedding_dim, vocab, vocab_namespace):
    vocab_tokens = vocab.get_vocab_tokens(vocab_namespace)
    vocab_size = vocab.get_vocab_size(vocab_namespace)
    embeddings = {}
    logger.info("Reading pretrained embeddings from: %s" % pretrained_file)
    with open(pretrained_file, 'r', encoding='utf-8') as embeddings_file:
        for line in tqdm.tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in vocab_tokens:
                fields = line.rstrip().split(' ')
                if len(fields) - 1 != embedding_dim:

                    logger.warning("Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                                   embedding_dim, len(fields) - 1, line)
                    continue

                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

    if not embeddings:
        ConfigureError("The embedding_dim or vocabulary does not fit the pretrained embedding.")
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std, (vocab_size, embedding_dim))
    embedding_matrix = embedding_matrix.astype(np.float32)
    num_tokens_found = 0
    index_to_tokens = vocab.get_vocab_index_to_token(vocab_namespace)
    for i in range(vocab_size):
        token = index_to_tokens[i]
        if token in embeddings:
            embedding_matrix[i] = embeddings[token]
            num_tokens_found += 1
        else:
            logger.debug("Token %s was not found in the embedding file. Initialising randomly.", token)

    logger.info("Pretrained embeddings were found for %d out of %d tokens",
                num_tokens_found, vocab_size)
    return embedding_matrix