import tensorflow as tf
from semmatch.utils.exception import ConfigureError


def rank_hinge_loss(labels, logits, params):
    num_retrieval = params.get('num_retrieval', None)
    if num_retrieval is None:
        raise ConfigureError("The parameter num_retrieval is not assigned or the dataset is not support rank loss.")
    margin = params.get('rank_loss_margin', 1.0)
    labels = tf.argmax(labels, axis=-1)
    labels = tf.reshape(labels, (-1, num_retrieval))
    logits = tf.reshape(logits, (-1, num_retrieval))
    label_mask = tf.cast(tf.sign(labels), tf.float32)
    label_count = tf.reduce_sum(label_mask, axis=-1)
    y_pos = tf.reduce_sum(label_mask * logits, axis=-1)/label_count
    y_neg = tf.reduce_sum((1.-label_mask) * logits, axis=-1)/(num_retrieval-label_count)
    loss = tf.maximum(0., margin-y_pos+y_neg)
    loss = tf.reduce_mean(loss)
    return loss