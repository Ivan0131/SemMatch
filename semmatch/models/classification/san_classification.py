import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders import Bert
from semmatch import nn
from functools import reduce
from operator import mul


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train, scope='bn')
        return activation_func(linear_map)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')


def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


def _linear(xs, output_size, bias, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs, -1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size, output_size], dtype=tf.float32, )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list()  # original shape
    tensor_shape = tensor.get_shape().as_list()  # current shape
    ref_stop = len(ref_shape) - keep  # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]  #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]  #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def query_encode_san(rep_tensor, rep_mask, direction=None, keep_prob=1., is_train=True,
                     wd=0., activation='elu'):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1. / scale * x)

    bs, ql, vec = rep_tensor.get_shape()  # 32, 20, 300: bs, ql, vec
    with tf.variable_scope('query_encode_san_%s' % direction):
        # mask generation
        indices = tf.range(ql, dtype=tf.int32)
        col, row = tf.meshgrid(indices, indices)
        if direction == 'forward':
            direct_mask = tf.greater(row, col)
        else:
            direct_mask = tf.greater(col, row)  # ql, ql
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [tf.shape(rep_tensor)[0], 1, 1])  # bs, ql, ql
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, ql, 1])  # bs, ql, ql
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs, ql, ql

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, vec, True, 0., 'bn_dense_map', activation,
                                 False, keep_prob, is_train)  # bs, ql, vec
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, ql, 1, 1])  # bs, ql, ql, vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)  # bs, ql, vec

        # attention
        with tf.variable_scope('attention'):  # bs, sl, ql, ql, vec
            f_bias = tf.get_variable('f_bias', [vec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, vec, False, scope='linear_dependent')  # bs, ql, vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs, 1, ql, vec
            head = linear(rep_map_dp, vec, False, scope='linear_head')  # bs, ql, vec
            head_etd = tf.expand_dims(head, 2)  # bs, ql, 1, vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs, ql, ql, vec

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)  # bs, ql, ql, vec
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs, ql, ql, vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)  # bs, ql, ql, vec

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs, ql, vec

        return attn_result


def fusion_gate(rep_tensor, rep_mask, fw, bw, keep_prob=1., is_train=True, wd=0.):
    with tf.variable_scope('fusion'):
        bs, ql, vec = rep_tensor.get_shape()  # 32, 20, 300: bs, ql, vec
        o_bias = tf.get_variable('o_bias',[vec], tf.float32, tf.constant_initializer(0.))
        gate = tf.nn.sigmoid(
            linear(fw, vec, True, 0., 'linear_fusion_fw', False, keep_prob, is_train) +
            linear(bw, vec, True, 0., 'linear_fusion_bw', False, keep_prob, is_train) +
            o_bias)
        output = gate * fw + (1 - gate) * bw  # bs, ql, vec
        output = mask_for_high_rank(output, rep_mask)  # bs, ql, vec
        return output


def query_encode_md(rep_tensor, rep_mask, keep_prob=1., is_train=True, wd=0., activation='elu'):
    bs, ql, vec = rep_tensor.get_shape()  # 32, 20, 300: bs, ql, vec
    with tf.variable_scope('query_encode_md'):
        map1 = bn_dense_layer(rep_tensor, vec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)  # bs, ql, vec
        map2 = bn_dense_layer(map1, vec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)  # bs, ql, vec
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)  # bs, ql, vec

        soft = tf.nn.softmax(map2_masked, 1)  # bs, ql, vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec
        return attn_output  # bs, vec


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)# for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    return out


@register.register_subclass('model', 'san_cls')
class SAN_CLS(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 hidden_dim: int = 300, dropout_rate:float = 0.5,
                 model_name: str = 'bilstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_rate = dropout_rate

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            premise_tokens_ids = features.get('premise/tokens', None)
            if premise_tokens_ids is None:
                premise_tokens_ids = features.get('premise/elmo_characters', None)

            if premise_tokens_ids is None:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens "
                                     "or elmo_characters.")

            prem_seq_lengths, prem_mask = nn.length(premise_tokens_ids)
            if features.get('premise/elmo_characters', None) is not None or isinstance(self._embedding_mapping.get_encoder('tokens'), Bert):
                prem_mask = nn.remove_bos_eos(prem_mask, prem_seq_lengths)
                prem_seq_lengths -= 2

            #prem_mask = tf.expand_dims(prem_mask, -1)
            prem_mask = tf.cast(prem_mask, tf.bool)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)

            with tf.variable_scope('san_fb1'):
                x_fw1 = query_encode_san(premise_tokens, prem_mask, 'forward')  # bs, ql, vec
                x_bw1 = query_encode_san(premise_tokens, prem_mask, 'backward')  # bs, ql, vec
                x_fusion = fusion_gate(premise_tokens, prem_mask, x_fw1, x_bw1)  # bs, ql, vec
            with tf.variable_scope('san_md'):
                x_code = query_encode_md(x_fusion, prem_mask)  # bs, vec

                pre_logits = tf.nn.relu(
                    linear(x_code,  self._hidden_dim, True, scope='pre_logits_linear', is_train=True))  # bs, vec
                logits = linear(pre_logits, self._num_classes, False, scope='get_output', is_train=True)  # bs, cn

            output_dict = self._make_output(logits, params)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels_embedding = features_embedding['label/labels']
                labels = features['label/labels']

                loss = self._make_loss(labels=labels_embedding, logits=output_dict['logits'], params=params)
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=output_dict['predictions'])
                metrics['precision'] = tf.metrics.precision(labels=labels, predictions=output_dict['predictions'])
                metrics['recall'] = tf.metrics.recall(labels=labels, predictions=output_dict['predictions'])
                    #tf.metrics.auc(labels=labels, predictions=predictions)
                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
            return output_dict
