import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch import nn
from tensorflow.python.ops.rnn_cell import LayerRNNCell, LSTMStateTuple
from tensorflow.python.keras.utils import tf_utils
from semmatch.utils.logger import logger
from tensorflow.python.eager import context
from tensorflow.python.layers import base as base_layer
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


@register.register_subclass('model', 'mlstm')
class MatchLSTM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer = AdamOptimizer(),
                 hidden_dim: int = 300, keep_prob: float = 0.5,
                 model_name: str = 'mlstm'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_prob = 1 - keep_prob

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            premise_tokens_ids = features.get('premise/tokens', None)
            if premise_tokens_ids is None:
                premise_tokens_ids = features.get('premise/elmo_characters', None)
            hypothesis_tokens_ids = features.get('hypothesis/tokens', None)
            if hypothesis_tokens_ids is None:
                hypothesis_tokens_ids = features.get('hypothesis/elmo_characters', None)

            if premise_tokens_ids is None:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens "
                                     "or elmo_characters.")
            if hypothesis_tokens_ids is None:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens "
                                     "or elmo_characters.")

            prem_seq_lengths, prem_mask = nn.length(premise_tokens_ids)
            hyp_seq_lengths, hyp_mask = nn.length(hypothesis_tokens_ids)
            if features.get('premise/elmo_characters', None) is not None:
                prem_mask = prem_mask[:, 1:-1]
                hyp_mask = hyp_mask[:, 1:-1]
            prem_mask = tf.expand_dims(prem_mask, -1)
            hyp_mask = tf.expand_dims(hyp_mask, -1)

            premise_tokens = features_embedding.get('premise/tokens', None)
            if premise_tokens is None:
                premise_tokens = features_embedding.get('premise/elmo_characters', None)
            hypothesis_tokens = features_embedding.get('hypothesis/tokens', None)
            if hypothesis_tokens is None:
                hypothesis_tokens = features_embedding.get('hypothesis/elmo_characters', None)

            h_s, c1 = nn.lstm(premise_tokens, self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            h_t, c2 = nn.lstm(hypothesis_tokens, self._hidden_dim, seq_len=hyp_seq_lengths,
                              name='hypothesis')

        lstm_m = MatchLSTMCell(self._hidden_dim, h_s, prem_mask)

        k_m, _ = tf.nn.dynamic_rnn(lstm_m, h_t, hyp_seq_lengths, dtype=tf.float32)

        k_valid = select(k_m, hyp_seq_lengths)
        logits = tf.layers.Dense(self._num_classes)(k_valid)

        predictions = tf.argmax(logits, -1)

        output_dict = {'logits': logits, 'predictions': predictions}

        probs = tf.nn.softmax(logits, -1)
        output_score = tf.estimator.export.PredictOutput(probs)
        export_outputs = {"output_score": output_score}
        output_dict['export_outputs'] = export_outputs

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            if 'label/labels' not in features:
                raise ConfigureError("The input features should contain label with vocabulary namespace "
                                     "labels int %s dataset." % mode)
            labels_embedding = features_embedding['label/labels']
            labels = features['label/labels']

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_embedding, logits=logits))
            output_dict['loss'] = loss
            metrics = dict()
            metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=predictions)
            metrics['precision'] = tf.metrics.precision(labels=labels, predictions=predictions)
            metrics['recall'] = tf.metrics.recall(labels=labels, predictions=predictions)
            # metrics['auc'] = tf.metrics.auc(labels=labels, predictions=predictions)
            output_dict['metrics'] = metrics
            # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
            #                          premise_ave, hypothesis_ave, diff, mul, h, h_mlp, logits]
        return output_dict


def select(parameters, length):
    """Select the last valid time step output as the sentence embedding
    :params parameters: [batch, seq_len, hidden_dims]
    :params length: [batch]
    :Returns : [batch, hidden_dims]
    """
    shape = tf.shape(parameters)
    idx = tf.range(shape[0])
    idx = tf.stack([idx, length - 1], axis=1)
    return tf.gather_nd(parameters, idx)


class MatchLSTMCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 premise,
                 premise_mask,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(MatchLSTMCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        if not state_is_tuple:
            logger.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logger.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                         "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._premise = premise
        self._premise_mask = premise_mask
        self._premise_length = tf.shape(premise)[1]

        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        self._kernel = self.add_variable(
            "kernel",
            shape=[input_depth + h_depth*2, 4 * self._num_units])
        self._bias = self.add_variable(
            "bias",
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._ws = self.add_variable("ws",
            shape=[self._num_units, self._num_units])
        self._wt = self.add_variable("wt",
            shape=[self._num_units, self._num_units])
        self._wm = self.add_variable("wm",
            shape=[self._num_units, self._num_units])
        self._we = self.add_variable("we",
            shape=[self._num_units, 1])

        self._ds = tf.tensordot(self._premise, self._ws, axes=[[-1], [0]])

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        dt = tf.tensordot(inputs, self._wt, axes=[[-1], [0]])
        dt = tf.tile(tf.expand_dims(dt, 1), [1, self._premise_length, 1])
        dm = tf.tensordot(h, self._wm, axes=[[-1], [0]])
        dm = tf.tile(tf.expand_dims(dm, 1), [1, self._premise_length, 1])
        e_kj = tf.tensordot(tf.nn.tanh(dt+self._ds+dm), self._we,  axes=[[-1], [0]])
        e_kj = e_kj + (1. - self._premise_mask) * tf.float32.min
        alpha = tf.nn.softmax(e_kj, axis=1)
        a_k = tf.reduce_sum(tf.multiply(alpha, self._premise), axis=1)

        m_k = tf.concat([a_k, inputs], axis=1)

        gate_inputs = math_ops.matmul(
            array_ops.concat([m_k, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "forget_bias": self._forget_bias,
            "state_is_tuple": self._state_is_tuple,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(MatchLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))