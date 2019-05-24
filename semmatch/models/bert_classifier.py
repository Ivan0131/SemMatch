import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.modules.embeddings.encoders.bert import create_initializer
from semmatch import nn


#BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
#https://arxiv.org/abs/1810.04805
@register.register_subclass('model', 'bert_classifier')
class BertClassifier(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(),
                 dropout_rate: float = 0.1, initializer_range: float = 0.02, model_name: str = 'bert_classifier'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate
        self._initializer_range = initializer_range

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            premise_tokens_ids = features.get('premise/tokens', None)
            hypothesis_tokens_ids = features.get('hypothesis/tokens', None)

            if premise_tokens_ids is None:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens "
                                     "or elmo_characters.")
            if hypothesis_tokens_ids is None:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens "
                                     "or elmo_characters.")

            premise_tokens = features_embedding.get('premise/tokens', None)

            hidden_size = premise_tokens.shape[-1].value

            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(premise_tokens[:, 0:1, :], axis=1)
                output_layer = tf.layers.dense(
                    first_token_tensor,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(self._initializer_range))

            output_weights = tf.get_variable(
                "output_weights", [self._num_classes, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [self._num_classes], initializer=tf.zeros_initializer())

            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            predictions = tf.argmax(logits, -1)

            output_dict = {'logits': logits, 'predictions': predictions}

            probs = tf.nn.softmax(logits, -1)
            output_score = tf.estimator.export.PredictOutput(probs)
            export_outputs = {"output_score": output_score}
            output_dict['export_outputs'] = export_outputs

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
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
                #                          v_1_ave, v_2_ave, h_mlp, logits]
            return output_dict
