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
                 dropout_rate: float = 0.1, initializer_range: float = 0.02, init_checkpoint=None, model_name: str = 'bert_classifier'):
        super().__init__(embedding_mapping=embedding_mapping, optimizer=optimizer, init_checkpoint=init_checkpoint, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate
        self._initializer_range = initializer_range

    def forward(self, features, labels, mode, params):
        features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
        with tf.variable_scope(self._model_name):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            premise_tokens_ids = features.get('premise/tokens', None)
            #hypothesis_tokens_ids = features.get('hypothesis/tokens', None)

            if premise_tokens_ids is None:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens "
                                     "or elmo_characters.")
            # if hypothesis_tokens_ids is None:
            #     raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens "
            #                          "or elmo_characters.")

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

            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            output_dict = self._make_output(output_layer, params)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels_embedding = features_embedding['label/labels']
                labels = features['label/labels']

                loss = self._make_loss(labels=labels_embedding, logits=output_dict['logits'], params=params)
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=labels, predictions=output_dict['predictions']['predictions'])
                metrics['precision'] = tf.metrics.precision(labels=labels, predictions=output_dict['predictions']['predictions'])
                metrics['recall'] = tf.metrics.recall(labels=labels, predictions=output_dict['predictions']['predictions'])
               # metrics['auc'] = tf.metrics.auc(labels=labels, predictions=predictions)

                output_dict['metrics'] = metrics
                # output_dict['debugs'] = [hypothesis_tokens, premise_tokens, hypothesis_bi, premise_bi,
                #                          v_1_ave, v_2_ave, h_mlp, logits]
            return output_dict
