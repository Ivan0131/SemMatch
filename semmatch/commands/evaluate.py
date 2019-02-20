import os
import tensorflow as tf
from semmatch.commands.command import Command
from semmatch.utils import register
from semmatch.config.parameters import Parameters
from semmatch.utils.exception import ConfigureError
from semmatch.data.data_readers.data_reader import DataSplit, DataReader
from tensorflow.python.saved_model import loader_impl
import csv
from openpyxl import Workbook
import numpy as np
from tensorflow.python.estimator import model_fn
from tensorflow.python.saved_model import signature_constants
from semmatch.models import Model
from semmatch.config.run_config import RunConfig
from semmatch.config.hparams import HParams
from semmatch.utils.logger import logger
from tensorflow.python.estimator import estimator
from semmatch.utils.saved_model import get_meta_graph_def_for_mode, get_signature_def_for_mode, \
    check_same_dtype_and_shape

@register.register_subclass('command', 'eval')
class Evaluate(Command):
    name = 'eval'
    description = 'eval a saved model on a specified dataset'
    parser = None

    def __init__(self, data_reader=None, eval_input_fn=None, vocab=None, export_dir=None, output_file=None):
        if data_reader is not None and eval_input_fn is None:
            self._eval_input_fn = data_reader.make_estimator_input_fn(DataSplit.EVAL, force_repeat=False)
            vocab = data_reader.get_vocab()
        else:
            self._eval_input_fn = eval_input_fn
        self.saved_model_loader = loader_impl.SavedModelLoader(export_dir)

        dataset = self._eval_input_fn()
        iterator = dataset.make_initializable_iterator()
        dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        mode = DataSplit.PREDICT
        signature_def = get_signature_def_for_mode(self.saved_model_loader, mode)

        input_map = self.generate_input_map(signature_def, next_element)
        output_tensor_names = [
            value.name for value in signature_def.outputs.values()]
        tags = model_fn.EXPORT_TAG_MAP[mode]
        saver, output_tensors = self.saved_model_loader.load_graph(
            tf.get_default_graph(), tags, input_map=input_map, return_elements=output_tensor_names)
        output_map = dict(zip(output_tensor_names, output_tensors))
        outputs = {key: output_map[value.name]
                   for (key, value) in signature_def.outputs.items()}

        # #############
        # prediction = list(outputs.values())[0]
        # example_name = input_names[0]

        #predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

        #####xlsx wirte######
        wb = Workbook(write_only=True)
        ws = wb.create_sheet('examples')
        ws.append(['question', 'answer', 'true_label', 'predict'])

        total_num = 0
        accuracy = 0
        confusion_matrix = [[0, 0] for i in range(2)]

        with tf.Session() as sess:
            self.saved_model_loader.restore_variables(sess, saver)
            self.saved_model_loader.run_init_ops(sess, tags)
            sess.run(iterator.initializer)
            while True:
                try:
                    outputs['inputs'] = next_element
                    output_vals = sess.run(outputs)

                    data_batch = output_vals['inputs']
                    premise_tokens_val, hypothesis_tokens_val, true_label_val = \
                        data_batch['premise/tokens'], data_batch['hypothesis/tokens'], data_batch['label/labels']
                    predictions = output_vals['output']
                    num_batch = predictions.shape[0]
                    #######################
                    total_num += num_batch

                    for i in range(predictions.shape[0]):
                        predict = predictions[i]
                        label = true_label_val[i]
                        if predict == label:
                            accuracy += 1
                        confusion_matrix[label][predict] += 1
                        ################
                    for i in range(num_batch):
                        premise_str = vocab.convert_indexes_to_tokens(premise_tokens_val[i], 'tokens')
                        premise_str = " ".join(premise_str)
                        hypothesis_str = vocab.convert_indexes_to_tokens(hypothesis_tokens_val[i], 'tokens')
                        hypothesis_str = " ".join(hypothesis_str)
                        true_label = true_label_val[i]
                        predict = predictions[i]
                        ws.append([premise_str, hypothesis_str, str(true_label), str(predict)])
                    print("process %s/%s correct/total instances with accuracy %s." % (accuracy, total_num, accuracy/float(total_num)))
                except tf.errors.OutOfRangeError as e:
                    logger.warning(e)
                    accuracy = accuracy/total_num
                    precise = confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])
                    recall = confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1])
                    f1score = (precise+recall)/2
                    print("metrics:")
                    confmx = """ label \ predict | 0 | 1 
                       ------------------------------------
                       0      |     {}     |    {}        
                       1   |     {}     |    {}      """.format(
                        confusion_matrix[0][0], confusion_matrix[0][1],
                        confusion_matrix[1][0], confusion_matrix[1][1])
                    print(confmx)
                    print("accuracy: %s, precise: %s, recall: %s, f1-score: %s" % (accuracy, precise, recall, f1score))
                    ws = wb.create_sheet(title='metrics')
                    ws.append(['label \ predict', "0", "1"])
                    ws.append(['0', str(confusion_matrix[0][0]), str(confusion_matrix[0][1])])
                    ws.append(['1', str(confusion_matrix[1][0]), str(confusion_matrix[1][1])])
                    ws.append([])
                    ws.append([])
                    ws.append(['accuracy', 'precise', 'recall', 'f1-score'])
                    ws.append([str(accuracy), str(precise), str(recall), str(f1score)])
                    if output_file:
                        if not output_file.endswith(".xlsx"):
                            output_file += '.xlsx'
                        wb.save(output_file)
                    break

    def generate_input_map(self, signature_def, features, labels=None):
        inputs = signature_def.inputs
        input_map = {}
        for (key, tensor_info) in inputs.items():
            input_name = tensor_info.name
            if ':' in input_name:
                input_name = input_name[:input_name.find(':')]
            control_dependency_name = '^' + input_name
            if key in features:
                features[key] = tf.cast(features[key], tf.int64)
                check_same_dtype_and_shape(features[key], tensor_info, key)
                input_map[input_name] = input_map[control_dependency_name] = features[key]
            elif labels is not None and key in labels:
                check_same_dtype_and_shape(labels[key], tensor_info, key)
                input_map[input_name] = input_map[control_dependency_name] = labels[key]
            else:
                raise ValueError(
                    'Key \"%s\" not found in features or labels passed in to the model '
                    'function. All required keys: %s' % (key, inputs.keys()))
        return input_map

    @classmethod
    def init_from_params(cls, params):
        # ####data reader##############
        data_reader = DataReader.init_from_params(params.pop('data'))
        export_dir = params.pop('export_dir')
        output_file = params.pop('output_file')
        #####embedding mapping##########
        params.assert_empty(cls.__name__)
        cls(data_reader=data_reader, export_dir=export_dir, output_file=output_file)

    @classmethod
    def add_subparser(cls, parser):
        cls.parser = parser.add_parser(name=cls.name, description=cls.description, help='evaluate a model')
        cls.parser.add_argument('--config_path', type=str,
                                help='the config path where store the params.')
        cls.parser.set_defaults(func=cls.init_train_from_args)
        return cls.parser

    @classmethod
    def init_train_from_args(cls, args):
        config_path = args.config_path
        cls.init_train_from_config_file(config_path)

    @classmethod
    def init_train_from_config_file(cls, config_path):
        params = Parameters.init_from_file(config_path)
        return cls.init_from_params(params)


##################
#######################
# data_batch = sess.run(next_element)
# premise_tokens_val, hypothesis_tokens_val, true_label_val = \
#     data_batch['premise/tokens'], data_batch['hypothesis/tokens'], data_batch['label/labels']
# examples = []
# num_batch = true_label_val.shape[0]
# for i in range(num_batch):
#     features = {}
#     features['premise/tokens'] = tf.train.Feature(int64_list=tf.train.Int64List(value=premise_tokens_val[i]))
#     features['hypothesis/tokens'] = tf.train.Feature(int64_list=tf.train.Int64List(value=hypothesis_tokens_val[i]))
#     features['label/labels'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[true_label_val[i]]))
#
#     example = tf.train.Example(
#         features=tf.train.Features(
#             feature=features
#         )
#     )
#     examples.append(example.SerializeToString())
#
# predictions = predict_fn({'examples': examples})['output']
#########

