import os
import tensorflow as tf
from semmatch.commands.command import Command
from semmatch.utils import register
from semmatch.config.parameters import Parameters
from semmatch.data.data_readers.data_reader import DataSplit, DataReader
from tensorflow.python.saved_model import loader_impl
from openpyxl import Workbook
import numpy as np
from tensorflow.python.estimator import model_fn
from semmatch.utils.saved_model import get_meta_graph_def_for_mode, get_signature_def_for_mode, \
    check_same_dtype_and_shape
from semmatch.utils.logger import logger
from sklearn import metrics


@register.register_subclass('command', 'pred')
class Predict(Command):
    name = 'pred'
    description = 'predict a saved model on a specified dataset'
    parser = None

    def __init__(self, data_reader=None, pred_input_fn=None, vocab=None, export_dir=None, output_file=None):
        if data_reader is not None and pred_input_fn is None:
            self._pred_input_fn = data_reader.make_estimator_input_fn(DataSplit.PREDICT, force_repeat=False)
            vocab = data_reader.get_vocab()
        else:
            self._pred_input_fn = pred_input_fn

        dataset = self._pred_input_fn()
        iterator = dataset.make_initializable_iterator()
        dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.saved_model_loader = loader_impl.SavedModelLoader(export_dir)
        mode = DataSplit.PREDICT
        signature_def = get_signature_def_for_mode(self.saved_model_loader, mode)

        input_map = self.generate_input_map(signature_def, next_element)
        output_tensor_names = [
            value.name for value in signature_def.outputs.values()]
        try:
            tags = model_fn.EXPORT_TAG_MAP[mode]
        except AttributeError as e:
            tags = ['serve']
        saver, output_tensors = self.saved_model_loader.load_graph(
            tf.get_default_graph(), tags, input_map=input_map, return_elements=output_tensor_names)
        output_map = dict(zip(output_tensor_names, output_tensors))
        outputs = {key: output_map[value.name]
                   for (key, value) in signature_def.outputs.items()}


        #####xlsx wirte######
        # wb = Workbook(write_only=True)
        # ws = wb.create_sheet('examples')
        # ws.append(['index', 'question', 'answer', 'predict', 'score'])
        csv_file = open(output_file, 'w', encoding="utf-8")
        csv_file.write("\t".join(["index", "prediction"]) + "\n")
        #csv_file.write("\t".join(["index", "premise", "hypothesis", "prediction", "prob"])+"\n")
        total_num = 0
        #accuracy = 0
        #confusion_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]

        with tf.Session() as sess:
            self.saved_model_loader.restore_variables(sess, saver)
            self.saved_model_loader.run_init_ops(sess, tags)
            sess.run(iterator.initializer)
            while True:
                try:
                    outputs['inputs'] = next_element
                    output_vals = sess.run(outputs)

                    data_batch = output_vals['inputs']
                    if "index/index" in data_batch:
                        index_val, premise_tokens_val, hypothesis_tokens_val = \
                            data_batch['index/index'], data_batch['premise/tokens'], data_batch['hypothesis/tokens']
                        indexs = [str(index, 'utf-8') for index in index_val]
                    else:
                        premise_tokens_val, hypothesis_tokens_val = \
                            data_batch['premise/tokens'], data_batch['hypothesis/tokens']
                        index_val = list(range(total_num, total_num+hypothesis_tokens_val.shape[0]))
                        indexs = [str(index) for index in index_val]
                    probs = output_vals['output']
                    num_batch = probs.shape[0]
                    #######################
                    predictions = probs #np.argmax(probs, axis=1)
                    #predictions = (probs > 0.5).astype(np.int32)
                    total_num += num_batch
                    logger.info("processing %s/%s" % (num_batch, total_num))

                    # for i in range(probs.shape[0]):
                    #     predictions = (probs > 0.5).astype(np.int32)
                    #     predict = predictions[i]
                    #     label = true_label_val[i]
                    #     if predict == label:
                    #         accuracy += 1
                    #     confusion_matrix[label][predict] += 1
                        ################
                    for i in range(num_batch):
                        premise_str = vocab.convert_indexes_to_tokens(premise_tokens_val[i], 'tokens')
                        premise_str = " ".join(premise_str)
                        hypothesis_str = vocab.convert_indexes_to_tokens(hypothesis_tokens_val[i], 'tokens')
                        hypothesis_str = " ".join(hypothesis_str)
                        predict = predictions[i]
                        prob = probs[i]
                        index = indexs[i]
                        #csv_str = "\t".join([index, premise_str, hypothesis_str, str(predict), str(prob)])+"\n"
                        csv_str = "\t".join([index, str(predict)]) + "\n"
                        csv_file.write(csv_str)
                        #ws.append([index, premise_str, hypothesis_str, str(predict), str(prob)])
                    #print("process %s/%s correct/total instances with accuracy %s." % (accuracy, total_num, accuracy/float(total_num)))
                except tf.errors.OutOfRangeError as e:
                    # if output_file:
                    #     if not output_file.endswith(".xlsx"):
                    #         output_file += '.xlsx'
                    #     wb.save(output_file)
                    csv_file.close()
                    break

    def generate_input_map(self, signature_def, features, labels=None):
        features_mapping = {"input_query": "premise/tokens", "input_title": "hypothesis/tokens"}
        inputs = signature_def.inputs
        input_map = {}
        for (key, tensor_info) in inputs.items():
            input_name = tensor_info.name
            if ':' in input_name:
                input_name = input_name[:input_name.find(':')]
            control_dependency_name = '^' + input_name
            if features_mapping is not None and key in features_mapping:
                feature_key = features_mapping[key]
            else:
                feature_key = key
            if feature_key in features:
                check_same_dtype_and_shape(features[feature_key], tensor_info, key)
                input_map[input_name] = input_map[control_dependency_name] = features[feature_key]
            elif labels is not None and feature_key in labels:
                check_same_dtype_and_shape(labels[feature_key], tensor_info, key)
                input_map[input_name] = input_map[control_dependency_name] = labels[feature_key]
            else:
                logger.warning(
                    'Key \"%s\" not found in features or labels passed in to the model '
                    'function. All required keys: %s' % (feature_key, inputs.keys()))
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

