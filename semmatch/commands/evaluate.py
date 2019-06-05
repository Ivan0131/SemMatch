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
    check_same_dtype_and_shape, generate_input_map
import simplejson as json
from sklearn import metrics


@register.register_subclass('command', 'eval')
class Evaluate(Command):
    name = 'eval'
    description = 'evaluate a saved model on a specified dataset'
    parser = None

    def __init__(self, data_reader=None, eval_input_fn=None, num_classes=None, vocab=None, export_dir=None, output_file=None):
        if data_reader is not None and eval_input_fn is None:
            self._eval_input_fn = data_reader.make_estimator_input_fn(DataSplit.EVAL, force_repeat=False)
            vocab = data_reader.get_vocab()
        else:
            self._eval_input_fn = eval_input_fn

        dataset = self._eval_input_fn()
        iterator = dataset.make_initializable_iterator()
        dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.saved_model_loader = loader_impl.SavedModelLoader(export_dir)

        mode = DataSplit.PREDICT
        signature_def = get_signature_def_for_mode(self.saved_model_loader, mode)

        input_map = generate_input_map(signature_def, next_element)
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
        # #############
        # prediction = list(outputs.values())[0]
        # example_name = input_names[0]
        ##############
        graph = tf.get_default_graph()
        outputs['query_embedding'] = graph.get_tensor_by_name("esim/concat:0")
        outputs['title_embedding'] = graph.get_tensor_by_name("esim/concat_1:0")
        outputs['query_lstm_1'] = graph.get_tensor_by_name("esim/concat_2:0")
        outputs['title_lstm_1'] = graph.get_tensor_by_name("esim/concat_3:0")
        outputs['query_attention'] = graph.get_tensor_by_name('esim/bi_uni_attention/MatMul:0')
        outputs['title_attention'] = graph.get_tensor_by_name('esim/bi_uni_attention/MatMul_1:0')
        outputs['query_lstm_2'] = graph.get_tensor_by_name("esim/concat_6:0")
        outputs['title_lstm_2'] = graph.get_tensor_by_name("esim/concat_7:0")
        ###############
        #predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

        #####xlsx wirte######
        wb = Workbook(write_only=True)
        ws = wb.create_sheet('examples')
        ws.append(['question', 'answer', 'true_label', 'predict', 'score'])


        y_true = []
        y_pred = []
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
                    if 'premise/tokens' in data_batch.keys() and 'hypothesis/tokens' in data_batch.keys():
                        premise_tokens_val, hypothesis_tokens_val, true_label_val = \
                            data_batch['premise/tokens'], data_batch['hypothesis/tokens'], data_batch['label/labels']
                    else:
                        true_label_val = data_batch['label/labels']
                        premise_tokens_val = [[] for i in range(len(true_label_val))]
                        hypothesis_tokens_val = [[] for i in range(len(true_label_val))]
                    #probs = output_vals['output_score']
                    probs = output_vals['output']
                    num_batch = probs.shape[0]
                    total_num += num_batch
                    print("processing %s/%s"%(num_batch, total_num))
                    #######################
                    #print(probs)
                    predictions = np.argmax(probs, axis=1)
                    #predictions = (probs > 0.5).astype(np.int32)
                    #print(predictions)
                    y_true.append(true_label_val)
                    y_pred.append(predictions)
                    #print(predictions)
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
                        true_label = true_label_val[i]
                        predict = predictions[i]
                        prob = probs[i]
                        #ws.append([premise_str, hypothesis_str, str(true_label), str(predict), str(prob)])
                        ws.append([premise_str, hypothesis_str, str(true_label), str(predict), str(prob),
                                   json.dumps(output_vals['query_embedding'][i].tolist()), json.dumps(output_vals['title_embedding'][i].tolist()),
                                   json.dumps(output_vals['query_lstm_1'][i].tolist()), json.dumps(output_vals['title_lstm_1'][i].tolist()),
                                   json.dumps(output_vals['query_attention'][i].tolist()), json.dumps(output_vals['title_attention'][i].tolist()),
                                   json.dumps(output_vals['query_lstm_2'][i].tolist()), json.dumps(output_vals['title_lstm_2'][i].tolist())])
                    #print("process %s/%s correct/total instances with accuracy %s." % (accuracy, total_num, accuracy/float(total_num)))
                except tf.errors.OutOfRangeError as e:
                    #logger.warning(e)
                    y_true = np.concatenate(y_true, axis=0)
                    y_pred = np.concatenate(y_pred, axis=0)
                    avg_param = 'micro'
                    if num_classes == 2:
                        avg_param = 'binary'
                    accuracy = metrics.accuracy_score(y_true, y_pred)#accuracy/total_num
                    precise, recall, f1score, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=avg_param)
                    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
                    #confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])
                    #recall = confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1])
                    #f1score = (precise+recall)/2
                    print("metrics:")
                    confmx_str = "label \ predict "
                    for i in range(num_classes):
                        confmx_str += "| %s | "%i
                    confmx_str += "\n"
                    for i in range(num_classes):
                        confmx_str += "| %s | "%i
                        for j in range(num_classes):
                            confmx_str += "| %s | "%confusion_matrix[i][j]
                        confmx_str += "\n"

                    print(confmx_str)
                    print("accuracy: %s, precise: %s, recall: %s, f1-score: %s" % (accuracy, precise, recall, f1score))
                    ws = wb.create_sheet(title='metrics')
                    legend = ["label \ predict "]
                    for i in range(num_classes):
                        legend.append(str(i))
                    ws.append(legend)
                    for i in range(num_classes):
                        row = [str(i)]
                        for j in range(num_classes):
                            row.append(str(confusion_matrix[i][j]))
                        ws.append(row)
                    ws.append([])
                    ws.append([])
                    ws.append(['accuracy', 'precise', 'recall', 'f1-score'])
                    ws.append([str(accuracy), str(precise), str(recall), str(f1score)])
                    if output_file:
                        if not output_file.endswith(".xlsx"):
                            output_file += '.xlsx'
                        wb.save(output_file)
                    break

    @classmethod
    def init_from_params(cls, params):
        # ####data reader##############
        data_reader = DataReader.init_from_params(params.pop('data'))
        export_dir = params.pop('export_dir')
        output_file = params.pop('output_file')
        num_classes = params.pop_int('num_classes')
        #####embedding mapping##########
        params.assert_empty(cls.__name__)
        cls(data_reader=data_reader, export_dir=export_dir, output_file=output_file, num_classes=num_classes)

    @classmethod
    def add_subparser(cls, parser):
        cls.parser = parser.add_parser(name=cls.name, description=cls.description, help='evaluate a model')
        cls.parser.add_argument('--config_path', type=str,
                                help='the config path where store the params.')
        cls.parser.set_defaults(func=cls.init_from_args)
        return cls.parser

    @classmethod
    def init_from_args(cls, args):
        config_path = args.config_path
        cls.init_from_config_file(config_path)

    @classmethod
    def init_from_config_file(cls, config_path):
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

