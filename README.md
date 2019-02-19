## INSTALL
```
git clone https://git.sogou-inc.com/hangzhou_research/SemMatch.git
virtualenv sogou_semmatch
source sogou_semmatch/bin/activate
cd SemMatch
pip3 install -r requirements.txt
python3 setup.py install #只支持python3.6+
```
### 测试
```
cd examples
cd text_matching_bilstm
python3 script.py 
#python3 -m semmatch.run train --config_path='./config.yaml'
```

模型接口暂时按照以下方式，之后再调整。
## 从命令行运行
```
python3 -m semmatch.run train --config_path='./config.yaml'
```
配置文件示例如下：

```
data:
  name: "quora" # or quora_data_reader
  data_path: './data'
  train_filename: 'train.tsv'
  valid_filename: 'dev.tsv'
  batch_size: 32

model:
  name: 'text_matching_bilstm'
  num_classes: 2
  hidden_dim: 300
  keep_prob: 0.5
  embedding_mapping:
    name: 'base'
    encoders:
      tokens:
        name: 'embedding'
        embedding_dim: 300
        trainable: true
        pretrained_file: "../data/glove.840B.300d.txt"
      labels:
        name: 'one_hot'
        n_values: 2
  optimizer:
    name: 'adam'
    learning_rate: 0.00001
run_config:
  model_dir: './outputs'
hparams:
  train_steps: 100000
  eval_steps: 100
```

## 作为类库运行
```python
import semmatch


semmatch.list_available() #get the available modules
semmatch.list_available('data') #get the available data reader

data_reader = semmatch.get_by_name('data', 'quora')(data_path='./data') #get quoar data reader
optimizer = semmatch.get_by_name('optimizer', 'adam')(learning_rate=0.001) #get adam optimizer
vocab = data_reader.get_vocab()
encoders = {"tokens": semmatch.get_by_name('encoder', 'embedding')(embedding_dim=300, trainable=True,
                                                                   pretrained_file="../data/glove.840B.300d.txt",
                                                                   vocab=vocab, vocab_namespace='tokens'),
            'labels': semmatch.get_by_name('encoder', 'one_hot')(n_values=2)} #create encoders for embedding mapping
embedding_mapping = semmatch.get_by_name('embedding_mapping', 'base')(encoders=encoders) #create embedding mapping
model = semmatch.get_by_name('model', 'text_matching_bilstm')(embedding_mapping=embedding_mapping,
                                       optimizer=optimizer, num_classes=2) #create model
train = semmatch.get_by_name('command', 'train')(data_reader=data_reader, model=model) #train model
```

## 添加自定义数据
数据的读取模块参考了AllenNLP中field，instance，token_indexer的概念。因此添加数据模块的方式也跟AllenNLP相似。下面以Quora这个数据集为例，说明如何添加数据集。

```python
import os
from typing import Dict, List
from semmatch.data.data_readers import data_reader
from semmatch.data.fields import Field, TextField, LabelField
from semmatch.data.tokenizers import WordTokenizer, Tokenizer
from semmatch.data import Instance
from semmatch.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from semmatch.utils import register


@register.register_subclass('data', 'quora')
class QuoraDataReader(data_reader.DataReader):

    def __init__(self, data_name: str = "quora", data_path: str = None, batch_size: int = 32, train_filename="train.tsv",
                 valid_filename="dev.tsv", test_filename=None, max_length: int = 48, tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: List[Tokenizer] = None):
        super().__init__(data_name=data_name, data_path=data_path, batch_size=batch_size, train_filename=train_filename,
                         valid_filename=valid_filename, test_filename=test_filename)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or [SingleIdTokenIndexer(namespace='tokens')]
        self._max_length = max_length

    def _read(self, mode: str):
        filename = self.get_filename_by_mode(mode)
        if filename:
            filename = os.path.join(qqp_dir, filename)
            for example in self.example_generator(filename):
                yield example
        else:
            return None

    def _process(self, example):
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(example['premise'])
        tokenized_hypothesis = self._tokenizer.tokenize(example['hypothesis'])
        fields["premise"] = TextField(tokenized_premise, self._token_indexers, max_length=self._max_length)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers, max_length=self._max_length)
        if 'label' in example:
            fields['label'] = LabelField(example['label'])
        return Instance(fields)

    def example_generator(self, filename):
        skipped = 0
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            if len(split_line) < 6:
                skipped += 1
                tf.logging.info("Skipping %d" % skipped)
                continue
            s1, s2, l = split_line[3:]
            inputs = [[s1, s2], [s2, s1]]
            for inp in inputs:
                example = {
                    "premise": inp[0],
                    "hypothesis": inp[1],
                    "label": int(l)
                }
                yield self._process(example)
```    
上述代码中，我们新建了一个`QuoraDataReader`。通过`@register.register_subclass`往`data`中注册了`quora`。`__init__`中我们采用了type hints是为了从配置文件中入去参数新建类的时候确定参数的类型。`QuoraDataReader`实现了基类中的两个函数`_read`和`_process`。`_read`实现了每个数据集具体的数据读写细节，`_process`负责将读取得到的数据进行数据预处理（例如分词等），然后转化为`Instance`。`Instance`是一个`Dict[str, Field]`，每个`Field`需要指定`token_indexer`(例如添加基于词、字符等特征)。

接下来，我们的框架会自动根据`Instance`建立相应的vocabulary，并将数据保存到tfrecord文件中，并自动从这些文件中读取数据。tfrecord文件保存和读取时候的features的key的值为`field_name/vocab_namespace`。`field_name`在`_process`函数中指定，例如‘premise’，‘hypothesis’和‘label’。`vocab_namespace`在_token_indexers中指定例如在`__init__`新建`SingleIdTokenIndexer`时，我们指定了‘tokens’，没有指定的话采用默认值。不同的`vocab_namespace`会采用不同的字典。


## 添加自定义模型
下面以text matching的bilstm模型作为例子。

```python
import tensorflow as tf
from semmatch.models import Model
from semmatch.utils import register
from semmatch.utils.exception import ConfigureError
from semmatch.modules.embeddings import EmbeddingMapping
from semmatch.modules.optimizers import Optimizer, AdamOptimizer
from semmatch.nn import layers


@register.register_subclass('model', 'text_matching_bilstm')
class BiLSTM(Model):
    def __init__(self, embedding_mapping: EmbeddingMapping, num_classes, optimizer: Optimizer=AdamOptimizer(), hidden_dim: int = 300, keep_prob:float = 0.5,
                 model_name: str = 'bilstm'):
        super().__init__(optimizer=optimizer, model_name=model_name)
        self._embedding_mapping = embedding_mapping
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._dropout_prob = keep_prob
        self._reuse = False

    def forward(self, features, labels, mode, params):
        with tf.variable_scope(self._model_name) as scope:
            if self._reuse:
                scope.reuse_variables()
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            if 'premise/tokens' not in features:
                raise ConfigureError("The input features should contain premise with vocabulary namespace tokens.")
            if 'hypothesis/tokens' not in features:
                raise ConfigureError("The input features should contain hypothesis with vocabulary namespace tokens.")
            prem_seq_lengths, prem_mask = layers.length(features['premise/tokens'])
            hyp_seq_lengths, hyp_mask = layers.length(features['hypothesis/tokens'])
            features_embedding = self._embedding_mapping.forward(features, labels, mode, params)
            premise_tokens = features_embedding['premise/tokens']
            hypothesis_tokens = features_embedding['hypothesis/tokens']

            premise_outs, c1 = layers.biLSTM(premise_tokens, dim=self._hidden_dim, seq_len=prem_seq_lengths, name='premise')
            hypothesis_outs, c2 = layers.biLSTM(hypothesis_tokens, dim=self._hidden_dim, seq_len=hyp_seq_lengths, name='hypothesis')

            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            ### Mean pooling
            premise_sum = tf.reduce_sum(premise_bi, 1)
            premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

            hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
            hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

            ### Mou et al. concat layer ###
            diff = tf.subtract(premise_ave, hypothesis_ave)
            mul = tf.multiply(premise_ave, hypothesis_ave)
            h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

            # MLP layer
            h_mlp = tf.contrib.layers.fully_connected(h, self._hidden_dim, scope='fc1')
            # Dropout applied to classifier
            h_drop = tf.layers.dropout(h_mlp, self._dropout_prob, training=is_training)
            # Get prediction
            logits = tf.contrib.layers.fully_connected(h_drop, self._num_classes, activation_fn=None, scope='logits')
            predictions = tf.arg_max(logits, -1)
            output_dict = {'logits': logits, 'predictions': predictions}

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if 'label/labels' not in features:
                    raise ConfigureError("The input features should contain label with vocabulary namespace "
                                         "labels int %s dataset."%mode)
                labels = features_embedding['label/labels']
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
                output_dict['loss'] = loss
                metrics = dict()
                metrics['accuracy'] = tf.metrics.accuracy(labels=tf.argmax(labels, -1), predictions=predictions)
                output_dict['metrics'] = metrics
            self._reuse = True
            return output_dict
```
`@register.register_subclass`在`model`中注册了`text_matching_bilstm `。初始化函数中主要包含两个模块`EmbeddingMapping`和`Optimizer`。主要具体实现一个函数`forward`，返回一个output\_dict。output\_dict为一个字典，包含`tf.estimator.EstimatorSpec`所需要的参数，例如：loss、metrics、predictions等。`features_embedding = self._embedding_mapping.forward(features, labels, mode, params)`是通过embedding将输入数据进行编码，例如将词根据词向量进行编码，对标签根据one hot进行编码。

## to do list
* Tokenizer扩充，目前只是正则匹配的英文分词，可以添加已经分好词的输入数据（由空格分开）的分词处理，nltk，jieba分词，中英文分词的处理
* token_indexer扩充，例如字符，NER，POS tag
* Tokenizer里面包含了文本的一些数据处理，目前都比较简单，除了分词之外，没有做其它处理，添加其它文本的预处理
* data\_reader 只添加了Quora，需要添加其它的常见数据集，以及根据列表数据（csv格式、json等）的通用的data_reader
* vocabulary从counter建立，目前写得过去简单，可以根据词频，embedding 文件对单词进行一定的筛选
* embedding 目前只添加了单层的embedding，预训练文件读取txt文件内容。以后需要添加其它格式的支持以及其它的embedding，例如elmo、bert等
* model只是简单的实现了bilstm，需要实现其它常见的模型
* commands只实现了train，还需要实现其它操作，例如predict、embedding pretrain等。train部分还需要进一步补充
* optimizer 目前只添加了adam optimizer，以后需要添加更多的optimizer
* 用户调用的API的调整
* 常用的函数，模块的整理，例如 attention等



