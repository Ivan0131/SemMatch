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




