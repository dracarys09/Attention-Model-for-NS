import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.callbacks import ModelCheckpoint
from AttentionLayer import AttentionWeightedAverage

DATA_PATH='/home/development/abhijeetd/Attention-Model-for-NS/data/data.tsv'                                                        
GLOVE_DIR_PATH='/home/development/abhijeetd/Attention-Model-for-NS/glove.twitter.27B'                                               
CHECKPOINT_FILE_PATH='/home/development/abhijeetd/Attention-Model-for-NS/HAN_models/100d/weights.best.hdf5'                                  
SAVE_DATA=False                                                                                                                 
MAX_SENT_LENGTH=100                                                                                                                 
MAX_SENTS=15                                                                                                                        
MAX_NB_WORDS=20000                                                                                                                  
EMBEDDING_DIM=100                                                                                                                   
VALIDATION_SPLIT=0.2                                                                                                                
NUM_EPOCHS=12                                                                                                                       
BATCH_SIZE=128 

EVAL=False

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string.decode("utf-8"))
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv(DATA_PATH, sep='\t')
print(data_train.shape)

from nltk import tokenize

reviews = []
labels = []
texts = []

y_val = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")
    text = clean_str(text.get_text().encode('ascii','ignore'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
    
    if idx >= 9181 and idx <= 11023:
        y_val.append(data_train.sentiment[idx])
    elif idx >= 52631 and idx <= 60947:
        y_val.append(data_train.sentiment[idx])
    else:
        labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
print(data.shape)

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1



x_val_neg = data[52631:,:,:]
data = np.delete(data, (np.s_[52631:60948]), axis=0)

x_val_pos = data[9181:11024,:,:]
data = np.delete(data, (np.s_[9181:11024]), axis=0)

x_val = np.concatenate((x_val_pos, x_val_neg), axis=0)

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
y_val = to_categorical(np.asarray(y_val))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

#x_train = data
#y_train = labels

if SAVE_DATA==True:
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    
    print('..........................Saving Data.......................')

    print(type(y_train))
    print(type(x_train.shape))
    print(type(y_val))
    print(type(x_val.shape))

    with open('x_train.pkl', 'wb') as f:
        cPickle.dump(x_train, f)
    with open('x_val.pkl', 'wb') as f:
        cPickle.dump(x_val, f)
    with open('y_train.pkl', 'wb') as f:
        cPickle.dump(y_train, f)
    with open('y_val.pkl', 'wb') as f:
        cPickle.dump(y_val, f)

else:
    with open('/home/development/abhijeetd/Attention-Model-for-NS/x_val.pkl', 'rb') as f:
        x_val = cPickle.load(f)
    with open('/home/development/abhijeetd/Attention-Model-for-NS/y_val.pkl', 'rb') as f:
        y_val = cPickle.load(f)
    with open('/home/development/abhijeetd/Attention-Model-for-NS/x_train.pkl', 'rb') as f:
        x_train = cPickle.load(f)
    with open('/home/development/abhijeetd/Attention-Model-for-NS/y_train.pkl', 'rb') as f:
        y_train = cPickle.load(f)



print('Number of positive and negative numerical sarcastic tweets in traing and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR_PATH, 'glove.twitter.27B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(200, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttentionWeightedAverage()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(200, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttentionWeightedAverage()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)


if EVAL==True:

    MODEL_PATH='/home/development/abhijeetd/Attention-Model-for-NS/HAN_models/200d/weights.best.hd5'

    model.load_weights(MODEL_PATH)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print("............................Evaluating the model.............................")

    with open('/home/development/abhijeetd/Attention-Model-for-NS/x_val.pkl', 'rb') as f:
        x_test = cPickle.load(f)
    with open('/home/development/abhijeetd/Attention-Model-for-NS/y_val.pkl', 'rb') as f:
        y_test = cPickle.load(f)

    score = model.evaluate(x_test, y_test)
    print(score)

    print('...........................Making Predictions....................')
    prediction = model.predict(x_test)
    print ("Predictions")
    print (len(prediction))

    f = open('Predictions_200d_Glove_Embeddings.txt',"w")
    for pred in prediction:
        f.write(str(pred))
        f.write("\n")
        f.close()
    exit(0)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

checkpoint = ModelCheckpoint(CHECKPOINT_FILE_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


print("model fitting - Attention Network")
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)


