import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import sys
import tflearn
import io
from operator import add
from BytePairEncoder import *
import math

stop_words = sorted(stopwords.words('english'))

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    total_words, dimension = map(int, fin.readline().split())
    vectors = {}
    i = 1
    for line in fin:
        sys.stdout.write('\rProcessing %d/%d Word' % (i, total_words))
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vector = list(map(float, tokens[1:]))
        vectors[word] = vector
        i += 1
    print('')
    return vectors

def get_dimension(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    dimension = int(fin.readline().split()[1])
    return dimension

def build_word_piece_model(vectors):
    vector_corpus = list(vectors.keys())
    n_iters = 100
    encoder = BytePairEncoder(n_iters)
    encoder.train(vector_corpus)
    return encoder

def get_sub_words(encoder, word):
    return encoder.tokenize(word).split(' ')

def load_data(filename):
    df = pd.read_excel(filename, sheet_name='Sheet1')
    return df

def pre_process(df):
    classes = []
    documents = []
    # loop through each utterance in the intents
    for index, row in df.iterrows():
        utterance = row['Utterance']
        label = row['Label']
        # tokenize each word in the utterance
        tokens = nltk.word_tokenize(utterance)
        # lower each token and remove stop words
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
        # add to documents in the corpus
        documents.append((tokens, label))
        # add to classes list
        if label not in classes:
            classes.append(label)
    return documents, classes

def create_train_data(vectors, dimension, max_doc_len, documents, classes):
    encoder = build_word_piece_model(vectors)
    training = []
    output = []
    # training set, bag of words for each utterance
    for doc in documents:
        # create an empty array for output
        output = [0] * len(classes)
        # initialize our bag of words
        input = []
        # list of tokenized words for the utterance
        utterance_words = doc[0]
        for i in range(0, max_doc_len):
            if i < len(utterance_words):
                utterance_word = utterance_words[i]
                if utterance_word in vectors:
                    vector = vectors[utterance_word]
                    input.extend(vector)
                else:
                    avg_vector = [0] * dimension
                    sub_words = get_sub_words(encoder, utterance_word)
                    for sub_word in sub_words:
                        if sub_word in vectors:
                            vector = vectors[sub_word]
                            avg_vector = list(map(add, avg_vector, vector))
                    avg_vector = [round(value / len(sub_words), 4) for value in avg_vector]
                    input.extend(avg_vector)
            else:
                input.extend([0]*dimension)
        # output is a '0' for each label and '1' for current label
        output[classes.index(doc[1])] = 1
        training.append([input, output])
    # turn features into np.array
    training = np.array(training)
    # create input and output lists
    train_input = list(training[:, 0])
    train_output = list(training[:, 1])
    return train_input, train_output

def train(train_input, train_output):
    # build neural network
    net = tflearn.input_data(shape=[None, len(train_input[0])])
    net = tflearn.fully_connected(net, 64)
    net = tflearn.fully_connected(net, 64)
    net = tflearn.fully_connected(net, len(train_output[0]), activation='softmax')
    net = tflearn.regression(net)
    # define model and setup tensorboard
    model = tflearn.DNN(net)
    # start training (apply gradient descent algorithm)
    model.fit(train_input, train_output, n_epoch=100, batch_size=8, show_metric=True)
    model.save('model.tflearn')

def get_max_doc_len(documents):
    max_doc_len = float('-inf')
    for doc in documents:
        # list of tokenized words for the utterance
        utterance_words = doc[0]
        if len(utterance_words) > max_doc_len:
            max_doc_len = len(utterance_words)
    return max_doc_len


if '__main__':
    if len(sys.argv) != 3:
        print('Example usages: python Task1.py <vector_filepath> <dataset_filepath>')
        sys.exit(2)
    vec_filepath = sys.argv[1]
    dataset_path = sys.argv[2]
    dimension = get_dimension(vec_filepath)
    print('')
    print('Reading Word Vectors')
    vectors = load_vectors(vec_filepath)
    print('Reading Dataset')
    df = load_data(dataset_path)
    print('Preprocessing Dataset')
    documents, classes = pre_process(df)
    print('Gathering Maximum Document Length')
    max_doc_len = get_max_doc_len(documents)
    print('Preparing Training Data')
    train_input, train_output = create_train_data(vectors, dimension, max_doc_len, documents, classes)
    print('Training Model')
    train(train_input, train_output)
