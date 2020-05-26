import io
import sys
from operator import add
import csv
from BytePairEncoder import *
import nltk

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

def get_num_words(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    num_words = int(fin.readline().split()[0])
    return num_words

def get_dimension(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    dimension = int(fin.readline().split()[1])
    return dimension

def read_dataset(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = fin.read()
    words = set(nltk.word_tokenize(data))
    return words

def build_word_piece_model(vectors):
    vector_corpus = list(vectors.keys())
    n_iters = 100
    encoder = BytePairEncoder(n_iters)
    encoder.train(vector_corpus)
    return encoder

def get_sub_words(encoder, word):
    return encoder.tokenize(word).split(' ')

def get_vectors(vectors, words, dimension):
    encoder = build_word_piece_model(vectors)
    word_vectors = {}
    for word in words:
        if word in vectors:
            vector = vectors[word]
            word_vectors[word] = vector
        else:
            avg_vector = [0] * dimension
            sub_words = get_sub_words(encoder, word)
            for sub_word in sub_words:
                if sub_word in vectors:
                    vector = vectors[sub_word]
                    avg_vector = list(map(add, avg_vector, vector))
            avg_vector = [round(value/len(sub_words),4) for value in avg_vector]
            word_vectors[word] = avg_vector
    return word_vectors

def write_csv(filename, word_vectors):
    with open(filename, mode='w', encoding='utf-8', newline='\n') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ')
        for word, vector in word_vectors.items():
            row = [word] + vector
            csv_writer.writerow(row)

if '__main__':
    if len(sys.argv) != 3:
        print('Example usages: python Task1.py <vector_filepath> <dataset_filepath>')
        sys.exit(2)
    vec_filepath = sys.argv[1]
    dataset_path = sys.argv[2]
    num_words = get_num_words(vec_filepath)
    dimension = get_dimension(vec_filepath)
    print('Demensions: Total Words = %d; Vector Size = %d' % (num_words, dimension))
    print('Reading Word Vectors')
    vectors = load_vectors(vec_filepath)
    print('Reading Dataset')
    words = read_dataset(dataset_path)
    print('Getting Word Vectors')
    vectors = get_vectors(vectors, words, dimension)
    print('Writing CSV')
    write_csv('word_vectors.csv', vectors)