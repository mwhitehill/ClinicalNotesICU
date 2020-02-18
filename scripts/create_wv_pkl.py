import pickle
import time
from gensim.models import KeyedVectors
import numpy as np
from itertools import islice

file = '../BioWordVec_PubMed_MIMICIII_d200.vec.bin'

# Save wv model from binary
def load_binary():

    model = KeyedVectors.load_word2vec_format(file, binary=True)
    model.save('wv.model')

# Load wv model
def load_model():

    model = KeyedVectors.load("wv.model")
    return(model)

# Create index2word
def create_index2word(model):

    index2word_tensor = model.index2word
    index2word_tensor.pop()
    index2word_tensor.append('<pad>')
    word2index_lookup = {word: index for index, word in enumerate(index2word_tensor)}
    with open("index2word.pkl", 'wb') as handle:
        pickle.dump(word2index_lookup, handle)
    return(index2word_tensor)

# Load word vectors (the numpy file will auto be created when you save the model)
def load_vectors():
    vectors = np.load('wv.model.vectors.npy')
    return(vectors)

def take(n, iterable):
    return list(islice(iterable, n))

def save_index2word():
    with open("index2word.pkl", 'rb') as handle:
        word2index_lookup = pickle.load(handle)

    word2index_lookup_small = {'<pad>': 0}

    for k,v in word2index_lookup.items():
        word2index_lookup_small[k] = v+1
        if len(word2index_lookup_small.keys())==1000:
            break

    with open("index2word_small.pkl", 'wb') as handle:
        pickle.dump(word2index_lookup_small, handle)


def save_smaller_vectors():
    vectors = load_vectors()
    vectors_small = vectors[:1000, :]
    np.save('vectors_small.npy', vectors_small)

if __name__ == '__main__':
    # save_smaller_vectors(vectors)
    save_index2word()