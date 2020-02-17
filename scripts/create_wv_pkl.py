import pickle
import time
from gensim.models import KeyedVectors,
import numpy as np

file = '../BioWordVec_PubMed_MIMICIII_d200.vec.bin'

# Save wv model from binary
model = KeyedVectors.load_word2vec_format(file, binary=True)
model.save('wv.model')

# Load wv model
# model = KeyedVectors.load("wv.model")

# Create index2word
index2word_tensor = model.index2word
index2word_tensor.pop()
index2word_tensor.append('<pad>')
word2index_lookup = {word: index for index, word in enumerate(index2word_tensor)}
with open("index2word.pkl", 'wb') as handle:
    pickle.dump(word2index_lookup,handle)

# Load word vectors (the numpy file will auto be created when you save the model)
# vectors = np.load('wv.model.vectors.npy')