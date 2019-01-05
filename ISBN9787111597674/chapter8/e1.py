# -*- coding: utf-8 -*-
import numpy as np


wordsList = np.load('/sdb/traindatas/ISBN9787111597674/chapter8/wordsList.npy')
print('loaded word list')

wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]

wordVectors = np.load('/sdb/traindatas/ISBN9787111597674/chapter8/wordVectors.npy')
print('loaded word vectors')

print(len(wordsList))
print(wordVectors.shape)

home_index = wordsList.index('home')
print(wordVectors[home_index])
