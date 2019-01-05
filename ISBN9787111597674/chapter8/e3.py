# -*- coding: utf-8 -*-
import re
import numpy as np
import os
from os.path import isfile, join


strip_special_chars = re.compile('[^A-Za-z0-9 ]+')
max_seq_num = 300

wordsList = np.load('/sdb/traindatas/ISBN9787111597674/chapter8/wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
print('loaded word list')

pos_file_path = '/sdb/traindatas/ISBN9787111597674/chapter8/aclImdb/train/pos/'
neg_file_path = '/sdb/traindatas/ISBN9787111597674/chapter8/aclImdb/train/neg/'
pos_files = [pos_file_path + f for f in os.listdir(pos_file_path) if isfile(join(pos_file_path, f))]
neg_files = [neg_file_path + f for f in os.listdir(neg_file_path) if isfile(join(neg_file_path, f))]
num_files = len(pos_files) + len(neg_files)

file_count = 0
idx = np.zeros((num_files, max_seq_num), dtype='int32')


def cleanSentences(string):
    string = string.lower().replace('<br />', ' ')
    return re.sub(strip_special_chars, '', string.lower())


for pf in pos_files:
    with open(pf, 'r', encoding='utf-8') as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                idx[file_count][indexCounter] = wordsList.index(word)
            except ValueError:
                idx[file_count][indexCounter] = 399999
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
        file_count = file_count + 1


for nf in neg_files:
    with open(nf, 'r', encoding='utf-8') as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                idx[file_count][indexCounter] = wordsList.index(word)
            except ValueError:
                idx[file_count][indexCounter] = 399999
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
        file_count = file_count + 1

np.save('/sdb/traindatas/ISBN9787111597674/chapter8/idsMatrix', idx)
