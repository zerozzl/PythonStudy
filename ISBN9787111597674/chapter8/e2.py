# -*- coding: utf-8 -*-
import os
from os.path import isfile, join
import matplotlib.pyplot as plt


pos_file_path = '/sdb/traindatas/ISBN9787111597674/chapter8/aclImdb/train/pos/'
neg_file_path = '/sdb/traindatas/ISBN9787111597674/chapter8/aclImdb/train/neg/'
pos_files = [pos_file_path + f for f in os.listdir(pos_file_path) if isfile(join(pos_file_path, f))]
neg_files = [neg_file_path + f for f in os.listdir(neg_file_path) if isfile(join(neg_file_path, f))]

num_words = []
for pf in pos_files:
    with open(pf, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('complete processing pos file')

for nf in neg_files:
    with open(nf, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('complete processing neg file')

num_files = len(num_words)
print('total file: %s' % num_files)
print('words number: %s' % sum(num_words))
print('average word length of file: %s' % (sum(num_words) / len(num_words)))


plt.hist(num_words, 50, facecolor='g')
plt.xlabel('text length')
plt.ylabel('count')
plt.axis([0, 1200, 0, 8000])
plt.show()
