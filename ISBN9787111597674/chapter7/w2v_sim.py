# -*- coding: utf-8 -*-
import gensim
import numpy as np
from jieba import analyse


def keyword_extract(filepath):
    tfidf = analyse.extract_tags
    keywords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for l in lines:
            keywords.extend(tfidf(l))
    return keywords


def word2vec(model, wordvec_size, keywords):
    word_vec_all = np.zeros(wordvec_size)
    for word in keywords:
        if model.__contains__(word):
            word_vec_all += model[word]
    return word_vec_all


def simlarityCalu(vec1, vec2):
    vec1Mod = np.sqrt(vec1.dot(vec1))
    vec2Mod = np.sqrt(vec2.dot(vec2))
    if vec1Mod != 0 and vec2Mod != 0:
        simlarity = (vec1.dot(vec2)) / (vec1Mod * vec2Mod)
    else:
        simlarity = 0
    return simlarity


if __name__ == '__main__':
    wordvec_size = 192
    model = gensim.models.Word2Vec.load('/sdb/traindatas/ISBN9787111597674/chapter7/model/w2v/zhiwiki_news.word2vec')
    p1 = '/sdb/traindatas/ISBN9787111597674/chapter7/p1.txt'
    p2 = '/sdb/traindatas/ISBN9787111597674/chapter7/p2.txt'
    kw1 = keyword_extract(p1)
    kw2 = keyword_extract(p2)
    vec1 = word2vec(model, wordvec_size, kw1)
    vec2 = word2vec(model, wordvec_size, kw2)
    print(simlarityCalu(vec1, vec2))
