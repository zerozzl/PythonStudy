# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train(data_path, model_path):
    wiki_news = open(data_path, 'r')
    model = Word2Vec(LineSentence(wiki_news), sg=0, size=192, window=5, min_count=5, workers=32)
    model.save(model_path)


if __name__ == '__main__':
    data_path = '/sdb/traindatas/ISBN9787111597674/chapter7/recuce_zhiwiki.txt'
    model_path = '/sdb/traindatas/ISBN9787111597674/chapter7/model/w2v/zhiwiki_news.word2vec'
    train(data_path, model_path)
