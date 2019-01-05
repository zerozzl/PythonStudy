# -*- coding: utf-8 -*-
import gensim


def test(model_path):
    model = gensim.models.Word2Vec.load(model_path)
    print(model.similarity('西红柿', '番茄')) # 0.63
    print(model.similarity('西红柿', '香蕉')) # 0.44

    word = '中国'
    if word in model.wv.index2word:
        print(model.most_similar(word))


if __name__ == '__main__':
    model_path = '/sdb/traindatas/ISBN9787111597674/chapter7/model/w2v/zhiwiki_news.word2vec'
    test(model_path)
