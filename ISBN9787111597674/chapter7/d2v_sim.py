# -*- coding: utf-8 -*-
import numpy as np
import codecs
import jieba
import gensim.models as g


def doc2vec(filepath, model, start_alpha, infer_epoch):
    doc = [w for x in codecs.open(filepath, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all


def simlarityCalu(vec1, vec2):
    vec1Mod = np.sqrt(vec1.dot(vec1))
    vec2Mod = np.sqrt(vec2.dot(vec2))
    if vec1Mod != 0 and vec2Mod != 0:
        simlarity = (vec1.dot(vec2)) / (vec1Mod * vec2Mod)
    else:
        simlarity = 0
    return simlarity


if __name__ == '__main__':
    model_path = '/sdb/traindatas/ISBN9787111597674/chapter7/model/d2v/zhiwiki_news.doc2vec'
    p1 = '/sdb/traindatas/ISBN9787111597674/chapter7/p1.txt'
    p2 = '/sdb/traindatas/ISBN9787111597674/chapter7/p2.txt'
    start_alpha = 0.01
    infer_epoch = 1000
    docvec_size = 192
    model = g.Doc2Vec.load(model_path)

    vec1 = doc2vec(p1, model, start_alpha, infer_epoch)
    vec2 = doc2vec(p2, model, start_alpha, infer_epoch)
    print(simlarityCalu(vec1, vec2))
