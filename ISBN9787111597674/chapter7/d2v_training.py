# -*- coding: utf-8 -*-
import gensim.models as g
from gensim.corpora import WikiCorpus
import logging
from ISBN9787111597674.chapter7.langconv import *
import jieba


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield g.doc2vec.LabeledSentence(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])


def train():
    docvec_size = 192
    zhwiki_name = '/sdb/traindatas/ISBN9787111597674/chapter7/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

    model = g.Doc2Vec(documents, dm=0, dbow_words=1, size=docvec_size, window=8, min_count=19, iter=5, workers=32)
    model.save('/sdb/traindatas/ISBN9787111597674/chapter7/model/d2v/zhiwiki_news.doc2vec')


if __name__ == '__main__':
    train()
