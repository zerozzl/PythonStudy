# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus
import jieba
from ISBN9787111597674.chapter7.langconv import *


def process(data_path, output_path):
    space = ' '
    i = 0
    l = []
    f = open(output_path, 'w')
    wiki = WikiCorpus(data_path, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        for temp_sentence in text:
            temp_sentence = Converter('zh-hans').convert(temp_sentence)
            seg_list = list(jieba.cut(temp_sentence))
            for temp_term in seg_list:
                l.append(temp_term)
        f.write(space.join(l) + '\n')
        l = []
        i = i + 1

        if (i % 200) == 0:
            print('Saved %s articles' % i)
    f.close()


if __name__ == '__main__':
    data_path = '/sdb/traindatas/ISBN9787111597674/chapter7/zhwiki-latest-pages-articles.xml.bz2'
    output_path = '/sdb/traindatas/ISBN9787111597674/chapter7/recuce_zhiwiki.txt'
    process(data_path, output_path)
