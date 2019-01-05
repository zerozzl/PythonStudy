#-*- coding: utf-8 -*-
import os
import jieba
from nltk.parse import stanford


if __name__ == '__main__':
    string = '他骑自行车去了菜市场'
    seg_list = jieba.cut(string, cut_all=False, HMM=True)
    seg_str = ''
    for w in seg_list:
        seg_str += w + ' '

    print(seg_str)
    parser_path = '/sdb/traindatas/ISBN9787111597674/chapter6/stanford-parser-full-2018-10-17/stanford-parser.jar'
    model_path = '/sdb/traindatas/ISBN9787111597674/chapter6/stanford-chinese-corenlp-2018-10-05-models.jar'

    pcfg_path = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

    parser = stanford.StanfordParser(path_to_jar=parser_path,
                                     path_to_models_jar=model_path,
                                     model_path=pcfg_path)

    sentence = parser.raw_parse(seg_str)
    for line in sentence:
        print(line.leaves())
        line.draw()
