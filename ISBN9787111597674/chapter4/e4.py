#-*- coding: utf-8 -*-
import os
import CRFPP


def f1(path):
    with open(path) as f:
        all_tag = 0
        loc_tag = 0
        pred_loc_tag = 0
        correct_tag = 0
        correct_loc_tag = 0
        states = ['B', 'M', 'E', 'S']
        for line in f:
            line = line.strip()
            if line == '':
                continue
            _, r, p = line.split()
            all_tag += 1

            if r == p:
                correct_tag += 1
                if r in states:
                    correct_loc_tag += 1
            if r in states:
                loc_tag += 1
            if p in states:
                pred_loc_tag += 1

        loc_P = 1.0 * correct_loc_tag / pred_loc_tag
        loc_R = 1.0 * correct_loc_tag / loc_tag
        print('loc_P:{0}, loc_R:{1}, loc_F1:{2}'.format(loc_P, loc_R, (2 * loc_P * loc_R) / (loc_P + loc_R)))


def load_model(path):
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    else:
        return None


def location_ner(model_path, text):
    tagger = load_model(model_path)
    for c in text:
        tagger.add(c)

    result = []
    tagger.parse()
    word = ''
    for i in range(0, tagger.size()):
        for j in range(0, tagger.xsize()):
            ch = tagger.x(i, j)
            tag = tagger.y2(i)
            if tag == 'B':
                word = ch
            elif tag == 'M':
                word += ch
            elif tag == 'E':
                word += ch
                result.append(word)
            elif tag == 'S':
                word = ch
                result.append(word)
    return result


if __name__ == '__main__':
    f1('/sdb/traindatas/ISBN9787111597674/chapter4/test.rst')
    model_path = '/sdb/traindatas/ISBN9787111597674/chapter4/model'

    text = '我中午要去北京饭店，下午去中山公园，晚上回亚运村'
    print(text, location_ner(model_path, text), sep='==> ')

    text = '我去回龙观，不去南锣鼓巷'
    print(text, location_ner(model_path, text), sep='==> ')

    text = '打的去北京南站地铁站'
    print(text, location_ner(model_path, text), sep='==> ')
