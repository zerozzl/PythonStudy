# -*- coding: utf-8 -*-
import jieba.posseg as psg

sent = '中文分词是文本处理不可或缺的一部！'
seg_list = psg.cut(sent)
print(' '.join(['{0}/{1}'.format(w, t) for w, t in seg_list]))
