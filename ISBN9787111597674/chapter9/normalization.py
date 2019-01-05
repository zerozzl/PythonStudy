# -*- coding: utf-8 -*-
import re
import string
import jieba

with open('/sdb/traindatas/ISBN9787111597674/chapter9/stop_words.utf8', encoding='utf-8') as f:
    stopword_list = f.readlines()
    stopword_list = [w.strip() for w in stopword_list]


def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    normalize_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalize_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalize_corpus.append(text)
    return normalize_corpus
