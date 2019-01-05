# -*- coding: utf-8 -*-
import numpy as np
import re
import gensim
import jieba
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from ISBN9787111597674.chapter9.normalization import normalize_corpus
from ISBN9787111597674.chapter9.feature_extractors import bow_extractor, tfidf_extractor


def get_date():
    with open('/sdb/traindatas/ISBN9787111597674/chapter9/ham_data.txt', encoding='utf-8') as ham_f,open('/sdb/traindatas/ISBN9787111597674/chapter9/spam_data.txt', encoding='utf-8') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()

        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()

        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels


def get_metrics(true_labels, predicted_labels):
    print('accuracy: %s' % (np.round(metrics.accuracy_score(true_labels, predicted_labels), 2)))
    print('precision: %s' % (np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2)))
    print('recall: %s' % (np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2)))
    print('f1 score: %s' % (np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2)))


def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions


def main():
    corpus, labels = get_date()
    print('total doc: %s' % len(labels))

    corpus, labels = remove_empty_docs(corpus, labels)
    print('sample: %s' % corpus[10])
    print('sample label: %s' % labels[10])

    label_name_map = ['spam', 'ham']
    print('actual type: %s, %s' % (label_name_map[int(labels[10])] ,label_name_map[int(labels[5900])]))

    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus, labels, test_data_proportion=0.3)

    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)

    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]
    tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]

    model = gensim.models.Word2Vec(tokenized_train, size=500, window=100, min_count=30, sample=1e-3)

    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', n_iter=100)
    lr = LogisticRegression()

    print('native bayes base on bow')
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    print('lr base on bow')
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    print('svm base on bow')
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    print('native bayes base on tfidf')
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=tfidf_train_features,
                                                       train_labels=train_labels,
                                                       test_features=tfidf_test_features,
                                                       test_labels=test_labels)

    print('lr base on tfidf')
    lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)

    print('svm base on tfidf')
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                        train_features=tfidf_train_features,
                                                        train_labels=train_labels,
                                                        test_features=tfidf_test_features,
                                                        test_labels=test_labels)

    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 0 and predicted_label == 0:
            print('mail type: %s' % label_name_map[int(label)])
            print('predict mail type: %s' % label_name_map[int(predicted_label)])
            print('text:-')
            print(re.sub('\n', '', document))

            num += 1
            if num == 4:
                break

    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 1 and predicted_label == 0:
            print('mail type: %s' % label_name_map[int(label)])
            print('predict mail type: %s' % label_name_map[int(predicted_label)])
            print('text:-')
            print(re.sub('\n', '', document))

            num += 1
            if num == 4:
                break



if __name__ == '__main__':
    main()
