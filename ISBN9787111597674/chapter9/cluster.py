# -*- coding: utf-8 -*-
import pandas as pd
import jieba
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans


def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text = " ".join(jieba.lcut(text))
        normalized_corpus.append(text)

    return normalized_corpus


def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters, max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def get_cluster_data(clustering_obj, book_data, feature_names, num_clusters, topn_features=10):
    cluster_details = {}
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books
    return cluster_details


def print_cluster_data(cluster_data):
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster %s details: ' % cluster_num)
        print('-' * 20)
        print('Key features: %s' % cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)


book_data = pd.read_csv('/sdb/traindatas/ISBN9787111597674/chapter9/data.csv')
print(book_data.head())
book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()

print('book titiles: %s' % book_titles[0])
print('book content: %s' % book_content[0][:10])

# normalize corpus
norm_book_content = normalize_corpus(book_content)
vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                  feature_type='tfidf',
                                                  min_df=0.2, max_df=0.90,
                                                  ngram_range=(1, 2))

print(feature_matrix.shape)
feature_names = vectorizer.get_feature_names()
print(feature_names[:10])

num_clusters = 10
km_obj, clusters = k_means(feature_matrix=feature_matrix, num_clusters=num_clusters)
book_data['Cluster'] = clusters

c = Counter(clusters)
print(c.items())

cluster_data = get_cluster_data(clustering_obj=km_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=num_clusters,
                                topn_features=5)
print_cluster_data(cluster_data)
