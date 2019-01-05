# -*- coding: utf-8 -*-
from random import randint
import numpy as np
import tensorflow as tf
import os
import datetime


num_dimensions = 300
max_seq_num = 300
batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 200000
lr = 0.001
ids = np.load('/sdb/traindatas/ISBN9787111597674/chapter8/idsMatrix.npy')


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2) == 0:
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1: num]
    return arr, labels


def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1: num]
    return arr, labels


wordVectors = np.load('/sdb/traindatas/ISBN9787111597674/chapter8/wordVectors.npy')
print('loaded word vectors')

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batch_size, num_labels])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])
data = tf.Variable(tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    if os.path.exists('/sdb/traindatas/ISBN9787111597674/chapter8/models') and os.path.exists('/sdb/traindatas/ISBN9787111597674/chapter8/models/checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint('/sdb/traindatas/ISBN9787111597674/chapter8/models'))
    else:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = '/sdb/traindatas/ISBN9787111597674/chapter8/tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for step in range(iterations):

        next_batch, next_batch_labels = get_train_batch()
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})

        if (step % 10) == 0:
            print('traning %s/%s' % (step, iterations))

        if (step % 50) == 0:
            summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
            writer.add_summary(summary, step)

        if ((step % 10000) == 0) and (step != 0):
            save_path = saver.save(sess, '/sdb/traindatas/ISBN9787111597674/chapter8/models/pretrained_lstm.ckpt', global_step=step)
            print('saved to %s' % save_path)

    writer.close()




