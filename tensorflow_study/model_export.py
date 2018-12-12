import os
import tensorflow as tf


if __name__ == '__main__':
    export_path = '/sdb'
    meta_file = '/sdb/ckpt_model_d/InsightFace_iter_best_710000.ckpt.meta'
    model_file = '/sdb/ckpt_model_d/InsightFace_iter_best_710000.ckpt.data-00000-of-00001'
    print('Exporting model...')
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    model_file = os.path.join(export_path, 'embedding.pb')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, model_file)

            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names=['embeddings'])
            with tf.gfile.FastGFile(model_file, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('save model to  %s' % model_file)