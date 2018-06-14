# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import sys
from vocab_utils import Vocab
import namespace_utils

import tensorflow as tf
import SentenceMatchTrainer
from SentenceMatchModelGraph import SentenceMatchModelGraph
from SentenceMatchDataStream import SentenceMatchDataStream

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()
    
    # load the configuration file
    print('Loading configurations.')
    options = namespace_utils.load_namespace(args.model_prefix + ".config.json")

    if args.word_vec_path is None: args.word_vec_path = options.word_vec_path


    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(args.word_vec_path, fileformat='txt3')
    label_vocab = Vocab(args.model_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    if options.with_char:
        char_vocab = Vocab(args.model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    
    print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchDataStream(args.in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                            label_vocab=label_vocab,
                                            isShuffle=False, isLoop=True, isSort=True, options=options)
    print('Number of instances in devDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    best_path = args.model_prefix + ".best.model"
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  is_training=False, options=options)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        print("Restoring model from " + best_path)
        saver.restore(sess, best_path)
        print("DONE!")
        result = SentenceMatchTrainer.predict(sess, valid_graph, testDataStream, outpath=args.out_path,
                                              label_vocab=label_vocab)
        print(result)


