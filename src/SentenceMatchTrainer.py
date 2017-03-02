# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf

from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils

FLAGS = None

def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt')
    for line in infile:
        line = line.decode('utf-8').strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = re.split("\\s+",items[1].lower())
        sentence2 = re.split("\\s+",items[2].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def evaluate(dataStream, valid_graph, sess, outpath=None, label_vocab=None, mode='prediction',char_vocab=None, POS_vocab=None, NER_vocab=None):
    if outpath is not None: outfile = open(outpath, 'wt')
    total_tags = 0.0
    correct_tags = 0.0
    dataStream.reset()
    for batch_index in xrange(dataStream.get_num_batch()):
        cur_dev_batch = dataStream.get_batch(batch_index)
        (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch) = cur_dev_batch
        feed_dict = {
                    valid_graph.get_truth(): label_id_batch, 
                    valid_graph.get_question_lengths(): sent1_length_batch, 
                    valid_graph.get_passage_lengths(): sent2_length_batch, 
                    valid_graph.get_in_question_words(): word_idx_1_batch, 
                    valid_graph.get_in_passage_words(): word_idx_2_batch, 
#                     valid_graph.get_question_char_lengths(): sent1_char_length_batch, 
#                     valid_graph.get_passage_char_lengths(): sent2_char_length_batch, 
#                     valid_graph.get_in_question_chars(): char_matrix_idx_1_batch, 
#                     valid_graph.get_in_passage_chars(): char_matrix_idx_2_batch, 
                }

        if char_vocab is not None:
            feed_dict[valid_graph.get_question_char_lengths()] = sent1_char_length_batch
            feed_dict[valid_graph.get_passage_char_lengths()] = sent2_char_length_batch
            feed_dict[valid_graph.get_in_question_chars()] = char_matrix_idx_1_batch
            feed_dict[valid_graph.get_in_passage_chars()] = char_matrix_idx_2_batch

        if POS_vocab is not None:
            feed_dict[valid_graph.get_in_question_poss()] = POS_idx_1_batch
            feed_dict[valid_graph.get_in_passage_poss()] = POS_idx_2_batch

        if NER_vocab is not None:
            feed_dict[valid_graph.get_in_question_ners()] = NER_idx_1_batch
            feed_dict[valid_graph.get_in_passage_ners()] = NER_idx_2_batch


        total_tags += len(label_batch)
        correct_tags += sess.run(valid_graph.get_eval_correct(), feed_dict=feed_dict)
        if outpath is not None:
            if mode =='prediction':
                predictions = sess.run(valid_graph.get_predictions(), feed_dict=feed_dict)
                for i in xrange(len(label_batch)):
                    outline = label_batch[i] + "\t" + label_vocab.getWord(predictions[i]) + "\t" + sent1_batch[i] + "\t" + sent2_batch[i] + "\n"
                    outfile.write(outline.encode('utf-8'))
            else:
                probs = sess.run(valid_graph.get_prob(), feed_dict=feed_dict)
                for i in xrange(len(label_batch)):
                    outfile.write(label_batch[i] + "\t" + output_probs(probs[i], label_vocab) + "\n")

    if outpath is not None: outfile.close()

    accuracy = correct_tags / total_tags * 100
    return accuracy

def output_probs(probs, label_vocab):
    out_string = ""
    for i in xrange(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()

def main(_):
    print('Configurations:')
    print(FLAGS)

    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    path_prefix = log_dir + "/SentenceMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    best_path = path_prefix + '.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    POS_path = path_prefix + ".POS_vocab"
    NER_path = path_prefix + ".NER_vocab"
    has_pre_trained_model = False
    POS_vocab = None
    NER_vocab = None
    if os.path.exists(best_path):
        has_pre_trained_model = True
        label_vocab = Vocab(label_path, fileformat='txt2')
        char_vocab = Vocab(char_path, fileformat='txt2')
        if FLAGS.with_POS: POS_vocab = Vocab(POS_path, fileformat='txt2')
        if FLAGS.with_NER: NER_vocab = Vocab(NER_path, fileformat='txt2')
    else:
        print('Collect words, chars and labels ...')
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path, with_POS=FLAGS.with_POS, with_NER=FLAGS.with_NER)
        print('Number of words: {}'.format(len(all_words)))
        print('Number of labels: {}'.format(len(all_labels)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels,dim=2)
        label_vocab.dump_to_txt2(label_path)

        print('Number of chars: {}'.format(len(all_chars)))
        char_vocab = Vocab(fileformat='voc', voc=all_chars,dim=FLAGS.char_emb_dim)
        char_vocab.dump_to_txt2(char_path)
        
        if FLAGS.with_POS:
            print('Number of POSs: {}'.format(len(all_POSs)))
            POS_vocab = Vocab(fileformat='voc', voc=all_POSs,dim=FLAGS.POS_dim)
            POS_vocab.dump_to_txt2(POS_path)
        if FLAGS.with_NER:
            print('Number of NERs: {}'.format(len(all_NERs)))
            NER_vocab = Vocab(fileformat='voc', voc=all_NERs,dim=FLAGS.NER_dim)
            NER_vocab.dump_to_txt2(NER_path)
            

    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    print('tag_vocab shape is {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    print('Build SentenceMatchDataStream ... ')
    trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, 
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=True, isLoop=True, isSort=True, 
                                              max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length)
                                    
    devDataStream = SentenceMatchDataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True, 
                                              max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length)

    testDataStream = SentenceMatchDataStream(test_path, word_vocab=word_vocab, char_vocab=char_vocab, 
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True, 
                                              max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length)

    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    
    sys.stdout.flush()
    if FLAGS.wo_char: char_vocab = None

    best_accuracy = 0.0
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
#         with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                 dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                 lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                 aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=True, MP_dim=FLAGS.MP_dim, 
                 context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                 fix_word_vec=FLAGS.fix_word_vec,with_filter_layer=FLAGS.with_filter_layer, with_highway=FLAGS.with_highway,
                 word_level_MP_dim=FLAGS.word_level_MP_dim,
                 with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                 highway_layer_num=FLAGS.highway_layer_num,with_lex_decomposition=FLAGS.with_lex_decomposition, 
                 lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                 with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                 with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match), 
                 with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match))
            tf.summary.scalar("Training Loss", train_graph.get_loss()) # Add a scalar summary for the snapshot loss.
        
#         with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                 dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                 lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                 aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim, 
                 context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                 fix_word_vec=FLAGS.fix_word_vec,with_filter_layer=FLAGS.with_filter_layer, with_highway=FLAGS.with_highway,
                 word_level_MP_dim=FLAGS.word_level_MP_dim,
                 with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                 highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition, 
                 lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                 with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                 with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match), 
                 with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match))

                
        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
#             if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
         
        sess = tf.Session()
        sess.run(initializer)
        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            # read data
            cur_batch = trainDataStream.nextBatch()
            (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch) = cur_batch
            feed_dict = {
                         train_graph.get_truth(): label_id_batch, 
                         train_graph.get_question_lengths(): sent1_length_batch, 
                         train_graph.get_passage_lengths(): sent2_length_batch, 
                         train_graph.get_in_question_words(): word_idx_1_batch, 
                         train_graph.get_in_passage_words(): word_idx_2_batch, 
#                          train_graph.get_question_char_lengths(): sent1_char_length_batch, 
#                          train_graph.get_passage_char_lengths(): sent2_char_length_batch, 
#                          train_graph.get_in_question_chars(): char_matrix_idx_1_batch, 
#                          train_graph.get_in_passage_chars(): char_matrix_idx_2_batch, 
                         }
            if char_vocab is not None:
                feed_dict[train_graph.get_question_char_lengths()] = sent1_char_length_batch
                feed_dict[train_graph.get_passage_char_lengths()] = sent2_char_length_batch
                feed_dict[train_graph.get_in_question_chars()] = char_matrix_idx_1_batch
                feed_dict[train_graph.get_in_passage_chars()] = char_matrix_idx_2_batch

            if POS_vocab is not None:
                feed_dict[train_graph.get_in_question_poss()] = POS_idx_1_batch
                feed_dict[train_graph.get_in_passage_poss()] = POS_idx_2_batch

            if NER_vocab is not None:
                feed_dict[train_graph.get_in_question_ners()] = NER_idx_1_batch
                feed_dict[train_graph.get_in_passage_ners()] = NER_idx_2_batch

            _, loss_value = sess.run([train_graph.get_train_op(), train_graph.get_loss()], feed_dict=feed_dict)
            total_loss += loss_value
            
            if step % 100==0: 
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                print()
                # Print status to stdout.
                duration = time.time() - start_time
                start_time = time.time()
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                total_loss = 0.0

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                accuracy = evaluate(devDataStream, valid_graph, sess,char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab)
                print("Current accuracy is %.2f" % accuracy)
                if accuracy>best_accuracy:
                    best_accuracy = accuracy
                    saver.save(sess, best_path)

    print("Best accuracy on dev set is %.2f" % best_accuracy)
    # decoding
    print('Decoding on the test set:')
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                 dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                 lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                 aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim, 
                 context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                 fix_word_vec=FLAGS.fix_word_vec,with_filter_layer=FLAGS.with_filter_layer, with_highway=FLAGS.with_highway,
                 word_level_MP_dim=FLAGS.word_level_MP_dim,
                 with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                 highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition, 
                 lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                 with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                 with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match), 
                 with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match))
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
                
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)

        accuracy = evaluate(testDataStream, valid_graph, sess,char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab)
        print("Accuracy for test set is %.2f" % accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--MP_dim', type=int, default=10, help='Number of perspectives for matching vectors.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', required=True, help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--with_highway', default=False, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_filter_layer', default=False, help='Utilize filter layer.', action='store_true')
    parser.add_argument('--word_level_MP_dim', type=int, default=-1, help='Number of perspectives for word-level matching.')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_lex_decomposition', default=False, help='Utilize lexical decomposition features.', action='store_true')
    parser.add_argument('--lex_decompsition_dim', type=int, default=-1, help='Number of dimension for lexical decomposition features.')
    parser.add_argument('--with_POS', default=False, help='Utilize POS information.', action='store_true')
    parser.add_argument('--with_NER', default=False, help='Utilize NER information.', action='store_true')
    parser.add_argument('--POS_dim', type=int, default=20, help='Number of dimension for POS embeddings.')
    parser.add_argument('--NER_dim', type=int, default=20, help='Number of dimension for NER embeddings.')
    parser.add_argument('--wo_left_match', default=False, help='Without left to right matching.', action='store_true')
    parser.add_argument('--wo_right_match', default=False, help='Without right to left matching', action='store_true')
    parser.add_argument('--wo_full_match', default=False, help='Without full matching.', action='store_true')
    parser.add_argument('--wo_maxpool_match', default=False, help='Without maxpooling matching', action='store_true')
    parser.add_argument('--wo_attentive_match', default=False, help='Without attentive matching', action='store_true')
    parser.add_argument('--wo_max_attentive_match', default=False, help='Without max attentive matching.', action='store_true')
    parser.add_argument('--wo_char', default=False, help='Without character-composed embeddings.', action='store_true')

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

