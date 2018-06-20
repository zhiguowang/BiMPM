# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import argparse
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import time
import re
import tensorflow as tf
import json
import logging

import matplotlib
#matplotlib.use('TKAgg')  # necessary on OS X
matplotlib.use('Agg')
from matplotlib import pyplot as pl

from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils
import jieba

jieba.load_userdict('mydict/mydict.txt')

# 获取logger实例，如果参数为空则返回root logger
logger = logging.getLogger("BiMPM")
# 指定logger输出格式
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

# 文件日志
file_handler = logging.FileHandler("train.log")
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter  # 也可以直接给formatter赋值

# 为logger添加的日志处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 指定日志的最低输出级别，默认为WARN级别
logger.setLevel(logging.INFO)


def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt')
    i = 0
    for line in infile:
        #line = line.decode('utf-8').strip()
        if line.startswith('-'): continue
        lineno, sentence1, sentence2, label = line.strip().split('\t')
        # items = re.split("\t", line)
        # label = items[0]
        # sentence1 = re.split("\\s+",items[1].lower())
        # sentence2 = re.split("\\s+",items[2].lower())
        all_labels.add(label)
        # 中文分词， 如果不分词，all_words跟all_chars一样。
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        stopwords = '，。！？*'
        words1 = [w for w in jieba.cut(sentence1) if w.strip() and w not in stopwords]
        words2 = [w for w in jieba.cut(sentence2) if w.strip() and w not in stopwords]
        #sentence1 = ' '.join(words1)
        #sentence2 = ' '.join(words2)
        sentence1 = words1
        sentence2 = words2
        i += 1
        if i < 3:
            print(sentence1)
            print(type(sentence1))
            print(all_words)
        all_words.update(sentence1)
        if i < 3:
            print(all_words)
        all_words.update(sentence2)
        # if with_POS:
        #     all_POSs.update(re.split("\\s+",items[3]))
        #     all_POSs.update(re.split("\\s+",items[4]))
        # if with_NER:
        #     all_NERs.update(re.split("\\s+",items[5]))
        #     all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        #print(word)
        #print(type(word))
        for char in word:
            #print(char)
            #print(type(char))
            all_chars.add(char)
    # print(all_chars)
    # print('-------------------------------------------------')
    # print(all_words)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def output_probs(probs, label_vocab):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()


def evaluation(sess, valid_graph, devDataStream, outpath=None, label_vocab=None):
    if outpath is not None:
        result_json = {}
    # 评估标准
    TP = 0  # True Positive（TP）意思表示做出同义的判定，而且判定是正确的，TP的数值表示正确的同义判定的个数；
    FP = 0  # False Positive（FP）数值表示错误的同义判定的个数；
    TN = 0  # True Negative（TN）数值表示正确的不同义判定个数；
    FN = 0  # False Negative（FN）数值表示错误的不同义判定个数。

    F1_score = -1
    total = 0
    correct = 0
    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=True)
        [cur_correct, probs, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        correct += cur_correct
        if outpath is not None:
            for i in range(cur_batch.batch_size):
                (label, sentence1, sentence2, _, _, _, _, _, cur_ID) = cur_batch.instances[i]
                result_json[cur_ID] = {
                    "ID": cur_ID,
                    "truth": label,
                    "sent1": sentence1,
                    "sent2": sentence2,
                    "prediction": label_vocab.getWord(predictions[i]),
                    "probs": output_probs(probs[i], label_vocab),
                }

                if label_vocab.getWord(predictions[i]) == "1":
                    if label == "1":
                        TP += 1
                    else:
                        FP += 1
                else:
                    if label == "0":
                        TN += 1
                    else:
                        FN += 1
    # 准确率（precision rate）、召回率（recall rate）和accuracy、F1-score
    acc = correct / total * 100
    if outpath is not None:
        try:
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            precision_rate = TP / (TP + FP)
            recall_rate = TP / (TP + FN)
            F1_score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)

            print('************evaluation result****************')
            print('**********TP: {}'.format(TP))
            print('**********FP: {}'.format(FP))
            print('**********TN: {}'.format(TN))
            print('**********FN: {}'.format(FN))
            print('**********precision_rate: {:.4f}'.format(precision_rate))
            print('**********recall_rate: {:.4f}'.format(recall_rate))
            print('**********accuracy: {:.4f}'.format(accuracy))
            print('**********F1_score: {:.4f}'.format(F1_score))
            print('************evaluation result****************')
        except ZeroDivisionError:
            logger.error('ZeroDivisionError occur!')
            F1_score = -1
        with open(outpath, 'w') as outfile:
            json.dump(result_json, outfile)

    return acc, F1_score


# 预测评估函数
def predict(sess, valid_graph, devDataStream, outpath=None, label_vocab=None):
    """
    预测结果生成格式 ： "行号\t预测结果"
    :param sess:
    :param valid_graph:
    :param devDataStream:
    :param outpath:
    :param label_vocab:
    :return:
    """
    if not outpath:
        print('输出文件不存在！')
        return "预测失败！"
    result_file = open(outpath, 'w')
    total = 0
    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=True)
        [_, _, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)

        for i in range(cur_batch.batch_size):
            (label, sentence1, sentence2, _, _, _, _, _, cur_ID) = cur_batch.instances[i]
            result_file.write(cur_ID + '\t' + label_vocab.getWord(predictions[i]) + '\n')

    result_file.close()
    print("预测数量： {}".format(total))
    return "预测完成！"


def train(sess, saver, train_graph, valid_graph, trainDataStream,
          devDataStream, options, best_path):
    best_acc = -1
    # 损失函数
    train_loss = []
    # 准确率
    dev_accuracy = []

    for epoch in range(options.max_epochs):
        logger.info('Train in epoch {}'.format(epoch))
        # training
        trainDataStream.shuffle()
        num_batch = trainDataStream.get_num_batch()
        start_time = time.time()
        total_loss = 0
        for batch_index in range(num_batch):  # for each batch
            cur_batch = trainDataStream.get_batch(batch_index)
            feed_dict = train_graph.create_feed_dict(cur_batch, is_training=True)
            _, loss_value = sess.run([train_graph.train_op, train_graph.loss], feed_dict=feed_dict)
            total_loss += loss_value
            if batch_index % 100 == 0:
                print('{} '.format(batch_index), end="")
                sys.stdout.flush()

        print()
        duration = time.time() - start_time
        epoch_loss = total_loss / num_batch
        logger.info('Epoch {}: loss = {:.4f} ({:.4f} sec)'.format(epoch, epoch_loss, duration))
        train_loss.append(epoch_loss)
        # evaluation
        acc, _ = evaluation(sess, valid_graph, devDataStream)
        dev_accuracy.append(acc)
        logger.info("Accuracy: {:.4f}".format(acc))
        if acc >= best_acc:
            best_acc = acc
            saver.save(sess, best_path)

    # 画图
    # 1.Train Loss
    epoch_seq = range(0, options.max_epochs, 1)
    pl.plot(epoch_seq, train_loss, 'k-', label='Train Set')
    pl.title('Train Loss')
    pl.xlabel('Epochs')
    pl.ylabel('Loss')
    # pl.show(block=False)
    pl.savefig('Loss.png')
    # 2.Dev Accuracy
    pl.plot(epoch_seq, dev_accuracy, 'r-', label='Dev Set')
    pl.title('Train Loss')
    pl.xlabel('Epochs')
    pl.ylabel('Accuracy')
    # pl.show(block=False)
    pl.savefig('accuracy.png')


def main(FLAGS):
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
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
    has_pre_trained_model = False
    char_vocab = None
    if os.path.exists(best_path + ".index"):
        has_pre_trained_model = True
        logger.info('Loading vocabs from a pre-trained model ...')
        label_vocab = Vocab(label_path, fileformat='txt2')
        if FLAGS.with_char: char_vocab = Vocab(char_path, fileformat='txt2')
    else:
        logger.info('Collecting words, chars and labels ...')
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path)
        logger.info('Number of words: {}'.format(len(all_words)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels,dim=2)
        label_vocab.dump_to_txt2(label_path)

        if FLAGS.with_char:
            logger.info('Number of chars: {}'.format(len(all_chars)))
            char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=FLAGS.char_emb_dim)
            char_vocab.dump_to_txt2(char_path)
        
    logger.info('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    logger.info("Number of labels: {}".format(num_classes))
    sys.stdout.flush()

    logger.info('Build SentenceMatchDataStream ... ')
    trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab,
                                              isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    logger.info('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    logger.info('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    sys.stdout.flush()
                                    
    devDataStream = SentenceMatchDataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True, isSort=True, options=FLAGS)
    logger.info('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    logger.info('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  is_training=True, options=FLAGS, global_step=global_step)

        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                is_training=False, options=FLAGS)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
#             if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
         
        sess = tf.Session()
        sess.run(initializer)
        if has_pre_trained_model:
            logger.info("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            logger.info("DONE!")

        # training
        train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, FLAGS, best_path)

def enrich_options(options):
    if not options.__dict__.has_key("in_format"):
        options.__dict__["in_format"] = 'tsv'

    return options


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
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--with_highway', default=False, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_full_match', default=False, help='With full matching.', action='store_true')
    parser.add_argument('--with_maxpool_match', default=False, help='With maxpooling matching', action='store_true')
    parser.add_argument('--with_attentive_match', default=False, help='With attentive matching', action='store_true')
    parser.add_argument('--with_max_attentive_match', default=False, help='With max attentive matching.', action='store_true')
    parser.add_argument('--with_char', default=False, help='With character-composed embeddings.', action='store_true')
    
    parser.add_argument('--config_path', type=str, help='Configuration file.')

#     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    args, unparsed = parser.parse_known_args()
    if args.config_path is not None:
        logger.info('Loading the configuration from ' + args.config_path)
        FLAGS = namespace_utils.load_namespace(args.config_path)
    else:
        FLAGS = args
    sys.stdout.flush()
    
    # enrich arguments to backwards compatibility
    FLAGS = enrich_options(FLAGS)

    main(FLAGS)

