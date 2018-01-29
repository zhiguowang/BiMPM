# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import re

# import math
class Vocab(object):
    def __init__(self, vec_path=None, dim=100, fileformat='bin',voc=None, word2id=None, word_vecs=None, unk_mapping_path=None):
        self.unk_label = '<unk>'
        self.stoplist = None
        if fileformat == 'bin':
            self.fromBinary(vec_path,voc=voc)
        elif fileformat == 'txt':
            self.fromText(vec_path,voc=voc)
        elif fileformat == 'txt2':
            self.fromText_format2(vec_path,voc=voc,pre_word_vecs=word_vecs)
        elif fileformat == 'txt3':
            self.fromText_format3(vec_path,voc=voc)
        elif fileformat == 'map':
            self.fromMap(word2id, word_vecs, word_dim=dim)
        else: # build a vocabulary with a word set
            self.fromVocabualry(voc, dim=dim)
        
        self.__unk_mapping = None
        if unk_mapping_path is not None:
            self.__unk_mapping = {}
            in_file = open(unk_mapping_path, 'rt')
            for line in in_file:
                items = re.split('\t', line)
                self.__unk_mapping[items[0]] = items[1]
            in_file.close()


    def fromVocabualry(self, voc, dim=100):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}
        
        self.vocab_size = len(voc) 
        self.word_dim = dim 
        for word in voc:
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index 
            self.id2word[cur_index] = word
        
#         self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
        shape = (self.vocab_size+1, self.word_dim)
        scale = 0.05
        self.word_vecs = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
#         self.word_vecs = None

    def fromMap(self, word2id, word_vecs, word_dim=100):
        self.word2id = word2id
        self.id2word = dict(zip(word2id.values(),word2id.keys()))
        
        self.vocab_size = len(word2id) 
        self.word_dim = word_dim 
        self.word_vecs = word_vecs



    def fromText(self, vec_path,voc=None):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}
        
        vec_file = open(vec_path, 'rt')
        header = vec_file.readline()
        self.vocab_size, self.word_dim = map(int, header.split())
        word_vecs = {}
        for line in vec_file:
            line = line.decode('utf-8').strip()
            parts = line.split(' ')
            word = parts[0]
            if (voc is not None) and (word not in voc): continue
            vector = np.array(parts[1:], dtype='float32')
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index 
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
        vec_file.close()

        self.vocab_size = len(self.word2id)
        self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
        for cur_index in xrange(self.vocab_size):
            self.word_vecs[cur_index] = word_vecs[cur_index]
    

    def fromText_format2(self, vec_path,voc=None,pre_word_vecs=None):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}
        
        vec_file = open(vec_path, 'rt')
        word_vecs = {}
        for line in vec_file:
            line = line.decode('utf-8').strip()
            parts = line.split('\t')
            cur_index = int(parts[0])
            word = parts[1]
            vector = np.array(map(float,re.split('\\s+', parts[2])), dtype='float32')
            self.word2id[word] = cur_index 
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
            self.word_dim = vector.size
        vec_file.close()

        self.vocab_size = len(self.word2id)

        if pre_word_vecs is not None:
            self.word_vecs = pre_word_vecs
        else:
            self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
            for cur_index in xrange(self.vocab_size):
                self.word_vecs[cur_index] = word_vecs[cur_index]


    def fromText_format3(self, vec_path,voc=None):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}
        
        vec_file = open(vec_path, 'rt')
#         header = vec_file.readline()
#         self.vocab_size, self.word_dim = map(int, header.split())
        word_vecs = {}
        for line in vec_file:
            line = line.decode('utf-8')
            if line[0] == line[1] == ' ':
                word = ' '
                parts = [' '] + line.strip().split(' ')
            else:
                parts = line.split(' ')
                word = parts[0]
            self.word_dim = len(parts[1:])
            if (voc is not None) and (word not in voc): continue
            vector = np.array(parts[1:], dtype='float32')
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index 
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
        vec_file.close()

        self.vocab_size = len(self.word2id)
        self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
        for cur_index in xrange(self.vocab_size):
            self.word_vecs[cur_index] = word_vecs[cur_index]



    def fromText_bak(self, vec_path,voc=None):
        # load freq table and build index for each word
        self.word2id = {}
        self.id2word = {}
        
        vec_file = open(vec_path, 'rt')
        header = vec_file.readline()
        self.vocab_size, self.word_dim = map(int, header.split())
        self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
        for line in vec_file:
            line = line.decode('utf-8').strip()
            parts = line.split(' ')
            word = parts[0]
            if (voc is not None) and (word not in voc): continue
            vector = np.array(parts[1:], dtype='float32')
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index 
            self.id2word[cur_index] = word
            self.word_vecs[cur_index] = vector
        vec_file.close()

    def fromBinary_with_voc(self, fname, voc, scale=0.05, stop_num=50):
        self.stoplist = voc[0:stop_num]
        voc = voc[stop_num:]
        voc.append(self.unk_label)
        self.word2id = {}
        self.id2word = {}
        for word in voc:
            curIndex = len(self.word2id)
            self.word2id[word] = curIndex 
            self.id2word[curIndex] = word

        with open(fname, "rb") as f:
            header = f.readline()
            cur_vocab_size, self.word_dim = map(int, header.split())
            word_vecs = {}
            binary_len = np.dtype('float32').itemsize * self.word_dim
            for idx in xrange(cur_vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in self.word2id.keys():
                    curIndex = self.word2id[word]
                else:
                    curIndex = len(self.word2id)
                    self.word2id[word] = curIndex 
                    self.id2word[curIndex] = word
                word_vecs[curIndex] = np.fromstring(f.read(binary_len), dtype='float32')  

        self.vocab_size = len(self.word2id)
        self.word_vecs = np.random.uniform(low=-scale, high=scale, size=(self.vocab_size+1, self.word_dim)).astype('float32')
        self.word_vecs[self.vocab_size] = self.word_vecs[self.vocab_size] * 0.0
        for cur_index in word_vecs.keys():
            self.word_vecs[cur_index] = word_vecs[cur_index]

    def fromBinary(self, fname, scale=0.05, voc=None):
        self.word2id = {}
        self.id2word = {}
        self.word2id[self.unk_label] = 0
        self.id2word[0] = self.unk_label
        # load word vector
        with open(fname, "rb") as f:
            header = f.readline()
            self.vocab_size, self.word_dim = map(int, header.split())
            word_vecs = {}
            binary_len = np.dtype('float32').itemsize * self.word_dim
            for idx in xrange(self.vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word == '': continue
                curIndex = len(self.word2id)
                self.word2id[word] = curIndex 
                self.id2word[curIndex] = word
                word_vecs[curIndex] = np.fromstring(f.read(binary_len), dtype='float32')  

        # add unkwords
        if voc is not None:
            for word in voc:
                if word == '': continue
                if self.word2id.has_key(word): continue
                curIndex = len(self.word2id)
                self.word2id[word] = curIndex 
                self.id2word[curIndex] = word
                word_vecs[curIndex] = np.random.uniform(low=-scale, high=scale, size=(self.word_dim,)).astype('float32') 

        self.vocab_size = len(self.word2id)
        self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
        for cur_index in xrange(self.vocab_size):
            if cur_index ==0 : continue
            self.word_vecs[cur_index] = word_vecs[cur_index]
        self.word_vecs[0] = np.random.uniform(low=-scale, high=scale, size=(self.word_dim,)).astype('float32') 
    
    def setWordvec(self,word_vecs):
        self.word_vecs = word_vecs

    def hasWord(self, word):
        return self.word2id.has_key(word)
    
    def size(self):
        return len(self.word2id)

    def getIndex(self, word):
        if self.stoplist is not None:
            if word in self.stoplist:
                return None
        if(self.word2id.has_key(word)):
            return self.word2id.get(word)
        else:
            return self.vocab_size

    def getWord(self, idx):
        return self.id2word.get(idx)

    def getVector(self, word):
        if(self.word2id.has_key(word)):
            idx = self.word2id.get(word)
            return self.word_vecs[idx]
        return None

    def to_index_sequence(self, sentence):
#         sentence = sentence.strip().lower()
        sentence = sentence.strip()
        seq = []
        for word in re.split('\\s+', sentence):
            idx = self.getIndex(word)
            if idx == None and self.__unk_mapping is not None and self.__unk_mapping.has_key(word):
                simWord = self.__unk_mapping[word]
                idx = self.getIndex(simWord)
            if idx == None: idx = self.vocab_size
            seq.append(idx)
        return seq

    def to_index_sequence_for_list(self, words):
        seq = []
        for word in words:
            idx = self.getIndex(word)
            if idx == None and self.__unk_mapping is not None and self.__unk_mapping.has_key(word):
                simWord = self.__unk_mapping[word]
                idx = self.getIndex(simWord)
            if idx == None: idx = self.vocab_size
            seq.append(idx)
        return seq

    def to_character_matrix(self, sentence, max_char_per_word=-1):
        sentence = sentence.strip()
        seq = []
        for word in re.split('\\s+', sentence):
            cur_seq = []
            for i in xrange(len(word)):
                cur_char = word[i]
                idx = self.getIndex(cur_char)
                if idx == None and self.__unk_mapping is not None and self.__unk_mapping.has_key(cur_char):
                    simWord = self.__unk_mapping[cur_char]
                    idx = self.getIndex(simWord)
                if idx == None: idx = self.vocab_size
                cur_seq.append(idx)
            if max_char_per_word != -1 and len(cur_seq) > max_char_per_word:
                cur_seq = cur_seq[:max_char_per_word]
            seq.append(cur_seq)
        return seq

    def to_index_sequence4binary_features(self, sentence):
        sentence = sentence.strip().lower()
        seq = []
        for word in re.split(' ', sentence):
            idx = self.getIndex(word)
            if idx == None: continue
            seq.append(idx)
        return seq

    def to_char_ngram_index_sequence(self, sentence):
        sentence = sentence.strip().lower()
        seq = []
        words = re.split(' ', sentence)
        for word in words:
            sub_words = collect_char_ngram(word)
            for sub_word in sub_words:
                idx = self.getIndex(sub_word)
                if idx == None: continue
                seq.append(idx)
        return seq

    def to_sparse_feature_sequence(self, sentence1, sentence2):
        words1 = set(re.split(' ', sentence1.strip().lower()))
        words2 = set(re.split(' ', sentence2.strip().lower()))
        intersection_words = words1.intersection(words2)
        seq = []
        for word in intersection_words:
            idx = self.getIndex(word)
            if idx == None: continue
            seq.append(idx)
        return seq

    def get_sentence_vector(self, sentence):
        sent_vec = np.zeros((self.word_dim,), dtype='float32')
        sentence = sentence.strip().lower()
        total = 0.0
        for word in re.split(' ', sentence):
            cur_vec = self.getVector(word)
            if cur_vec is None: continue
            sent_vec += cur_vec
            total += 1.0
        if total != 0.0: sent_vec /= total
        return sent_vec

    def dump_to_txt2(self, outpath):
        outfile = open(outpath, 'wt')
        for word in self.word2id.keys():
            cur_id = self.word2id[word]
            cur_vector = self.getVector(word)
#             print(word)
            word= word.encode('utf-8')
            outline = "{}\t{}\t{}".format(cur_id, word, vec2string(cur_vector))
            outfile.write(outline + "\n")
        outfile.close()

    def dump_to_txt3(self, outpath):
        outfile = open(outpath, 'wt')
        for word in self.word2id.keys():
            cur_vector = self.getVector(word)
            word= word.encode('utf-8')
            outline = word + " {}".format(vec2string(cur_vector))
            outfile.write(outline + "\n")
        outfile.close()

def vec2string(val):
    result = ""
    for v in val:
        result += " {}".format(v)
    return result.strip()


def collect_all_ngram(words, n=2): 
    all_ngrams = set()
    for i in xrange(len(words)-n):
        cur_ngram = words[i:i+n]
        all_ngrams.add(' '.join(cur_ngram))
    return all_ngrams

def collect_char_ngram(word, n=3):
    all_words = []
    if len(word)<=n: all_words.append(word)
    else:
        for i in xrange(len(word)-n+1):
            cur_word = word[i:i+3]
            all_words.append(cur_word)
    return all_words

def to_char_ngram_sequence(sentence, n=3):
    seq = []
    words = re.split(' ', sentence)
    for word in words:
        sub_words = collect_char_ngram(word)
        seq.extend(sub_words)
    return ' '.join(seq)

def collectVoc(trainpath):
    vocab = set()
    inputFile = file(trainpath, 'rt')
    for line in inputFile:
        line = line.strip()
        label, sentence = re.split('\t', line)
        sentence = sentence.lower()
        for word in re.split(' ', sentence):
            vocab.add(word)
    inputFile.close()
    return vocab

def collect_word_count(sentences, unk_num=1):
    word_count_map = {}
    for sentence in sentences:
        sentence = sentence.strip().lower()
        for word in re.split(' ', sentence):
            cur_count = 0
            if word_count_map.has_key(word):
                cur_count = word_count_map.get(word)
            word_count_map[word] = cur_count + 1
    word_count_list = []
    for word in word_count_map.keys():
        count = word_count_map.get(word)
        word_count_list.append((count, word))
    
    word_count_list = sorted(word_count_list,key=(lambda a:a[0]), reverse=True)
#     for i in xrange(50):
#         word, count = word_count_list[i]
#         print('{}\t{}'.format(word, count))
#     return word_count_list
    return [word for count, word in word_count_list if count>unk_num ]

def collect_word_count_with_max_vocab(sentences, max_vocab=600000):
    word_count_map = {}
    for sentence in sentences:
        sentence = sentence.strip().lower()
        for word in re.split(' ', sentence):
            cur_count = 0
            if word_count_map.has_key(word):
                cur_count = word_count_map.get(word)
            word_count_map[word] = cur_count + 1
    word_count_list = []
    for word in word_count_map.keys():
        count = word_count_map.get(word)
        word_count_list.append((count, word))
    
    word_count_list = sorted(word_count_list,key=(lambda a:a[0]), reverse=True)
#     for i in xrange(50):
#         word, count = word_count_list[i]
#         print('{}\t{}'.format(word, count))
#     return word_count_list
#     return [word for count, word in word_count_list if count>unk_num ]
    if len(word_count_list)<max_vocab: max_vocab = len(word_count_list)
    return [word for count, word in word_count_list[:max_vocab]]

def read_all_sentences(inpath):  
    all_sentences = []
    in_file = file(inpath, 'rt')
    for line in in_file:
        if line.startswith('<'): continue
        line = line.strip().lower()
        sentences = re.split('\t', line)
        for sentence in sentences:
            sentence = sentence.strip()
            all_sentences.append(sentence)
    in_file.close()
    return all_sentences

def read_sparse_features(inpath, threshold=0.0):  
    sparse_features = []
    in_file = file(inpath, 'rt')
    for line in in_file:
        line = line.strip().lower()
        items = re.split('\t', line)
        if len(items)!=2: continue
        (sparse_feature, count) = items
        count = float(count)
        if count< threshold: continue
        sparse_features.append(sparse_feature)
    in_file.close()
    return sparse_features

def build_word_index_file(word_vec_path, out_path):
    print('Loading word vectors ... ')
    vocab = Vocab(word_vec_path)
    print('Word_vecs shape: ', vocab.word_vecs.shape)
    word2id = vocab.word2id
    out_file = open(out_path,'wt')
    out_file.write('{}\t{}\n'.format(len(word2id), vocab.word_dim))
    for word in word2id.keys():
        wid = word2id[word]
        out_file.write('{}\t{}\n'.format(word, wid))
    out_file.close()
def load_word_index(index_path):
    word2id = {}
    in_file = open(index_path, 'rt')
    started = False
    for line in in_file:
        items = re.split('\t', line)
        if not started:
            started = True
            vocab_size = int(items[0])
            word_dim = int(items[1])
        else:
            if len(items)<2:
                word = ''
                word_id = int(items[0])
            else:
                word, word_id = items
            word2id[word] = int(word_id)
    in_file.close()
    return (vocab_size, word_dim, word2id)

if __name__ == '__main__':
    '''# load word vectors
    print('Loading word vectors ... ', end='')
    wordvec_path = '/u/zhigwang/zhigwang1/learn2rank/english_100_crop.bin'
    vocab = Vocab(wordvec_path)
    print('DONE!')
#     print(vocab.word_dim)
    print(vocab.to_index_sequence("usaa and ibm are partner ."))
    # '''
    '''
    while True:
        word = raw_input('Please input your word: ')
        if word == '=exit=':
            break
        pq = vocab.analogy(word)
        if pq is not None:
            while not pq.isEmpty():
                print(pq.pop())
    '''
    
    ''' # rank word based on count
    in_path = '/u/zhigwang/zhigwang1/paraphrase/data/paraphrase_train.tsv'
    sentences = read_all_sentences(in_path)
    words = collect_word_count(sentences)
    [print(word) for word in words]
    '''
    
    # build word vec index
    word_vec_path = '/u/zhigwang/zhigwang1/learn2rank/data/comp7.0/wordvec_crop.bin'
    out_path = '/u/zhigwang/zhigwang1/learn2rank/models/vocab'
    build_word_index_file(word_vec_path, out_path)
    print('DONE!')



