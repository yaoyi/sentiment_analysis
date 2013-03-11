#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
assgin2.py
Author: linyaoyi
Email:  linyaoyi011@gmail.com
 
Created on
2013-02-28
'''

from __future__ import division
from nltk.stem.wordnet import WordNetLemmatizer
from math import log
from optparse import OptionParser
import string
import codecs
import os
import sys

lmtzr = WordNetLemmatizer()
global IDF
 
def getTop2000Words(lines):
    words = {}
    for line in lines:
        line = line.strip().split()
        for word in line:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words = words.items()
    words.sort(key = lambda x: x[1], reverse=True)
    return [ i[0] for i in words[:2000]]
 
def filter_top_words(lines, words):
    legal_lines = []
    for line in lines:
        line = filter(lambda x: x in words, line.split())
        legal_lines.append(' '.join(line))
    return legal_lines
 
def wordStemming(line):
    line = line.strip().split()
    words = []
    for word in line:
        words.append(lmtzr.lemmatize(word))
    return ' '.join(words)
        
def mapWordsToIDs(lines):
    words = {}
    id = 1
    for line in lines:
        line = line.strip().split()
        for word in line:
            if not word in words:
                words[word] = id
                id += 1
    return words
 
def stop_words():
    words = []
    with open('stopwords.txt') as f:
        for word in f:
            words.append(word.strip())
    return words
 
def filter_stop_words(lines):
    s_words = stop_words()
    unstop_lines = []
    for line in lines:
        _ = filter(lambda x: x not in s_words, line.strip().split())
        unstop_lines.append(' '.join(_))
    return unstop_lines
 
def return_big(word, num=2):
    if len(word) > num:
        return word
 
def filter_punctuation(lines):
    lines = map(lambda x: x.replace('.', ' ').replace(',', ' ').replace('"', ' ').replace( "'", ' ')\
            .replace('(', ' ').replace(')', ' ').replace(':', ' ')\
            .replace('--', ' ').replace('/', ' '), lines) 
    # table = string.maketrans("","")
    # filter_lines = []
    # for line in lines:
    #     filter_lines.append(line.translate(table, string.punctuation))
    # print filter_lines

    lines = map(lambda x: ' '.join(filter(lambda i:return_big(i), x.split())) , lines)
    return lines
 
class idf(object):
    def __init__(self, lines):
        result = {}
        words = set()
        total_count = 0
        for line in lines:
            for word in line.split():
                words.add(word)
                total_count += 1
        for word in words:
            result[word] = 0
 
        for line in lines:
            for word in words:
                if word in line:
                    result[word] += 1
        for i,j in result.items():
            result[i] = log(total_count/j, 2)
        self.result = result
 
    @property
    def res(self):
        return self.result
 
 
def tf_idf(word, line):
    global IDF
    tf = frequence(word, line)
    idf = IDF.res[word]
    return tf*idf
 
def frequence(word, line):
    return line.count(word)/len(line)
 
METHOD = dict(
        TF_IDF = tf_idf,
        FREQUENCE = frequence,
        )
 
def generateUnigramFeatures(lines, method='TF_IDF'):
    result = []
    for line in lines:
        line_result = {}
        for word in line.split():
            line_result[word] = METHOD[method](word, line)
        result.append(line_result)
    return result   
 
def baye(word_one, word_two, line):
    result = line.count('%s %s'%(word_one, word_two))/ line.count(word_one)
    return result
 
def generateBigramFeatures(lines):
    result = []
    for line in lines:
        line_result = {}
        words = line.split()
        for i in xrange(len(words)-1):
            line_result['%s-%s'%(words[i], words[i+1])] = baye(words[i], words[i+1], line)
        result.append(line_result)
    return result
 
def neg_sentence():
    neg_lines = []
    with open('rt-polaritydata/rt-polarity.neg') as f:
        for line in f:
            neg_lines.append(line.strip())
    return neg_lines
 
def pos_sentence():
    pos_lines = []
    with open('rt-polaritydata/rt-polarity.pos') as f:
        for line in f:
            pos_lines.append(line.strip())
    return pos_lines
 
DATA_TYPE = dict(
            test = 1/5,
            train = 4/5
        )
 
def chop(data_list, data_type, data_corpus):
    data_len = len(data_list)
    data = []
    if data_type == 'train':
        data.extend(data_list[0:int(data_len*(data_corpus-1)/5)])
        if data_corpus<5:
            data.extend(data_list[int(data_len*(data_corpus)/5):])
    else:
        data.extend(data_list[int(data_len*(data_corpus-1)/5):int(data_len*(data_corpus/5))])
    return data
 
def expriment_one(data_type='train', corpus=1):
    neg_lines = neg_sentence()
    pos_lines = pos_sentence()
    neg_lines = filter_punctuation(neg_lines)
    # print neg_lines;
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines)
    names = mapWordsToIDs(lines)
    neg_result = generateUnigramFeatures(neg_lines, method='FREQUENCE')
    pos_result = generateUnigramFeatures(pos_lines, method='FREQUENCE')
    return format_result(neg_result, pos_result, names, data_type, corpus)
 
def expriment_two(data_type='train', corpus=1):
    neg_lines = neg_sentence()
    pos_lines = pos_sentence()
    neg_lines = filter_punctuation(neg_lines)
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
    neg_lines = filter(lambda x:wordStemming(x), neg_lines)
    pos_lines = filter(lambda x:wordStemming(x), pos_lines)
 
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines)
    names = mapWordsToIDs(lines)
    neg_result = generateUnigramFeatures(neg_lines, method='FREQUENCE')
    pos_result = generateUnigramFeatures(pos_lines, method='FREQUENCE')
    return format_result(neg_result, pos_result, names, data_type, corpus)
 
def expriment_three(data_type='train', corpus=1):
    neg_lines = neg_sentence()
    pos_lines = pos_sentence()
    neg_lines = filter_punctuation(neg_lines)
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
    neg_lines = filter(lambda x:wordStemming(x), neg_lines)
    pos_lines = filter(lambda x:wordStemming(x), pos_lines)
 
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines)
    global IDF
    IDF = idf(lines)
    names = mapWordsToIDs(lines)
    neg_result = generateUnigramFeatures(neg_lines, method='TF_IDF')
    pos_result = generateUnigramFeatures(pos_lines, method='TF_IDF')
    return format_result(neg_result, pos_result, names, data_type, corpus)
 
def expriment_four(data_type='train', corpus=1):
    neg_lines = neg_sentence()
    pos_lines = pos_sentence()
    neg_lines = filter_punctuation(neg_lines)
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
    neg_lines = filter(lambda x:wordStemming(x), neg_lines)
    pos_lines = filter(lambda x:wordStemming(x), pos_lines)
    neg_top_words = getTop2000Words(neg_lines)
    pos_top_words = getTop2000Words(pos_lines)
    neg_lines = filter_top_words(neg_lines, neg_top_words)
    pos_lines = filter_top_words(pos_lines, pos_top_words)
 
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines)
    names = mapWordsToIDs(lines)
    neg_result = generateUnigramFeatures(neg_lines, method='FREQUENCE')
    pos_result = generateUnigramFeatures(pos_lines, method='FREQUENCE')
    return format_result(neg_result, pos_result, names, data_type, corpus)
 
def expriment_five(data_type='train', corpus=1):
    neg_lines = neg_sentence()
    pos_lines = pos_sentence()
    neg_lines = filter_punctuation(neg_lines)
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines)
    global IDF
    IDF = idf(lines)
    
    neg_result = generateBigramFeatures(neg_lines)
    pos_result = generateBigramFeatures(pos_lines)
    lines = [' '.join(i.keys()) for i in neg_result]
    lines.extend([' '.join(i.keys()) for i in pos_result])
    names = mapWordsToIDs(lines)
    return format_result(neg_result, pos_result, names, data_type, corpus)
 
def format_result(neg_result, pos_result, names, data_type, corpus):
    result = ""
    for res in chop(neg_result, data_type, corpus):
        result += '-1'
        _ = res.items()
        _.sort(key=lambda x:names[x[0]])
        for k,v in _:
            result += ' %s:%s'%(names[k],v)
        result += '\n'
    for res in chop(pos_result, data_type, corpus):
        result += '1'
        _ = res.items()
        _.sort(key=lambda x:names[x[0]])
        for k,v in _:
            result += ' %s:%s'%(names[k],v)
        result += '\n'
    return result
 
EXPERIMENT = [expriment_one,
            expriment_two,
            expriment_three,
            expriment_four]
 
if __name__ == '__main__':
    print "prepare directories"
    sys.stdout.flush()
    base_path = os.getcwd()
    dir_path = os.path.join(base_path, "experiments")
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    for exp_index in range(1,4):
        subdir_path = os.path.join(dir_path, "fold" + str(exp_index))
        if not os.path.isdir(subdir_path):
            os.makedirs(subdir_path)
        print "perform experiment " + str(exp_index) + " ..."
        for fold_index in range(1, 5):
            train_data = EXPERIMENT[exp_index - 1]('train',fold_index)
            test_data = EXPERIMENT[exp_index - 1]('test',fold_index)
            filename = subdir_path + "/training_exp_" + str(exp_index) + "_f" + str(fold_index) + ".dat"
            print "generate " + filename + " ..."
            with codecs.open(filename, 'w', encoding='utf-8') as out_f:
                out_f.write(train_data)
                out_f.close()
            filename = subdir_path + "/testing_exp_" + str(exp_index) + "_f" + str(fold_index) + ".dat"
            print "generate " + filename + " ..."
            with codecs.open(filename, 'w', encoding='utf-8') as out_f:
                out_f.write(test_data)
                out_f.close()


    # parser = OptionParser(usage="usage: %prog [options] filename",
    #                       version="%prog 1.0")
    # parser.add_option("-t", "--type",
    #                   action="store",
    #                   dest="data_type",
    #                   default='train',
    #                   help="choose a data type")
    # parser.add_option("-c", "--corpus",
    #                   action="store", # optional because action defaults to "store"
    #                   dest="data_corpus",
    #                   default="1",
    #                   help="choose a data_corpus")
    # parser.add_option("-e", "--experiment",
    #                   action="store", # optional because action defaults to "store"
    #                   dest="experiment",
    #                   default="one",
    #                   help="choose your experiment",)
    # (options, args) = parser.parse_args()
    # if hasattr(options, 'data_corpus') and hasattr(options, 'data_type'):
    #     data_type = options.data_type
    #     data_corpus = options.data_corpus
    #     experiment = options.experiment
    #     EXPERIMENT[experiment](data_type, int(data_corpus))
    # else:
    #     print 'please choos an experiment'