#!/usr/bin/env python
# -*- coding: utf-8 -*-

# assgin2.py
# Author: linyaoyi
# Email:  linyaoyi011@gmail.com 

from __future__ import division
from PStemmer import PorterStemmer
from tfidf import TfIdf
from optparse import OptionParser
import string
import codecs
import os
import sys

p = PorterStemmer()
tfidf = TfIdf()

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

def filter_stop_words(lines):
    s_words = stop_words()
    unstop_lines = []
    for line in lines:
        _ = filter(lambda x: x not in s_words, line.strip().split())
        unstop_lines.append(' '.join(_))
    return unstop_lines

 
def wordStemming(line):
    line = line.strip().split()
    words = []
    for word in line:
        words.append(p.stem(word, 0, len(word)-1))
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
 

def filter_splashes(word, num=2):
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

    lines = map(lambda x: ' '.join(filter(lambda i:filter_splashes(i), x.split())) , lines)
    return lines
  
def tf_idf(word, line):
    tf = frequence(word, line)
    idf = tfidf.idf
    return tf*idf[word]
 
def frequence(word, line):
    return line.count(word)/len(line)
 
def generateUnigramFeatures(lines, method='TF_IDF'):
    result = []
    for line in lines:
        line_result = {}
        for word in line.split():
            if method == 'TF_IDF':
                line_result[word] = tf_idf(word,line)    
            else:
                line_result[word] = frequence(word,line)
        result.append(line_result)
    return result   
 
def sentence(filepath):
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(line.strip())
    return lines
 
def chop(data_list, data_type, data_fold):
    data_len = len(data_list)
    data = []
    if data_type == 'train':
        data.extend(data_list[0:int(data_len*(data_fold-1)/5)])
        if data_fold<5:
            data.extend(data_list[int(data_len*(data_fold)/5):])
    else:
        data.extend(data_list[int(data_len*(data_fold-1)/5):int(data_len*(data_fold/5))])
    return data


def format_result(neg_result, pos_result, names, data_type, fold):
    result = ""
    for res in chop(neg_result, data_type, fold):
        result += '-1'
        _ = res.items()
        _.sort(key=lambda x:names[x[0]])
        for k,v in _:
            result += ' %s:%s'%(names[k],v)
        result += '\n'
    for res in chop(pos_result, data_type, fold):
        result += '1'
        _ = res.items()
        _.sort(key=lambda x:names[x[0]])
        for k,v in _:
            result += ' %s:%s'%(names[k],v)
        result += '\n'
    return result

def preprocess_one(data_type, fold):
    neg_lines = sentence('rt-polaritydata/rt-polarity.neg')
    pos_lines = sentence('rt-polaritydata/rt-polarity.pos')
    neg_lines = filter_punctuation(neg_lines)
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines)
    names = mapWordsToIDs(lines)
    neg_result = generateUnigramFeatures(neg_lines, method='FREQUENCE')
    pos_result = generateUnigramFeatures(pos_lines, method='FREQUENCE')
    return format_result(neg_result, pos_result, names, data_type, fold)

def preprocess_two(data_type, fold):
    neg_lines = sentence('rt-polaritydata/rt-polarity.neg')
    pos_lines = sentence('rt-polaritydata/rt-polarity.pos')
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
    return format_result(neg_result, pos_result, names, data_type, fold)
 
def preprocess_three(data_type, fold):
    neg_lines = sentence('rt-polaritydata/rt-polarity.neg')
    pos_lines = sentence('rt-polaritydata/rt-polarity.pos')
    neg_lines = filter_punctuation(neg_lines)
    pos_lines = filter_punctuation(pos_lines)
    neg_lines = filter_stop_words(neg_lines)
    pos_lines = filter_stop_words(pos_lines)
 
    lines = []
    lines.extend(neg_lines)
    lines.extend(pos_lines) 
    tfidf.calcidf(lines)
    names = mapWordsToIDs(lines)
    neg_result = generateUnigramFeatures(neg_lines, method='TF_IDF')
    pos_result = generateUnigramFeatures(pos_lines, method='TF_IDF')
    return format_result(neg_result, pos_result, names, data_type, fold)

def preprocess_four(data_type, fold): 
    neg_lines = sentence('rt-polaritydata/rt-polarity.neg')
    pos_lines = sentence('rt-polaritydata/rt-polarity.pos')
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
    return format_result(neg_result, pos_result, names, data_type, fold)

def get_experiment_basedir(exp):
    base_path = os.getcwd()
    dir_path = os.path.join(base_path, "experiments")
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    subdir_path = os.path.join(dir_path, "fold" + str(exp) + "/")
    if not os.path.isdir(subdir_path):
        os.makedirs(subdir_path)
    
    return subdir_path
    
def generate_input_file(exp, data_type, fold, result):
    filename = data_type + "ing_exp" + str(exp) + "_f" + str(fold) + ".dat"
    filepath = get_experiment_basedir(exp) + "/" + filename
    print "generate " + filename + " ..."
    with codecs.open(filepath, 'w', encoding='utf-8') as out_f:
        out_f.write(result)
        out_f.close()

def train_classifier(exp, fold):
    train_filename = "training_exp" + str(exp) + "_f" + str(fold) + ".dat"
    train_file = get_experiment_basedir(exp) + train_filename
    model_filename = "model_exp" + str(exp) + "_f" + str(fold)
    model_file = get_experiment_basedir(exp) + model_filename
    os.system("./svm_learn" + " " + train_file + " " + model_file)

def evaluate_classifier(exp, fold):
    test_filename = "testing_exp" + str(exp) + "_f" + str(fold) + ".dat"
    test_file = get_experiment_basedir(exp) + test_filename
    model_filename = "model_exp" + str(exp) + "_f" + str(fold)
    model_file = get_experiment_basedir(exp) + model_filename
    output_filename = "output_exp" + str(exp) + "_f" + str(fold)
    output_file = get_experiment_basedir(exp) + output_filename
    os.system("./svm_classify" + " " + test_file + " " + model_file + " " + output_file)

PREPROCESS = [
    preprocess_one,
    preprocess_two,
    preprocess_three,
    preprocess_four,
]

def do_experiment(exp=1, fold=1):
    print str(fold) + "-fold:preprocess data"
    result = PREPROCESS[exp - 1]('train', fold)
    generate_input_file(exp, 'train', fold, result)
    result = PREPROCESS[exp - 1]('test', fold)
    generate_input_file(exp, 'test', fold, result)
    print str(fold) + "-fold:training classifier"
    train_classifier(exp, fold)
    print str(fold) + "-fold:cross_validation"
    evaluate_classifier(exp, fold)
 
def parse_args():
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
    parser.add_option("-e", "--experiment",
                      action="store",
                      type='int',
                      dest="experiment",
                      default=1,
                      help="choose an experiment",)
    (options, args) = parser.parse_args()
    if not hasattr(options, 'experiment') or options.experiment > 5:
        print 'please choos an experiment: 1 - 4'
    return options




if __name__ == '__main__':
    options = parse_args()
    print "experiment: " + str(options.experiment)
    for fold in range(1, 6):
        do_experiment(options.experiment,fold)
