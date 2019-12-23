#!/usr/bin/python3
# encoding: utf-8
# @author: lingyun
# @file: utils.py
# @time: 2019/7/8 10:56
# @desc:

import os
import pickle
from gensim.models import Word2Vec,KeyedVectors
import numpy as np

base_dir = os.getcwd()
print(base_dir)
corpus_path = os.path.join(base_dir,'datagrand/corpus.txt')
train_path = os.path.join(base_dir,'datagrand/train.txt')
test_path = os.path.join(base_dir,'datagrand/test.txt')

def get_words(path):
    words = []
    with open(path,'r') as f:
        for line in f.readlines():
            line_words = line.strip().split('_')
            words.extend(line_words)

    words = list(set(words))
    print(words)
    print(len(words))
    with open(base_dir + '/datagrand/words.txt','w') as wf:
        # line = '_'.join(words)
        # wf.write(line)
        for word in words:
            wf.write(word+'\n')

    return words

def get_BIO_label(item):
    str_label = []
    label = item[-1]
    item = item[:-2]
    chars = item.split('_')
    if label == 'o':
        for char in chars:
            str_label.append(char + ' ' + label + '\n')
    else:
        str_label.append(chars[0]+' B_'+label+'\n')
        # for char in chars[1:]:
        #     str_label.append(char+' I_'+label+'\n')
        #使用BIEO标记
        for char in chars[1:-1]:
            str_label.append(char+' I_'+label+'\n')
        #实体只有一个字符
        if len(chars[0:]) > 1:
            str_label.append(chars[-1] + ' E_' + label + '\n')

    return str_label


def turn_BIO(path):
    item = path.split('.txt')
    # item[-1] = '_proc.txt'
    item[-1] = '_proc_BIEO.txt'
    out_path = ''.join(item)
    print(out_path)

    text_label = []
    with open(path,'r') as f:
        for line in f.readlines():
            line_label = []
            #16784/c  4034_16201_11684_581/o  11061/c
            items = line.strip().split('  ')
            for item in items:
                #167_84/c----》['167 B_c\n' '84 I_c\n']
                str_label = get_BIO_label(item)
                line_label.extend(str_label)
            text_label.extend(line_label)
            text_label.extend('\n')

    with open(out_path,'w') as wf:
        for text in text_label:
            wf.write(text)

def get_word2id_corpus(path):
    item = path.split('.txt')
    item[-1] = '_word2id.pkl'
    out_path = ''.join(item)
    print(out_path)

    words = []
    with open(path,'r') as f:
        for line in f.readlines():
            line_words = line.strip().split('_')
            words.extend(line_words)

    words = list(set(words))
    print(words)
    print(len(words))

    vocab_dic = {}
    vocab_dic['<UNK>'] = 0
    for i, word in enumerate(words):
        vocab_dic[word] = i + 1

    with open(out_path,'wb') as wf:
        pickle.dump(vocab_dic, wf)

def get_word2id(path):
    item = path.split('.txt')
    item[-1] = '_word2id.pkl'
    out_path = ''.join(item)
    print(out_path)

    words = []
    with open(path,'r') as f:
        for line in f.readlines():
            line_label = []
            #16784/c  4034_16201_11684_581/o  11061/c
            items = line.strip().split('  ')
            for item in items:
                #167_84/c
                item = item[:-2]
                word = item.split('_')
                words.extend(word)

    words = list(set(words))
    print(len(words))
    vocab_dic = {}
    vocab_dic['<UNK>'] = 0
    for i, word in enumerate(words):
        vocab_dic[word] = i+1

    with open(out_path,'wb') as wf:
        pickle.dump(vocab_dic, wf)

def read_corpus_daguan(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        sent_ = line.strip().split('_')
        data.append((sent_, ['o']*len(sent_)))

    return data

def test_word2vec(path):
    model = Word2Vec.load(path)
    #model = KeyedVectors.load_word2vec_format(path, binary=True)
    vector = model.wv['45']
    em = np.array([model.wv[word] for word in model.wv.vocab])
    vo = model.wv.vocab
    print(len(vo))
    print(list(vo))
    print(set(vo))
    print(len(list(vo)))
    print(len(set(vo)))
    #print(model.wv.vocab)
    print(len(vector))
    print(len(em))

def get_word2vec_word2id(path):
    item = path.split('.txt')
    item[-1] = '_word2vec.model'
    out_path = ''.join(item)
    print(out_path)

    words = []
    sentences = []
    with open(path,'r') as f:
        for line in f.readlines():
            line_words = line.strip().split('_')
            words.extend(line_words)
            sentences.append(line_words)

    words = list(set(words))
    print(words)
    print(len(words))
    print(len(sentences))

    model = Word2Vec(sentences, sg=1, size=300, window=5, min_count=0, negative=5, sample=1e-4, workers=10)
    model.save(out_path)
    #model.wv.save_word2vec_format(out_path, binary=False)

    #获取word2id
    item = path.split('.txt')
    item[-1] = '_word2vec.pkl'
    out_path = ''.join(item)
    print(out_path)

    vocab_dic = {}
    vocab_dic['<UNK>'] = 0
    for i, word in enumerate(words):
        vocab_dic[word] = i + 1

    with open(out_path,'wb') as wf:
        pickle.dump(vocab_dic, wf)

def get_length_corpus(path):
    all = 0
    count_50 = 0
    count_100 = 0
    count_200 = 0
    count_300 = 0
    other_300 = 0

    with open(path,'r') as f:
        for line in f.readlines():
            line_words = line.strip().split('_')
            lens = len(line_words)
            all +=1
            if lens < 50:
                count_50 += 1
            elif lens < 100:
                count_100 +=1
            elif lens < 200:
                count_200 +=1
            elif lens < 300:
                count_300 +=1
            else:
                other_300 +=1

        print('The count length: all--{} /n <50: {}, <100: {}, <200: {}, <300: {}, 300+: {}'.format(all, count_50,count_100,count_200,count_300))



if __name__ == '__main__':
    #words = get_words(corpus_path)
    # turn_BIO(train_path)
    #get_word2id(train_path)
    #get_word2id_corpus(corpus_path)
    #test_word2vec(base_dir + '/datagrand/corpus_word2vec.model')
    #get_word2vec_word2id(corpus_path)

    #获取句子长度计数
    get_length_corpus(corpus_path)
