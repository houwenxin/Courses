# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:06:20 2019

@author: houwenxin
"""

import itertools

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
DIGIT = '<digit>'
SEP = '<sep>'

pre_defined_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<sep>"]
def build_vocab(train_pairs, opts):
    vocab = {}
    
    for text_tokens, keywords_tokens in train_pairs:
        tokens = text_tokens + list(itertools.chain(*keywords_tokens)) # itertools.chain(*)的功能就是把二维列表合并为一维列表
        for token in tokens:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    
    for token in pre_defined_tokens:
        if token in vocab:
            del vocab[token]
    
    word2id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        '<unk>': 3,
        '<sep>': 4,
    }
    id2word = {
        0: '<pad>',
        1: '<s>',
        2: '</s>',
        3: '<unk>',
        4: '<sep>',
    }    
    # word_freq_pair[0]为word（单词），word_freq_pair[1]为freq（出现次数）
    sorted_word2id = sorted(vocab.items(), key=lambda word_freq_pair: word_freq_pair[1], reverse=True)
    sorted_words = [word_freq_pair[0] for word_freq_pair in sorted_word2id]

    for index, word in enumerate(sorted_words):
        word2id[word] = index + 5  # 前5个被<pad>等占用，从第6个算起
    for index, word in enumerate(sorted_words):
        id2word[index + 5] = word  # here as well
        
    return word2id, id2word, vocab