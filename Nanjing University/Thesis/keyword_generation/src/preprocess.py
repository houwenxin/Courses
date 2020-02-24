# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:18:44 2019

@author: houwenxin
"""
import os
import argparse
from data import load_and_tokenize_pairs, load_stopwords, tokenize_func, split_and_save_data, build_and_export_dataset
from vocab import build_vocab
import torch

parser = argparse.ArgumentParser()
opts = parser.parse_args()
src_data_path = "../raw_data"
src_data_name = "doctor"
output_path = "../data"
inp_fields = ["title", "abstract"]
trg_fields = ["keyword"]
opts.train_ratio = 0.8
opts.valid_ratio = 0.2
opts.test_ratio = 0.1
opts.src_trunc_to = 300
opts.max_src_length = 300
opts.min_src_length = 20
opts.trg_trunc_to = 6
opts.max_trg_length = 6
opts.min_trg_length = None
opts.vocab_size = 50000
opts.max_oov_words = 1000

if not os.path.exists(output_path):
    os.mkdir(output_path)

def main():
    print("Loading raw data...")
    src_data = os.path.join(src_data_path, src_data_name)
    stopwords = load_stopwords("utils/stopwords.txt")
    print("Load and tokenize / filter pairs...")
    tokenized_pairs = load_and_tokenize_pairs(source_data=src_data, 
                                              input_fields=inp_fields, 
                                              target_fields=trg_fields, 
                                              tokenize_func=tokenize_func,
                                              stopwords_list=stopwords,
                                              opts=opts)
    del stopwords
    print(len(tokenized_pairs))
    train_pairs, valid_pairs, test_pairs = split_and_save_data(tokenized_pairs, src_data_path, data_name="doctor")
    del tokenized_pairs
    
    print("Building vocabulary...")
    word2id, id2word, vocab = build_vocab(train_pairs, opts)
    print("Size of vocabulary: ", len(vocab))
    opts.vocab_path = os.path.join(output_path, 'doctor.vocab.pth')
    torch.save([word2id, id2word, vocab], open(opts.vocab_path, 'wb'))
    print("Vocabulary saved to ", opts.vocab_path)
    
    print("Exporting output data...")
    build_and_export_dataset(train_pairs, word2id, id2word, opts, output_path, data_name="doctor", data_type="train")
    build_and_export_dataset(valid_pairs, word2id, id2word, opts, output_path, data_name="doctor", data_type="valid")
    build_and_export_dataset(test_pairs, word2id, id2word, opts, output_path, data_name="doctor", data_type="test")
    
if __name__ == "__main__":
    main()