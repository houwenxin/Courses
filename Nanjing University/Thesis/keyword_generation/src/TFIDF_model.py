# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:09:16 2019

@author: houwenxin
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from sklearn.externals import joblib
import os
import logging
import sys
from evaluate import get_match_result, evaluate


def load_text_data(data_path, logger, data_type):
    dataset = []
    keys = ["text_tokens", "keywords_tokens"]
    if data_type == "train":
        train_one2many = torch.load(data_path + ".train.one2many.pth", "rb")
        for single in train_one2many:
            data = {}
            for key in keys:
                data[key] = single[key]
            dataset.append(data)
        del train_one2many
    elif data_type == "valid":
        valid_one2many = torch.load(data_path + '.valid.one2many.pth', 'rb')
        for single in valid_one2many:
            data = {}
            for key in keys:
                data[key] = single[key]
            dataset.append(data)
        del valid_one2many
    elif data_type == "test":
        test_one2many = torch.load(data_path + '.test.one2many.pth', 'rb')
        for single in test_one2many:
            data = {}
            for key in keys:
                data[key] = single[key]
            dataset.append(data)
        del test_one2many

    logger.info('%s data size: %d' % (data_type, len(dataset)))
    return dataset

def TFIDF_keyword_extraction(documents, mode="train", extract_num=1):
    all_documents = []
    for document in documents:
        all_documents.append(" ".join(document["text_tokens"]))
    
    del documents
    if mode == "train":    
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.75, max_features=12500)
        model = vectorizer.fit(all_documents)
        if not os.path.exists("../tfidf_model"):
            os.mkdir("../tfidf_model")
        joblib.dump(model, "../tfidf_model/tfidf.model")
    else:
        model = joblib.load("../tfidf_model/tfidf.model")
    tfidf = model.transform(all_documents)
    tfidf_array = tfidf.toarray()
    print("TF-IDF Data Matrix Size: ", tfidf_array.shape)
    
    del all_documents, tfidf, model # 释放没用的内存
    if mode == "train":
        dictionary = dict(zip(vectorizer.vocabulary_.values(), vectorizer.vocabulary_.keys()))
        joblib.dump(dictionary, "../tfidf_model/tfidf.dict")
    else:
        dictionary = joblib.load("../tfidf_model/tfidf.dict")
        
    keywords = []
    for row in range(tfidf_array.shape[0]):
        keyword = []
        idxs = np.argsort(tfidf_array[row])[::-1][:extract_num]
        for idx in idxs:
            keyword.append([dictionary[idx]])
        keywords.append(keyword)
        
    del tfidf_array, dictionary, keyword # 继续释放没用的内存
    
    print("%s Keywords are successfully extracted." % mode.upper())
    print("-----------------------------------------------------------------")
    return keywords

def write_result_to(file_path, true_keywords, pred_keywords):
    print("Writing true and predicted keywords to file {}...".format(file_path))
    with open(file_path, "w", encoding="utf-8") as file:
        for i in range(len(true_keywords)):
            true_keyword = list_to_str(true_keywords[i])#.lower() # 把labels中的英文字母全部换成小写
            pred_keyword = list_to_str(pred_keywords[i])
            pairs = "True keywords: " + true_keyword + "\tPredictions: " + pred_keyword + "\n"
            file.write(pairs)
    print("Done.")

def list_to_str(lst):
    string = "" + lst[0][0]
    for i in range(1, len(lst)):
        string += " "
        string += lst[i][0]
    return string

def extract_keywords(dataset):
    keywords = []
    for single in dataset:
        keyword = single["keywords_tokens"]
        #keywords.append(list(itertools.chain(*keyword)))
        keywords.append(keyword)
    #print(keywords)
    return keywords


def init_logging(logger_name, log_file, redirect_to_stdout=False, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(level)

    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.setLevel(level)

    if redirect_to_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)

    logger.info('Initializing logger: %s' % logger_name)
    logger.info('Making log output file: %s' % log_file)
    logger.info(log_file[: log_file.rfind(os.sep)])

    #logger.removeHandler(fh)
    return logger, fh

def evaluate_on_dataset(data_path, data_type):
    log_file = os.path.join("..", "tfidf_model", "tfidf_output_%s.log" % data_type)
    if os.path.exists(log_file):
        os.remove(log_file)
    logger, file_handler = init_logging("TFIDF.py", log_file)

    dataset = load_text_data(data_path, logger, data_type)
    pred_keywords = TFIDF_keyword_extraction(dataset, mode=data_type, extract_num=10)
    true_keywords = extract_keywords(dataset)
    score_dict = {}
    topk_range = [5, 10]
    score_names = ['precision', 'recall', 'f_score']
    for true_keyword, pred_keyword in zip(true_keywords, pred_keywords):
        logger.info("-----------------------------------")
        logger.info(true_keyword)
        logger.info(pred_keyword)
        match_score_list_exact = get_match_result(true_seqs=true_keyword, pred_seqs=pred_keyword, type='exact')      
        match_score_list_soft = get_match_result(true_seqs=true_keyword, pred_seqs=pred_keyword, type='partial')
        for topk in topk_range:
            results_exact = evaluate(match_score_list_exact, pred_keyword, true_keyword, topk=topk)
            results_soft = evaluate(match_score_list_soft, pred_keyword, true_keyword, topk=topk)
            for k, v in zip(score_names, results_exact):
                if '%s@%d_exact' % (k, topk) not in score_dict:
                    score_dict['%s@%d_exact' % (k, topk)] = []
                score_dict['%s@%d_exact' % (k, topk)].append(v)
                
            for k, v in zip(score_names, results_soft):
                if '%s@%d_soft' % (k, topk) not in score_dict:
                    score_dict['%s@%d_soft' % (k, topk)] = []
                score_dict['%s@%d_soft' % (k, topk)].append(v)
    
    for topk in topk_range:
        logger.info("--------------------------------EXACT: %d----------------------" % topk)
        logger.info("\n --- total precision, recall, fscore: " + str(np.average(score_dict['precision@%d_exact' % (topk)])) + " , " +\
                            str(np.average(score_dict['recall@%d_exact' % (topk)])) + " , " +\
                            str(np.average(score_dict['f_score@%d_exact' % (topk)])))
        logger.info("------------------------------SOFT: %d--------------------------" % topk)
        logger.info("\n --- total precision, recall, fscore: " + \
                            str(np.average(score_dict['precision@%d_soft' % (topk)])) + " , " +\
                            str(np.average(score_dict['recall@%d_soft' % (topk)])) + " , " +\
                            str(np.average(score_dict['f_score@%d_soft' % (topk)])))
    
    #result_file_path = "../tfidf_model/tfidf_true_pred_pairs_%s.txt" % data_type
    #print(true_keywords)
    #write_result_to(result_file_path, true_keywords, pred_keywords)
    
    del dataset, true_keywords, pred_keywords
    
    logger.removeHandler(file_handler)
    
def main():
    data_path = "../data/doctor"
    evaluate_on_dataset(data_path, "train")
    evaluate_on_dataset(data_path, "valid")
    evaluate_on_dataset(data_path, "test")
    
    
if __name__ == "__main__":
    main()