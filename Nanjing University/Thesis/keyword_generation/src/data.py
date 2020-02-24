import torch
import jieba
import os
import pickle
import codecs
import json
import re
from vocab import UNK

def load_pairs(source_data, input_fields, target_fields, delimiter=",", stop_line_for_test=100):
    pairs = []
    with codecs.open(source_data, "r", "utf-8") as src_data:
        for idx, line in enumerate(src_data):
            if(stop_line_for_test and idx >= stop_line_for_test): # 测试用
                break
            json_line = json.loads(line)
            inp_data = "。".join([json_line[field] for field in input_fields])
            trg_data = []
            [trg_data.extend(json_line[field]) for field in target_fields]
            # if (not inp_data.isalnum()) and (not keyword.isalnum() for keyword in trg_data):  # 判断是否原文都是英文，都是英文的不能用
            # pairs的格式：[(text, keyword_list), ...]
            pairs.append((inp_data, trg_data))
    print("Pairs Number: ", len(pairs))
    return pairs

def tokenize_and_filter_pairs(pairs, tokenize_func, stopwords_list, opts):
    tokenized_pairs = []
    for idx, (text, keywords_list) in enumerate(pairs):
        keep_text = True # 是否要保留这一对
        
        text_tokens = list(tokenize_func(text, stopwords_list)) # 转换成列表格式
        # 筛选：
        if opts.src_trunc_to and len(text_tokens) > opts.src_trunc_to:
            text_tokens = text_tokens[:opts.src_trunc_to]
        if opts.max_src_length and len(text_tokens) > opts.max_src_length:
            keep_text = False
        if opts.min_src_length and len(text_tokens) < opts.min_src_length:
            keep_text=False
        if not keep_text:
            print("Index %d's text is dropped." %idx)
            #print(text_tokens)
            #print(keywords_list)
            continue
        
        keywords_tokens = []
        for keyword in keywords_list:
            keep_keyword = True
            
            keyword_tokens = list(tokenize_func(keyword, stopwords_list))
            if opts.trg_trunc_to and len(keyword_tokens) > opts.trg_trunc_to:
                keyword_tokens = keyword_tokens[:opts.trg_trunc_to]
            if opts.max_trg_length and len(keyword_tokens) > opts.max_trg_length:
                keep_keyword = False
            if opts.min_trg_length and len(keyword_tokens) < opts.min_trg_length:
                keep_keyword = False
            if not keep_keyword:
                print("Keyword %s of index %d is dropped." %(keyword, idx))
                continue
            if len(keyword_tokens) > 0:
                keywords_tokens.append(keyword_tokens)
        tokenized_pairs.append([text_tokens, keywords_tokens])
        
        if(idx % 20000 == 0):
            print("-" * 100)
            print("Index: ", idx)
            print(text)
            print(text_tokens)
            print(keywords_list)
            print(keywords_tokens)
    return tokenized_pairs

def load_and_tokenize_pairs(source_data, input_fields, target_fields, tokenize_func, stopwords_list, opts):
    # 缓存，以便下次直接使用
    tokenized_pairs_cache_path = source_data + "_tokenized_pairs.pkl"
    if os.path.exists(tokenized_pairs_cache_path):
        with open(tokenized_pairs_cache_path, "rb") as cache:
            print("Loading tokenized pairs from " + tokenized_pairs_cache_path)
            tokenized_pairs = pickle.load(cache) # 如果有就直接用pickle载入就行了
    else: # 如果之前没有缓存过的话
        pairs = load_pairs(source_data, input_fields, target_fields)
        tokenized_pairs = tokenize_and_filter_pairs(pairs=pairs, tokenize_func=tokenize_func, stopwords_list=stopwords_list, opts=opts)
        #print(tokenized_pairs)
        del pairs
        with open(tokenized_pairs_cache_path, "wb") as cache:
            pickle.dump(tokenized_pairs, cache)
        print("Tokenized pairs saved at " + tokenized_pairs_cache_path)
    return tokenized_pairs

def load_stopwords(stopwords_path): 
    stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='gbk').readlines()]
    print("Stopwords loaded. Number: ", len(stopwords))
    return stopwords 

def tokenize_func(string, stopwords, keep_english_words=False):
    string_tokens = list(jieba.cut(string))
    return_tokens_list = []
    for token in string_tokens:
        if token not in stopwords:
            if token != "\t" and token != " " and (not re.match(r"^\d+\.\d+$",token)) \
                                and (not token.encode("utf-8").isalnum()): # 把英文单词也去掉
                return_tokens_list.append(token)
    return return_tokens_list

def split_and_save_data(all_pairs, output_path, data_name):
    data = {}
    prop_train = 0.97
    prop_valid = 0.02
    num_train_pairs = int(prop_train * len(all_pairs))
    num_valid_pairs = int(prop_valid * len(all_pairs))
    data["train"] = all_pairs[:num_train_pairs]
    data["valid"] = all_pairs[num_train_pairs : num_train_pairs + num_valid_pairs]
    data["test"] = all_pairs[num_train_pairs + num_valid_pairs:]
    
    output_path = os.path.join(output_path, data_name)
    for category in ["train", "valid", "test"]:
        data_path = output_path + "_%s_tokenized_pairs.pkl" % category
        if os.path.exists(data_path):
            print("%s data has existed." %category)
        else:
            with open(data_path, "wb") as cache:
                pickle.dump(data[category], cache)
                print("Tokenized %s pairs saved at %s, data size: %d." %(category, data_path, len(data[category])))
    return data["train"], data["valid"], data["test"]

def add_oov(tokens, word2id, vocab_size, max_oov_words):
    tokens_oov = []
    oov_dict = {}
    for token in tokens:
        if token in word2id and word2id[token] < vocab_size:
            tokens_oov.append(word2id[token])
        else:
            if len(oov_dict) < max_oov_words:
                oov_word_id = oov_dict.get(token, vocab_size+len(oov_dict))
                # 上面的oov_dict.get()函数的第二个参数表示默认值，如果oov_dict中没有w这个词的话就返回从vocab_size向后的数字
                oov_dict[token] = oov_word_id
                tokens_oov.append(oov_word_id)
            else:
                tokens_oov.append(word2id[UNK])
    oov_list = [word for word, word_id in sorted(oov_dict.items(), key=lambda word_id_pair:word_id_pair[1])]
    return tokens_oov, oov_dict, oov_list

def process_data(pairs, word2id, id2word, opts, mode='one2one', include_original=False):
    return_data = []
    count_oov_in_keyword = 0
    max_oov_num_in_text = 0
    max_oov_text = ""
    for idx, (text_tokens, keywords_tokens) in enumerate(pairs):
        # 先处理text的部分
        text_id_in_vocab = [word2id[word] if word in word2id and word2id[word] < opts.vocab_size else word2id[UNK] for word in text_tokens]
        text_id_oov, oov_dict, oov_list = add_oov(text_tokens, word2id, opts.vocab_size, opts.max_oov_words)
        one2one_list = []
        oov_in_keyword = False
        if len(keywords_tokens) == 0 or sum([len(keyword_tokens) for keyword_tokens in keywords_tokens]) == 0:
            continue # 如果关键词个数为0就直接跳过，像这种：[]，或者这种：[[],[],[]]
        for kw_tokens in keywords_tokens:
            one2one = {}
            if include_original:
                one2one["text_tokens"] = text_tokens
                one2one["keyword_tokens"] = kw_tokens
            one2one["text_id"] = text_id_in_vocab
            one2one["text_id_oov"] = text_id_oov
            one2one['oov_dict'] = oov_dict
            one2one['oov_list'] = oov_list
            if len(oov_list) > max_oov_num_in_text:
                max_oov_num_in_text = len(oov_list)
                max_oov_text = text_tokens
            # 然后处理keyword的部分
            kw_id_in_vocab = [word2id[kw] if kw in word2id and word2id[kw] < opts.vocab_size else word2id[UNK] for kw in kw_tokens]
            one2one["keyword_id"] = kw_id_in_vocab
            
            kw_oov = []
            for kw in kw_tokens:
                if kw in word2id and word2id[kw] < opts.vocab_size:
                    kw_id = word2id[kw]
                elif kw in oov_dict:
                    kw_id = oov_dict[kw]
                else:
                    kw_id = word2id[UNK]
                kw_oov.append(kw_id)
            one2one['keyword_id_oov'] = kw_oov
            if any([kw >= opts.vocab_size for kw in kw_oov]):
                oov_in_keyword = True # 在关键词组中发现了oov的标识符
            one2one_list.append(one2one)
        if oov_in_keyword:
            count_oov_in_keyword += 1
        if mode == "one2one":
            return_data.extend(one2one_list)
        elif mode == "one2many":
            one2many = {}
            if include_original:
                one2many["text_tokens"] = text_tokens
                one2many["keywords_tokens"] = keywords_tokens # 注意这里和one2one的区别
            one2many['text_id'] = text_id_in_vocab
            one2many['text_id_oov'] = text_id_oov
            one2many['oov_dict'] = oov_dict
            one2many['oov_list'] = oov_list
            one2many['keywords_id'] = [one2one['keyword_id'] for one2one in one2one_list]
            one2many['keywords_id_oov'] = [one2one['keyword_id_oov'] for one2one in one2one_list]
            return_data.append(one2many)
    print('Number of texts that have oov in keyword(s) / all texts = %d / %d' % (count_oov_in_keyword, len(return_data)))
    print('Max number of oov words in a text: %d' % (max_oov_num_in_text))
    print('Corresponding text with max oov words: %s' % str(max_oov_text))
    print('Number of input pairs / return %s data = %d / %d' % (mode, len(pairs), len(return_data)))
    return return_data

def build_and_export_dataset(pairs, word2id, id2word, opts, output_path, data_name, data_type):
    assert data_type in ["train", "valid", "test"]
    print("Processing %s data, number: %d" % (data_type, len(pairs)))
    if data_type == "train":
        include_original = True
    else:
        include_original = True    
    one2one_data = process_data(pairs, word2id, id2word, opts, mode="one2one", include_original=include_original)
    torch.save(one2one_data, open(os.path.join(output_path, "%s.%s.one2one.pth" %(data_name, data_type)), 'wb'))
    print("One2one data saved at ", os.path.join(output_path, "%s.%s.one2one.pth" %(data_name, data_type)))
    del one2one_data
    one2many_data = process_data(pairs, word2id, id2word, opts, mode="one2many", include_original=include_original)
    torch.save(one2many_data, open(os.path.join(output_path, '%s.%s.one2many.pth' % (data_name, data_type)), 'wb'))
    print("One2many data saved at ", os.path.join(output_path, '%s.%s.one2many.pth' % (data_name, data_type)))
    del one2many_data
    print("Successfully building and exporting dataset!")
    print("-" * 100)


if __name__ == "__main__":
    src_data_path = "../raw_data"
    src_data_name = "doctor"
    inp_fields = ["title", "abstract"]
    trg_fields = ["keyword"]
    src_data = os.path.join(src_data_path, src_data_name)
    pairs = load_pairs(src_data, inp_fields, trg_fields,stop_line_for_test=100) # For test.
    del pairs