# coding: utf-8

import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.externals import joblib

filePath = "./"
stopWordPath = os.path.join(filePath, "stopwords.txt")

def loadStopWords(filePath):
    stopwords = [line.strip() for line in open(filePath, 'r').readlines()]
    return stopwords

def readFile(filePath, stopWordPath):
    document = ""
    stopwords = loadStopWords(stopWordPath)
    with open(filePath, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            line_words = list(jieba.cut(line))
            clean_words = ""
            for word in line_words:
                if word not in stopwords:
                    if word !=  '\t':
                        clean_words += word
                        clean_words += " "
            document = document + clean_words + "\n"
    return document
        
        
def readFiles(path):
    all_documents = []
    file_list = []
    files = os.listdir(path)
    for file in files:
        filePath = os.path.join(path, file)
        if os.path.isfile(filePath):
            if os.path.splitext(filePath)[1] == ".txt":
                print("Reading file: ", file)
                file_list.append(file)
                document = readFile(filePath, stopWordPath)
                all_documents.append(document)
                pass
    return file_list, all_documents

def writeInfo(filePath, neg_file_num, pos_file_num, neu_file_num):
    nums = np.array([neg_file_num, pos_file_num, neu_file_num])
    with open(filePath, 'w') as file:
        for i, name in enumerate(['neg_file_num', 'pos_file_num', 'neu_file_num']):
            # 中间插入i-1是为了表明类别，-1:负向，0:中立，1:正向
            file.write(name + ":" +str(i - 1) + ":" + str(nums[i]) + "\n")
    
def writeData(filepath, tfidf_array, neg_file_num, pos_file_num, neu_file_num):
    nonzero = np.nonzero(tfidf_array)
    all_elements = tfidf_array[nonzero]
    zipped_nonzero = list(zip(list(nonzero[0]), list(nonzero[1])))
    with open(filepath, 'w') as file:
        #file.write("neg_file_num:" + str(neg_file_num) + "\n")
        #file.write("pos_file_num:" + str(pos_file_num) + "\n")
        #file.write("neu_file_num:" + str(neu_file_num) + "\n")
        i = 0
        previous_idx = -1
        for row in zipped_nonzero:
            #file.write(" ".join(map(str, row)))
            # 首先拿到行数
            idx = row[0]
            # 判断是否和上一个行数一致，不一致则是新的一行
            if(idx != previous_idx):
                if(idx < neg_file_num):
                    flag = -1
                elif(idx >= neg_file_num and idx < neg_file_num + pos_file_num):
                    flag = 1
                elif(idx >= neg_file_num + pos_file_num):
                    flag = 0
                if idx == 0:
                    file.write(str(idx) + ", " + str(flag))
                else:
                    file.write("\n" + str(idx) + ", " + str(flag))
            #正常写入数据
            file.write(", " + str(row[1]))
            file.write(":" + str(all_elements[i]))
            i += 1
            previous_idx = idx
            
def processTrainData(train_file_path, feature_num, model_type):
    negative_path = os.path.join(train_file_path, "negative")
    positive_path = os.path.join(train_file_path, "positive")
    neutral_path = os.path.join(train_file_path, "neutral")
    
    all_documents = []
    neg_file_list, neg_documents = readFiles(negative_path)
    neg_file_num = len(neg_file_list)
    pos_file_list, pos_documents = readFiles(positive_path)
    pos_file_num = len(pos_file_list)
    neu_file_list, neu_documents = readFiles(neutral_path)
    neu_file_num = len(neu_file_list)
    
    all_documents = all_documents + neg_documents + pos_documents + neu_documents
    #print(file_list)
    #print(all_documents)
    #TfidfVectorizer默认的参数为token_pattern=r"(?u)\b\w\w+\b"，所以会把一个字的词过滤掉，这里得补回来
    if model_type == "count_model":
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.5, max_features=feature_num)
        model = vectorizer.fit(all_documents)
        joblib.dump(model, "count_model.model")
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(model.transform(all_documents))
        print(vectorizer.vocabulary_)
    elif model_type == "tfidf_model":
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.5, max_features=feature_num)
        model = vectorizer.fit(all_documents)
        joblib.dump(model, "tfidf_model.model")
        tfidf = model.transform(all_documents)
        print(vectorizer.vocabulary_)
    tfidf_array = tfidf.toarray()
    print("Train Data Matrix Size: ", tfidf_array.shape)
    #vocab = tfidf_model.vocabulary_
    #vocab = sorted(vocab.items(),key = lambda x:x[1],reverse = False)
    writeData("./traindata.txt", tfidf_array, neg_file_num, pos_file_num, neu_file_num)
    writeInfo("./info.txt", neg_file_num, pos_file_num, neu_file_num)
    
    del tfidf
    del tfidf_array
    return model

def processTestData(model_path, testFilePath, stopWordPath):
    test_file = os.path.join(testFilePath, "fulldata.txt")
    stopwords = loadStopWords(stopWordPath)
    model = joblib.load(model_path)
    print("Model loaded.")
    documents = []
    with open(test_file, "rb") as file:
        for line in file.readlines():
            line = line.strip()
            test_news = line.split("\t".encode("gbk"))[4]
            news_words = list(jieba.cut(test_news))
            clean_words = ""
            for word in news_words:
                if word not in stopwords:
                    clean_words += word
                    clean_words += " "
            documents.append(clean_words)
            
    test_data = model.transform(list(documents))
    test_data_array = test_data.toarray()
    nonzero = np.nonzero(test_data_array)
    all_elements = test_data_array[nonzero]
    zipped_nonzero = list(zip(list(nonzero[0]), list(nonzero[1])))
    print("Start writing test data.") 
    with open("testdata.txt", 'w') as file:
        i = 0
        previous_idx = -1
        for row in zipped_nonzero:
            #file.write(" ".join(map(str, row)))
            # 首先拿到行数
            idx = row[0]
            # 判断是否和上一个行数一致，不一致则是新的一行
            if(idx != previous_idx):
                if idx == 0:
                    file.write(str(idx))
                else:
                    file.write("\n" + str(idx))
                    if(idx - previous_idx > 1):
                        for temp in range(idx - previous_idx - 1):
                            print("Test data (line {}) is missed.".format(previous_idx + temp + 1))
            #正常写入数据
            file.write(", " + str(row[1]))
            file.write(":" + str(all_elements[i]))
            i += 1
            previous_idx = idx
    print("Test data is successfully processed.")
    del test_data, test_data_array
            
if __name__ == '__main__':
    train = 0
    model_type = "count_model"
    if(train == 1):
        model = processTrainData(filePath, feature_num=5000, model_type=model_type)
    else:
        model_name = model_type + ".model"
        model_path = os.path.join(filePath, model_name)
        processTestData(model_path, filePath, stopWordPath)
