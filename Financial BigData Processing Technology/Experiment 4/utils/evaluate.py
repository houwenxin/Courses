# coding: utf-8

import pandas as pd
import argparse

default_train_file = ""
#default_valid_file = "./Results/validation_NaiveBayes_d5000/part-r-00000"
#default_valid_file = "./Results/validation_KNN_d500_k20_Euclidean/part-r-00000"
#default_valid_file = "./Results/validation_KNN_d500_k10_Euclidean/part-r-00000"
#default_valid_file = "./Results/validation_KNN_d500_k5_Euclidean/part-r-00000"
default_valid_file = ""

def read_label(file, delimiter):
    labels = []
    indexes = []
    with open(file) as file:
        for line in file.readlines():
            index = line.strip().split(delimiter)[0]
            label = line.strip().split(delimiter)[1]
            indexes.append(index)
            labels.append(label)
    sorted_labels = pd.Series(labels, index=indexes)
    sorted_labels = sorted_labels.sort_index()
    return sorted_labels

def evaluate(train_file, valid_file):
    predict_label = read_label(valid_file, '\t')
    target_label = read_label(train_file, ', ')
    true_predict = 0
    #print(len(predict_label))
    #print(len(target_label))
    for i in range(len(predict_label)):
        #print(str(target_label[i]) + " : " + str(predict_label[i]))
        if target_label[i] == predict_label[i]:
            true_predict += 1
    accuracy = true_predict / len(predict_label) * 1.0
    return accuracy
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate performance on predictions.")
    parser.add_argument("--trainpath", type=str, default=default_train_file, help="Path of train labels")
    parser.add_argument("--validpath", type=str, default=default_valid_file, help="Path of valid labels")
    args = parser.parse_args()
    
    train_file = args.trainpath
    valid_file = args.validpath
    accuracy = evaluate(train_file, valid_file)
    print(accuracy)
