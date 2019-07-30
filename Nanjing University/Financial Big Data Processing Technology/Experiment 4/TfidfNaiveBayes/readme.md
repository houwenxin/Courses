## MapReduce Naive Bayes for Sentiment Classification  
  
### Usage:  
  
hadoop jar NaiveBayes.jar NaiveBayesMain \<input path\> \<output path\> \<mode (train / test / validate)\>  
  
Example:  
  
Train: hadoop jar NaiveBayes.jar traindata model train  
  
Test: hadoop jar NaiveBayes.jar testdata prediction\_NaiveBayes test  
  
* Before testing, model should first be trained and put into HDFS path.  
* Before testing, info.txt generated by Prepocess.py should be put into HDFS path.  
  
### Input File Format Example:  
  
Train Data Format:  
index, label, feature1:weight1, feature2:weight2, featurek:weightk, ...  
  
Test Data Format:  
index, label, feature1:count1, featurek:countk, ...  
  
* Train Data are TF-IDF features.  
* Test Data are count of every train data's TF-IDF feature.  
  
### Output File Format Example:  
Model File:  
TrainLabel#Feature:Weight  
    
Prediction File:  
\<Index of test or train data\> [tab] predictedLabel
