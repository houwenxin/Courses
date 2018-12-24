### MapReduce Sentiment Classification  

##### TFIDFNaiveBayes:  
  
Step 1: Run Preprocess.py: python3 Preprocess.py --mode \<train(default) / test\> --model count --feature\_num feature\_num (default: 1000).  
  
Step 2: Put info.txt, traindata.txt, testdata.txt into HDFS.  
  
Step 3: Run NaiveBayesMain train mode to get model.  
  
Step 4: Run NaiveBayesMain test mode to get prediction.  
  
##### TextKNN:  
  
Step 1: Run Preprocess.py: python3 Preprocess.py --mode \<train(default) / test\> --model tfidf --feature\_num feature\_num (default: 1000). 
  
Step 2: Put traindata.txt, textdata.txt into HDFS.  
  
Step 3: Run KNNMain \<input path\> \<output path\> \<featureNum\> \<k\> \<test / validate\> (Set featureNum=featureNum).  
