## MapReduce TextKNN for Sentiment Classification  

### Usage:   
hadoop jar TextKNN.jar KNNMain \<input path\> \<output path\> \<feature dimension\> \<k\>  
  
Example:  
hadoop jar HWX\_TextKNN.jar KNNMain testdata prediction\_KNN 1000 10  
  
  
### Input File Format Example:  
  
Train Data Format:  
index, label, feature1:weight1, feature5:weight5, featurek:weightk, ...  
  
Input Format (test data):  
index, feature1:weight1, feature2:weight2, featurek:weightk, ...  
  
* Both Train Data and Test Data are TF-IDF features.(Sparse Matrix)  

### Output File Format Example:
  
indexOfTestData [tab] predictedLabel
