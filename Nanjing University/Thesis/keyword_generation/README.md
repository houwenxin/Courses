# keyword_generation
A sequence-to-sequence model incorporating attention and copy mechanisms for Chinese keywords generation and extraction.  

This project is for my graduation thesis at Nanjing University.  


### Dataset:  

**CNKI Ph.D. dissertation dataset**: http://pan.baidu.com/s/1miGoNPY.  

**Acknowledgement**: https://github.com/roliygu/CNKICrawler  

### Execution:
1. Download raw data and put into `./raw/` from http://pan.baidu.com/s/1miGoNPY.  
2. Run `src/preprocess.py`.  
3. Run `src/train.py`, refer to `src/conf.py` for configuration information.  
4. Run `src/predict.py` for evaluation.  



### Figures:

1. Training data information after preprocessing:

![train_info](https://houwenxin.github.io/files/thesis/train_info.jpg)

2. Validation data information after preprocessing:

![train_info](https://houwenxin.github.io/files/thesis/valid_info.jpg)

3. Test data information after preprocessing:

![train_info](https://houwenxin.github.io/files/thesis/test_info.jpg)

4. Training loss and validation curves

![training_curves](https://houwenxin.github.io/files/thesis/[epoch=5,batch=6868,total_batch=56000]train_valid_test_curve.png.png)



------------

***The project is modified based on https://github.com/memray/seq2seq-keyphrase-pytorch***  