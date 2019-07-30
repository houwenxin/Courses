#pragma warning(disable:4996) 

# ifndef __SOFTMAXREGRESSION_H__
# define __SOFTMAXREGRESSION_H__

# include <iostream>
# include "data_input.h"

#define OK 1
typedef int Status;

class SoftmaxRegression {
public:
	SoftmaxRegression(int _numClass, int _numFeature):numClass(_numClass), numFeature(_numFeature) {
		// Weight Matrix: numClass * numFeature
		Weight = new double*[numClass];
		for (int i = 0; i < numClass; i++) {
			Weight[i] = new double[numFeature];
		}
		// Initialize weights.
		for (int i = 0; i < numClass; i++)
			for (int j = 0; j < numFeature; j++)
				Weight[i][j] = 0.01;
	}
	~SoftmaxRegression() {
		numClass = numFeature = 0;
		for (int i = 0; i < numClass; i++) {
			delete[] Weight[i];
			Weight[i] = NULL;
		}
		delete[] Weight;
		Weight = NULL;
	}
	void Train(dataInput &TrainData, int epoch = 20, float LearningRate = 0.5);
	void Accuracy(dataInput &TestData);
	void Predict(double *x, int &y_predict);
	void PreProcessMNIST(const unsigned char *InputImage, double *OutputImage, int Size);
	void SaveModel(const char *ModelName);
	void LoadModel(const char *ModelName);
protected:
	Status loadData(dataInput &InputData, double **OutputDataMatrix, int **OutputLabelMatrix, bool randPerm = true);
	void OneHotLabel(int label, int *LabelOneHot);
	void softmax(double *x);
	inline int argmax(double *ProbOneHot) {
		double max = 0, index = 0;
		for (int i = 0; i < numClass; i++) {
			if (max < ProbOneHot[i]) {
				max = ProbOneHot[i];
				index = i;
			}
		}
		return index;
	}
	inline int argmax(int *LabelOneHot) {
		int label = 0;
		for (int i = 0; i < numClass; i++)
			if (LabelOneHot[i] == 1)
				label = i;
		return label;
	}
private:
	int numClass;
	int numFeature;
	double **Weight;
};

#endif