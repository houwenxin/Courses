﻿// SoftmaxRegression1.cpp: 定义控制台应用程序的入口点。
//
# pragma warning(disable:4996) 

# include "SoftmaxRegression.h"
# include <time.h>

Status SoftmaxRegression::loadData(dataInput &InputData, double **OutputDataMatrix, int **OutputLabelMatrix, bool randPerm) {
	char *Image = new char[InputData.imageLength()];
	double *image = new double[InputData.imageLength()];
	int label = 0;
	int *idx = new int[InputData.numImage()];
	// Set Index in order to do random permutation.
	for (int i = 0; i < InputData.numImage(); i++)
		idx[i] = i;
	if (randPerm == true) {
		int RandIdx = 0, temp = 0;
		srand((int)time(NULL));
		for (int i = 0; i < InputData.numImage(); i++) {
			RandIdx = rand() % InputData.numImage();
			temp = idx[i];
			idx[i] = idx[RandIdx];
			idx[RandIdx] = temp;
		}
	}
	for (int i = 0; i < InputData.numImage(); i++) {
		OutputDataMatrix[idx[i]][0] = 1;
		InputData.read(&label, Image);
		// Convert the char* to double* for training.
		PreProcessMNIST((unsigned char *)Image, image, InputData.imageLength());
		for (int j = 0; j < numFeature; j++) {
			OutputDataMatrix[idx[i]][j + 1] = image[j];
		}
		// Convert label to one-hot matrix for training.
		OneHotLabel(label, OutputLabelMatrix[idx[i]]);
	}
	// Reset input data for next use.
	InputData.reset();
	delete[] idx; idx = NULL;
	delete[] Image; Image = NULL;
	delete[] image; image = NULL;
	return OK;
}
void SoftmaxRegression::PreProcessMNIST(const unsigned char *InputImage, double *OutputImage, int Size) {
	for (int i = 0; i < Size; i++)
		OutputImage[i] = (InputImage[i] >= 128) ? 1.0 : 0.0;
}
void SoftmaxRegression::OneHotLabel(int label, int *LabelOneHot) {
	for (int i = 0; i < numClass; i++) {
		if (label == i)
			LabelOneHot[i] = 1;
		else
			LabelOneHot[i] = 0;
	}
}
void SoftmaxRegression::softmax(double *x) {
	double max = 0;
	double sum = 0;
	for (int i = 0; i < numClass; i++)
		if (max < x[i]) max = x[i];
	for (int i = 0; i < numClass; i++) {
		x[i] = exp(x[i] - max);// -max is to avoid x[i] is to large.
		sum += x[i];
	}
	for (int i = 0; i < numClass; i++)
		x[i] /= sum;
}
void SoftmaxRegression::Train(dataInput &TrainData, int epoch, float LearningRate) {
	double **dataTrain;
	int **labelTrain;
	int numTrain = TrainData.numImage();
	dataTrain = new double *[numTrain];
	labelTrain = new int *[numTrain];
	for (int i = 0; i < numTrain; i++) {
		dataTrain[i] = new double[numFeature + 1];
		labelTrain[i] = new int[numClass];
	}

	cout << "Loading Training Data..." << endl;
	if (loadData(TrainData, dataTrain, labelTrain) == 0) {
		cerr << "[Error]: Training failed" << endl;
		return;
	}
	cout << "Start Training." << endl;
	cout << "Epoch: 0...";
	// Start Training.
	double *PredictProb = new double[numClass];
	for (int i = 0; i < epoch; i++) {
		cout << i+1 << "...";
		for (int j = 0; j < numTrain; j++) {
			for (int k = 0; k < numClass; k++) {
				PredictProb[k] = 0;
				for (int temp = 0; temp < numFeature + 1; temp++) {
					PredictProb[k] += Weight[k][temp]* dataTrain[j][temp];
				}
			}
			softmax(PredictProb);
			// Stochastic Gradient Descent.
			for (int k2 = 0; k2 < numClass; k2++) {
				for (int n = 0; n < numFeature + 1; n++) {
					Weight[k2][n] += LearningRate * (labelTrain[j][k2] - PredictProb[k2]) * dataTrain[j][n];
				}
			}
		}
	}
	cout << endl;
	delete[] PredictProb; PredictProb = NULL;
	for (int i = 0; i < numTrain; i++) {
		delete[] dataTrain[i]; dataTrain[i] = NULL;
		delete[] labelTrain[i]; labelTrain[i] = NULL;
	}
	delete[] dataTrain; dataTrain = NULL;
	delete[] labelTrain; labelTrain = NULL;
	cout << "Training Completed." << endl;
}

void SoftmaxRegression::Accuracy(dataInput &testData) {

	double **dataTest;
	int **labelTest;
	int numTest = testData.numImage();
	dataTest = new double *[numTest];
	labelTest = new int *[numTest];
	for (int i = 0; i < numTest; i++) {
		dataTest[i] = new double[numFeature + 1];
		labelTest[i] = new int[numClass];
	}

	cout << "Loading Test Data..." << endl;
	if (loadData(testData, dataTest, labelTest) == 0) {
		cerr << "[Error]: Test failed" << endl;
		return;
	}

	int PredictLabel = 0;
	int Count = 0;
	for (int i = 0; i < numTest; i++) {
		Predict(dataTest[i], PredictLabel);
		//cout << PredictLabel<<"    "<<argmax(labelTest[i]) << endl;
		if (PredictLabel == argmax(labelTest[i])) {
			Count++;
		}
	}
	double acc = 1.0 * Count / numTest;		// *1.0 to convert to double.
	cout << "Accuarcy: " << acc << endl;
	for (int i = 0; i < numTest; i++) {
		delete[] dataTest[i]; dataTest[i] = NULL;
		delete[] labelTest[i]; labelTest[i] = NULL;
	}
	delete[] dataTest; dataTest = NULL;
	delete[] labelTest; labelTest = NULL;
}
void SoftmaxRegression::Predict(double *x, int &y_predict) {
	double *PredOneHot = new double[numClass];
	for (int i = 0; i < numClass; i++) {
		PredOneHot[i] = 0;
		for (int j = 0; j < numFeature + 1; j++) {
			PredOneHot[i] += Weight[i][j] * x[j];
		}
	}
	softmax(PredOneHot);
	y_predict = argmax(PredOneHot);
	delete[] PredOneHot; PredOneHot = NULL;
}

void SoftmaxRegression::SaveModel(const char *ModelName) {
	ofstream ModelFile;
	ModelFile.open(ModelName);
	for (int i = 0; i < numClass; i++) {
		for (int j = 0; j < numFeature + 1; j++) {
			ModelFile << Weight[i][j];
			if (j != numFeature)
				ModelFile << "	";	// One Tab between two Weights.
		}
		ModelFile << "\r\n";
	}
	ModelFile << endl;
	cout << "Model Saved to " << ModelName << endl;
	ModelFile.close();
}
void SoftmaxRegression::LoadModel(const char *ModelName) {
	fstream ModelFile;
	ModelFile.open(ModelName);
	for (int i = 0; i < numClass; i++) {
		for (int j = 0; j < numFeature + 1; j++) {
			ModelFile >> Weight[i][j];
		}
	}
	cout << "Model Loaded from " << ModelName << endl;
	ModelFile.close();
}
