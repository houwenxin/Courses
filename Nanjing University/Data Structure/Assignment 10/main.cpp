# include "SoftmaxRegression.h"

int main() {
	const char *ImagefileTrain = "./train-images.idx3-ubyte";
	const char *LabelfileTrain = "./train-labels.idx1-ubyte";
	const char *ImagefileTest = "./t10k-images.idx3-ubyte";
	const char *LabelfileTest = "./t10k-labels.idx1-ubyte";

	dataInput dataTrain;
	dataTrain.openImageFile(ImagefileTrain);
	dataTrain.openLabelFile(LabelfileTrain);

	int numClass, numFeature;
	int epoch = 10;
	float LearningRate = 5e-5;

	numClass = 10;
	numFeature = dataTrain.imageLength();
	SoftmaxRegression DigitRecognizer(numClass, numFeature);
	// This is for pretrained model.
	DigitRecognizer.LoadModel("DigitRecognizer.txt");
	DigitRecognizer.Train(dataTrain, epoch, LearningRate);

	dataInput dataTest;
	dataTest.openImageFile(ImagefileTest);
	dataTest.openLabelFile(LabelfileTest);
	DigitRecognizer.Accuracy(dataTest);

	DigitRecognizer.SaveModel("DigitRecognizer.txt");
	dataTest.reset();
	char *Image = new char[dataTest.imageLength()];
	double *image = new double[dataTest.imageLength()];
	int Label = 0, Index = 1;
	
	cout << "Now get into free test: " << endl;
	while (1) {
		cout << "Enter an index you want to predict, Index >= 1: " << endl;
		cin >> Index;
		if (Index == 0) break;
		else if (Index > dataTest.numImage()) {
			cout << "Index should not be larger than " << dataTest.numImage() << endl;
			continue;
		}
		for (int i = 0; i < Index; i++) {
			dataTest.read(&Label, Image);
		}
		DigitRecognizer.PreProcessMNIST((unsigned char *)Image, image, dataTest.imageLength());
		int y_predict;
		DigitRecognizer.Predict(image, y_predict);
		cout << "Label: " << Label << ", Prediction: " << y_predict << endl;
	}
	system("pause");
	delete[] Image; Image = NULL;
	delete[] image; image = NULL;
	return 0;
}