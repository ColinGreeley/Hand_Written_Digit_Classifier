#pragma once
#include <iostream>

#define INPUT_SIZE 784
#define HIDDEN_LAYER1_SIZE 32
#define HIDDEN_LAYER2_SIZE 32
#define OUTPUT_SIZE 10

#define MINI_BATCH_SIZE 50
#define MINI_BATCH_TRAINING_CYCLE 100

#define DATASET_SIZE 60000
#define TESTSET_SIZE 500

//#define LEARNING_RATE 0.01
#define LEARNING_RATE 0.0005

struct Images {

	int *label;
	float **pixleMap;

	Images() {
		label = new int[DATASET_SIZE];
		pixleMap = new float*[DATASET_SIZE];
		for (int i = 0; i < DATASET_SIZE; i++) {
			pixleMap[i] = new float[INPUT_SIZE];
		}
	}
	void freeMemory() {
		
		delete label;
		for (int i = 0; i < DATASET_SIZE; i++) {
			delete pixleMap[i];
		}
		delete pixleMap;
	}
};

class ANN {

public:
	ANN();
	Images imageDataSet;
	void setInputLayer(float inputValue, int index) { inputLayer[index] = inputValue; }
	void setImageLable(float newImageLable) { imageLable = newImageLable; }

	void generateRadomWeights();
	void generateRandomBias();
	void createImages();
	void createTestImages();
	void createMiniBatch();
	void testMiniBatch();
	void testWholeSet(float &slr);

	void feedForward();
	float sigmoid(float x);
	float sigmoidPrime(float x);
	float tanh(float x);
	float tanhPrime(float x);
	float RELU(float x);
	float RELUPrime(float x);
	void softMax();
	float crossEntropy(float x, float y);
	int computersChoice();
	bool findCost();
	void backpropagation(float slr);

	void saveNetworkValues();
	void loadNetworkValues();
	void freeMemory();

	void train(float slr);

	float getInputLayer(int i) { return inputLayer[i]; }
	float getHiddenLayer1(int i) { return hiddenLayer1[i]; }
	float getHiddenLayer2(int i) { return hiddenLayer2[i]; }
	float getOutputLayer(int i) { return outputLayer[i]; }

	float getInputWeights(int i, int j) { return inputWeights[i][j]; }
	float getHiddenLayer1Weights(int i, int j) { return hiddenLayer1Weights[i][j]; }
	float getHiddenLayer2Weights(int i, int j) { return hiddenLayer2Weights[i][j]; }

	int getCorrectAnswers() { return correctAnswers; }

private:
	float *inputLayer;
	float **inputWeights;
	float *inputBias;

	float *hiddenLayer1;
	float **hiddenLayer1Weights;
	float *hiddenLayer1Bias;

	float *hiddenLayer2;
	float **hiddenLayer2Weights;
	float *hiddenLayer2Bias;

	float *outputLayer;
	float correctAnswer[OUTPUT_SIZE];
	float totalCost;
	
	float *outputError;
	float *hiddenLayer2Error;
	float *hiddenLayer1Error;
	float *inputBiasError;
	float *hiddenLayer1BiasError;
	float *hiddenLayer2BiasError;

	int imageLable;
	int correctAnswers = 0;
};