#include "ANN.h"
#include <random>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

ANN::ANN() {

	inputLayer = new float[INPUT_SIZE];
	inputBias = new float[HIDDEN_LAYER1_SIZE];
	inputBiasError = new float[HIDDEN_LAYER1_SIZE];
	for (int i = 0; i < INPUT_SIZE; i++) {
		inputLayer[i] = 0;
	}

	inputWeights = new float*[HIDDEN_LAYER1_SIZE];
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		inputWeights[i] = new float[INPUT_SIZE];
	}
	hiddenLayer1 = new float[HIDDEN_LAYER1_SIZE];
	hiddenLayer1Bias = new float[HIDDEN_LAYER1_SIZE];
	hiddenLayer1BiasError = new float[HIDDEN_LAYER1_SIZE];
	hiddenLayer1Weights = new float*[HIDDEN_LAYER2_SIZE];
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		hiddenLayer1Weights[i] = new float[HIDDEN_LAYER1_SIZE];
	}

	hiddenLayer2 = new float[HIDDEN_LAYER2_SIZE];
	hiddenLayer2Bias = new float[OUTPUT_SIZE];
	hiddenLayer2BiasError = new float[OUTPUT_SIZE];
	hiddenLayer2Weights = new float*[OUTPUT_SIZE];
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		hiddenLayer2Weights[i] = new float[HIDDEN_LAYER2_SIZE];
	}

	outputLayer = new float[OUTPUT_SIZE];
	totalCost = 0;

	outputError = new float[OUTPUT_SIZE];
	hiddenLayer2Error = new float[HIDDEN_LAYER2_SIZE];
	hiddenLayer1Error = new float[HIDDEN_LAYER1_SIZE];

}

void ANN::generateRadomWeights() {

	srand(time(NULL));

	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			inputWeights[i][j] = ((float)rand()) / RAND_MAX * 2 - 1;
		}
	}

	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
			hiddenLayer1Weights[i][j] = ((float)rand()) / RAND_MAX * 2 - 1;
		}
	}

	for (int i = 0; i < OUTPUT_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			hiddenLayer2Weights[i][j] = ((float)rand()) / RAND_MAX * 2 - 1;
		}
	}

}

void ANN::generateRandomBias() {

	srand(time(NULL));

	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) 
		inputBias[i] = ((float)rand()) / RAND_MAX * 2 - 1;
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++)
		hiddenLayer1Bias[i] = ((float)rand()) / RAND_MAX * 2 - 1;
	for (int i = 0; i < OUTPUT_SIZE; i++)
		hiddenLayer2Bias[i] = ((float)rand()) / RAND_MAX * 2 - 1;
}

void ANN::createImages() {

	std::ifstream mnistDataSet;
	mnistDataSet.open("mnist_train.csv");
	std::string line;
	std::string tempString;
	int count;
	std::string time;
	float timeCount = 0;

	for (int j = 0; j < DATASET_SIZE; j++) {
		count = 0;
		getline(mnistDataSet, line, ',');
		imageDataSet.label[j] = stoi(line);
		getline(mnistDataSet, line, '\n');
		imageDataSet.pixleMap[j][0] = 0;
		for (int i = 1; i < line.size(); i++) {
			if (line[i] != ',') {
				tempString.push_back(line[i]);
				count++;
			}
			else {
				imageDataSet.pixleMap[j][i - count] = strtof(tempString.c_str(), 0) / (float)255;
				tempString.clear();
			}
		}
		timeCount++;
		if ((int)timeCount % 600 == 0) std::cout << "Creating Images: " << (int)((timeCount / (float)DATASET_SIZE) * 100) << "% complete\n";
	}
}

void ANN::createTestImages() {

	std::ifstream mnistDataSet;
	mnistDataSet.open("mnist_test.csv");
	std::string line;
	std::string tempString;
	int count;

	for (int j = 0; j < TESTSET_SIZE; j++) {
		count = 0;
		getline(mnistDataSet, line, ',');
		imageDataSet.label[j] = stoi(line);
		getline(mnistDataSet, line, '\n');
		imageDataSet.pixleMap[j][0] = 0;
		for (int i = 1; i < line.size(); i++) {
			if (line[i] != ',') {
				tempString.push_back(line[i]);
				count++;
			}
			else {
				imageDataSet.pixleMap[j][i - count] = strtof(tempString.c_str(), 0) / (float)255;
				tempString.clear();
			}
		}
	}
}

void ANN::createMiniBatch() {

	srand(time(NULL));

	int random[MINI_BATCH_SIZE];
	for (int i = 0; i < MINI_BATCH_SIZE; i++) {
		random[i] = rand() % DATASET_SIZE + 0;
	}

	for (int i = 0; i < MINI_BATCH_TRAINING_CYCLE; i++) {
		for (int j = 0; j < MINI_BATCH_SIZE; j++) {
			imageLable = imageDataSet.label[random[j]];
			for (int k = 0; k < INPUT_SIZE; k++) {
				inputLayer[k] = imageDataSet.pixleMap[random[j]][k];
			}
			train(1);
		}
	}
}

void ANN::testMiniBatch() {

	int s = 0;
	correctAnswers = 0;
	int testSize = 1;

	while (s < testSize) {
		createMiniBatch();
		s++;
		std::cout << s / (float)(testSize/100) << "% complete, " << (float)((float)correctAnswers / (float)(MINI_BATCH_SIZE * MINI_BATCH_TRAINING_CYCLE)) * (float)100 << "% accuracy\n";
	}
}

void ANN::testWholeSet(float &slr) {

	int s = 0;
	correctAnswers = 0;

	for (int i = 0; i < DATASET_SIZE; i++) {
		imageLable = imageDataSet.label[i];
		for (int k = 0; k < INPUT_SIZE; k++) {
			inputLayer[k] = imageDataSet.pixleMap[i][k];
		}
		train(slr);
		s++;
		if (s % 600  == 0) std::cout << s / (DATASET_SIZE / 100) << "% complete, " << ((float)correctAnswers/(float)s) * 100 << "% accuracy\n";
		//if (s % 500 == 0) slr *= 0.9;
	}
	slr *= 0.8;
}


void ANN::feedForward()  {

	float temp;

	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		temp = 0;
		for (int j = 0; j < INPUT_SIZE; j++) {
			if (inputLayer[j] != 0) temp += inputWeights[i][j] * inputLayer[j];
		}
		hiddenLayer1[i] = temp + inputBias[i];
		//std::cout << hiddenLayer1[i] << "\n";
	}

	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		temp = 0;
		for (int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
			if (hiddenLayer1[j] != 0) temp += hiddenLayer1Weights[i][j] * RELU(hiddenLayer1[j]);
		}
		hiddenLayer2[i] = temp + hiddenLayer1Bias[i];
		//std::cout << hiddenLayer2[i] << "\n";
		//std::cout << hiddenLayer1Bias [i] << "\n";
	}
	
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		temp = 0;
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			if (hiddenLayer2[j] != 0) temp += hiddenLayer2Weights[i][j] * RELU(hiddenLayer2[j]);
		}
		outputLayer[i] = temp + hiddenLayer2Bias[i];
		//std::cout << outputLayer[i] << "\n";
	}
	softMax();
}

float ANN::sigmoid(float x) {

	return 1 / (1 + exp(-x));
}

float ANN::sigmoidPrime(float x) {

	return sigmoid(x) * (1 - sigmoid(x));
}

float ANN::tanh(float x) {

	return (2 / (1 + exp(-2 * x))) - 1;
}

float ANN::tanhPrime(float x) {

	return 1 - (x * x);
}

float ANN::RELU(float x) {

	if (x < 0) {
		return 0;
	}
	else {
		return x;
	}
}

float ANN::RELUPrime(float x) {

	if (x < 0) {
		return 0;
	}
	else {
		return 1;
	}
}

void ANN::softMax() {

	float outputSum = 0;

	for (int i = 0; i < OUTPUT_SIZE; i++) {
		outputSum += exp(outputLayer[i]);
	}

	for (int i = 0; i < OUTPUT_SIZE; i++) {
		outputLayer[i] = exp(outputLayer[i]) / outputSum;
	}
}

float ANN::crossEntropy(float x, float y) {

	if (y == 1) {
		return -log(x);
	}
	else {
		return -log(1 - x);
	}
}


int ANN::computersChoice() {

	int bestGuessIndex = 0;
	float bestGuessValue = 0;

	for (int i = 0; i < OUTPUT_SIZE; i++) {
		if (outputLayer[i] > bestGuessValue) {
			bestGuessValue = outputLayer[i];
			bestGuessIndex = i;
		}
	}
	return bestGuessIndex;
}

bool ANN::findCost() {

	for (int i = 0; i < OUTPUT_SIZE; i++) {
		correctAnswer[i] = 0;
	}
	correctAnswer[imageLable] = 1;

	totalCost = 0;
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		totalCost += 0.5 * pow(outputLayer[i] - correctAnswer[i], 2);
	}
	//std::cout << "Total cost: " << totalCost << ", Computer's guess: " << computersChoice() << ", Correct answer: " << imageLable << "\n";
	//std::cout << "Total cost: " << totalCost << "\n";
	if (computersChoice() == imageLable) {
		correctAnswers++;
		return true;
	}
	else
		return false;
}

void ANN::backpropagation(float slr) {

	// bias derivative calculation
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		hiddenLayer2BiasError[i] = (outputLayer[i] - correctAnswer[i]) * RELUPrime(outputLayer[i]);
	}
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		hiddenLayer1BiasError[i] = 0;
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			hiddenLayer1BiasError[i] += hiddenLayer2BiasError[j] * RELUPrime(hiddenLayer2[i]);
		}
	}
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		inputBiasError[i] = 0;
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			inputBiasError[i] += hiddenLayer1BiasError[j] * RELUPrime(hiddenLayer1[i]);
		}
	}
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		hiddenLayer2Bias[i] -= LEARNING_RATE * slr * hiddenLayer2BiasError[i];
	}
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		hiddenLayer1Bias[i] -= LEARNING_RATE * slr * hiddenLayer1BiasError[i];
	}
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		inputBias[i] -= LEARNING_RATE * slr * inputBiasError[i];
	}

	// weights derivative calculation
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		outputError[i] = (outputLayer[i] - correctAnswer[i]) * RELUPrime(outputLayer[i]);
	}
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		hiddenLayer2Error[i] = 0;
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			hiddenLayer2Error[i] += outputError[j] * hiddenLayer2Weights[j][i] * RELUPrime(hiddenLayer2[i]);
		}
	}
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		hiddenLayer1Error[i] = 0;
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			hiddenLayer1Error[i] += hiddenLayer2Error[j] * hiddenLayer1Weights[j][i] * RELUPrime(hiddenLayer1[i]);
		}
	}

	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			hiddenLayer2Weights[j][i] -= LEARNING_RATE * slr * outputError[j] * RELU(hiddenLayer2[i]);
		}
	}
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			hiddenLayer1Weights[j][i] -= LEARNING_RATE * slr * hiddenLayer2Error[j] * RELU(hiddenLayer1[i]);
		}
	}
	for (int i = 0; i < INPUT_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
			inputWeights[j][i] -= LEARNING_RATE * slr * hiddenLayer1Error[j] * RELU(inputLayer[i]);
		}
	}

}

void ANN::freeMemory() {

	delete inputLayer;
	delete inputBias;
	delete inputBiasError;
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		delete inputWeights[i];
	}
	delete inputWeights;

	delete hiddenLayer1;
	delete hiddenLayer1Bias;
	delete hiddenLayer1BiasError;
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		delete hiddenLayer1Weights[i];
	}
	delete hiddenLayer1Weights;

	delete hiddenLayer2;
	delete hiddenLayer2Bias;
	delete hiddenLayer2BiasError;
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		delete hiddenLayer2Weights[i];
	}
	delete hiddenLayer2Weights;

	delete outputLayer;

	delete outputError;
	delete hiddenLayer2Error;
	delete hiddenLayer1Error;
}

void ANN::train(float slr) {

	feedForward();
	findCost();
	backpropagation(slr);
}
