#include "ANN.h"
#include <iostream>
#include <fstream>
#include <string>

void ANN::saveNetworkValues() {

	std::ofstream outfile;
	outfile.open("NetworkValues.txt");

	outfile << "input weight values\n";
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			outfile << inputWeights[i][j] << "\n";
		}
		outfile << std::endl;
	}

	outfile << "input bias value\n";
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		outfile << inputBias[i] << "\n";
	}
	outfile << std::endl;

	outfile << "hidden layer 1 weight values\n";
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
			outfile << hiddenLayer1Weights[i][j] << "\n";
		}
		outfile << std::endl;
	}

	outfile << "hidden layer 1 bias value\n";
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		outfile << hiddenLayer1Bias[i] << "\n";
	}
	outfile << std::endl;

	outfile << "hidden layer 2 weight values\n";
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			outfile << hiddenLayer2Weights[i][j] << "\n";
		}
		outfile << std::endl;
	}

	outfile << "hidden layer 2 bias value\n";
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		outfile << hiddenLayer2Bias[i] << "\n";
	}
	outfile << std::endl;
}

void ANN::loadNetworkValues() {

	std::ifstream infile;
	std::string tempString;
	infile.open("NetworkValues.txt");

	getline(infile, tempString);
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			getline(infile, tempString);
			inputWeights[i][j] = strtof(tempString.c_str(), 0);
		}
		getline(infile, tempString);
	}

	getline(infile, tempString);
	for (int i = 0; i < HIDDEN_LAYER1_SIZE; i++) {
		getline(infile, tempString);
		inputBias[i] = strtof(tempString.c_str(), 0);
	}

	getline(infile, tempString);

	getline(infile, tempString);
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER1_SIZE; j++) {
			getline(infile, tempString);
			hiddenLayer1Weights[i][j] = strtof(tempString.c_str(), 0);
		}
		getline(infile, tempString);
	}

	getline(infile, tempString);
	for (int i = 0; i < HIDDEN_LAYER2_SIZE; i++) {
		getline(infile, tempString);
		hiddenLayer1Bias[i] = strtof(tempString.c_str(), 0);
	}

	getline(infile, tempString);

	getline(infile, tempString);
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		for (int j = 0; j < HIDDEN_LAYER2_SIZE; j++) {
			getline(infile, tempString);
			hiddenLayer2Weights[i][j] = strtof(tempString.c_str(), 0);
		}
		getline(infile, tempString);
	}

	getline(infile, tempString);
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		getline(infile, tempString);
		hiddenLayer2Bias[i] = strtof(tempString.c_str(), 0);
	}

	getline(infile, tempString);

}