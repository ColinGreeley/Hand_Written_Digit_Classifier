#include <iostream>
#include "ANN.h"
#include "Window.h"

// Colin Greeley
// First ANN application with MNIST dataset
// 10/10/2018

int main(void) {

	std::string answer;

	while (answer != "test" && answer != "train") {
		std::cout << "'train' or 'test'\n";
		std::cin >> answer;
	}
	if (answer == "train") {
		//*
		int s = 0;
		ANN a;
		float stochasticLearningRate = 1;
		a.loadNetworkValues();
		a.createImages();
		//a.generateRadomWeights();
		//a.generateRandomBias();
		//while (s < 100) {
			//a.testMiniBatch();
			//s++;
		//}
		while (s < 10) {
			a.testWholeSet(stochasticLearningRate);
			s++;
		}
		a.saveNetworkValues();
		a.freeMemory();
		a.imageDataSet.freeMemory();
		//*/
	}
	if (answer == "test") {
		renderWindow();
	}
	return 0;
}