#include "pch.h"
#include "SLProgram.h"
#include "SLAlgorithm.h"
#include "Perceptrons.h"

int main()
{
	srand((unsigned)time(0));

	SLProgram program = SLProgram();
	String directory = "files/InternetSurveyDataset.csv" ;
	float testSplit = 0.1;
	int repetitions = 5;
	int maxIterations = 50000;
	float learningRate = 0.2f;
	int metric = 3;
	
	do
	{
		std::cout << "Please enter the directory of the csv file: ";
		std::cin >> directory;
		std::cout << "Please enter the test split from the known set: ";
		std::cin >> testSplit; 
		std::cout << "Please enter the number of repetitions for the Repeated K-Fold Cross Validation: ";
		std::cin >> repetitions; 
		std::cout << "Please enter the maximum number of iterations for the perceptrons: ";
		std::cin >> maxIterations; 
		std::cout << "Please enter the learning rate of the perceptrons: ";
		std::cin >> learningRate; 
		do
		{
			std::cout << "Please enter the score metric you want to evaluate:\n";
			std::cout << "[1] - Accuracy\n[2] - Sensitivity\n[3] - Specificity\n[4] - Precision\n[5] - Recall\n[6] - F-Measure\nResponse: ";
			std::cin >> metric;
		} while (metric < 1 || metric > 6);
	} while (!program.Initialize(directory, testSplit, repetitions, maxIterations, learningRate, metric)); 

	program.RunProgram();

	return 0;
}