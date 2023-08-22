#include "pch.h"
#include "KMeansSimulator.h"


int main()
{
	srand(time(0));

	KMeansSimulator simulator;

	int numFeatures = 2;
	int maxIterations = 10;
	std::string directory = "files/test_dataset.csv";

	do
	{
		std::cout << "Please enter the number of features: ";
		std::cin >> numFeatures;
		std::cout << "Please enter the maximum iterations on for each k means: ";
		std::cin >> maxIterations;
		std::cout << "Please enter the directory of the csv file: ";
		std::cin >> directory;

	} while (!simulator.Initialize(numFeatures, maxIterations, directory));

	simulator.PerformClustering();
	simulator.PrintResult();

	return 0;
}