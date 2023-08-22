#include "pch.h"
#include "CSVHandler.h"


CSVHandler::CSVHandler()
{

}

bool CSVHandler::RetrieveDataFromDirectory(std::string directory, int numFeatures)
{
	std::ifstream inputFile;

	inputFile.open(directory, std::ios::in);
	if (inputFile)
	{
		HandleRawInputsFromFile(inputFile, numFeatures);
		return true;
	}
	else
	{
		std::cout << "Cannot open file at: " << directory << "\n";
		return false;
	}
}

Matrix<float> CSVHandler::GetSamples()
{
	return samples;
}

void CSVHandler::HandleRawInputsFromFile(std::ifstream& inputFile, int numFeatures)
{
	std::string line;
	std::getline(inputFile, line); // skip first line

	while (std::getline(inputFile, line))
	{
		std::istringstream str(line);
		std::string input;

		std::vector<float> featuresList;
		int column = 0;

		while (std::getline(str, input, ','))
		{
			featuresList.push_back(std::stof(input));
			column++; 
			
			if (column == numFeatures)
			{
				samples.push_back(featuresList);
			}
		}
	}
}
