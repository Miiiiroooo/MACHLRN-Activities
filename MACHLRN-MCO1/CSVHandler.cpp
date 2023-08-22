#include "pch.h"
#include "CSVHandler.h"

CSVHandler::CSVHandler()
{

}

bool CSVHandler::RetrieveDataFromDirectory(String directory)
{
	std::ifstream inputFile;

	inputFile.open(directory, std::ios::in);
	if (inputFile)
	{
		HandleRawInputsFromFile(inputFile);
		return true;
	}
	else
	{
		std::cout << "Cannot open file at: " << directory << "\n";
		return false;
	}
}

Matrix<int> CSVHandler::GetUnknownSet()
{
	return unknownSet;
}

Matrix<int> CSVHandler::GetKnownSet()
{
	return knownSet;
}

bool CSVHandler::PrintOutMessage(String directory, String message, std::ios::openmode mode)
{
	std::ofstream outputFile;

	outputFile.open(directory, mode);
	if (outputFile)
	{
		outputFile << message;
	}
	else
	{
		std::cout << "Cannot open file at: " << directory << "\n";
		return false;
	}
}

void CSVHandler::HandleRawInputsFromFile(std::ifstream& inputFile)
{
	String line;
	std::getline(inputFile, line); // skip first line
	
	while (std::getline(inputFile, line))
	{
		std::istringstream str(line);   // makes the comma-separated string manipulatable
		String input;                   // takes the values from comma-separated string as 'input'
		List<int> inputsList;           // list of all the inputs (aka features)

		int column = 0;                 // keeps track of the current order of the 'column' or feature
		bool isTest = true;             // determine if test set or training set

		while (std::getline(str, input, ','))
		{
			column++;

			if (column < 5)
			{
				continue;
			}
			else if (column == 46)
			{
				isTest = false;
				inputsList.push_back(std::stoi(input));
			}
			else
			{
				inputsList.push_back(std::stoi(input) == 0 ? -1 : 1);
			}
		}

		if (isTest)
		{
			unknownSet.push_back(inputsList);
		}
		else
		{
			knownSet.push_back(inputsList); 
		}
	}
}