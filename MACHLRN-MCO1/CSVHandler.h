#pragma once


class CSVHandler
{
public:
	CSVHandler();

	bool RetrieveDataFromDirectory(String directory);
	Matrix<int> GetUnknownSet();
	Matrix<int> GetKnownSet();

	bool PrintOutMessage(String directory, String message, std::ios::openmode mode);

private:
	void HandleRawInputsFromFile(std::ifstream& inputFile);


private:
	Matrix<int> unknownSet;
	Matrix<int> knownSet;
};

