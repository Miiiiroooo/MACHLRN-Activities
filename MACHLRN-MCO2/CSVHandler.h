#pragma once


class CSVHandler
{
public:
	CSVHandler();

	bool RetrieveDataFromDirectory(String directory);
	List<Sample> GetUnknownSet();
	List<Sample> GetKnownSet();

	bool PrintOutMessage(String directory, String message, std::ios::openmode mode);

private:
	void HandleRawInputsFromFile(std::ifstream& inputFile);


private:
	List<Sample> unknownSet;
	List<Sample> knownSet;
};

