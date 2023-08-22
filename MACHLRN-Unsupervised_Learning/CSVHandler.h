#pragma once

class CSVHandler
{
public:
	CSVHandler();

	bool RetrieveDataFromDirectory(std::string directory, int numFeatures);
	Matrix<float> GetSamples();

private:
	void HandleRawInputsFromFile(std::ifstream& inputFile, int numFeatures);


private:
	Matrix<float> samples;
};

