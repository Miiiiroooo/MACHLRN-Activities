#pragma once


class SLAlgorithm
{
public:
	SLAlgorithm();

	String GetName();
	float GetAccuracy();
	float GetRecall();
	float GetPrecision();
	float GetFMeasure();
	float GetSensitivity();
	float GetSpecificity();

	void PrintOutConfusionMatrix(String directory, String message, std::ios::openmode mode);
	void PrintPredictions(const List<Sample>& unknownSet, const List<int>& results);
	
	virtual void Reset();

	virtual void PerformTraining(const List<Sample>& trainSet) = 0;
	virtual void PerformTest(const List<Sample>& testSet) = 0;
	virtual void PerformPredictions(const List<Sample>& unknownSet) = 0;

protected:
	void ResetMatrix(Matrix<int>& matrix);
	void CalculatePerformance();


protected:
	bool canPerformCalculations;
	String name;

	Matrix<int> confusionMatrix;
	float accuracy;
	float recall;
	float precision;
	float f_measure;
	float sensitivity;
	float specificity;
};

