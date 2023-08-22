#pragma once

#include "SLAlgorithm.h"

class Perceptrons : public SLAlgorithm
{
public:
	Perceptrons();

	void InitializePerceptron(int maxIterations, float learningRate); 
	void Reset();

	void PerformTraining(const List<Sample>& trainSet);
	void PerformTest(const List<Sample>& testSet);
	void PerformPredictions(const List<Sample>& unknownSet);

private:
	void RandomizeInitialWeights(int numFeatures);


private:
	float w0;
	List<float> weightsList;

	int maxIterations;
	float learningRate;
};

