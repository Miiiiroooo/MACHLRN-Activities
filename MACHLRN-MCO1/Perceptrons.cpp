#include "pch.h"
#include "Perceptrons.h"

Perceptrons::Perceptrons()
{
	name = "Perceptrons";

	w0 = 0.f;
	maxIterations = 0;
	learningRate = 0.f;
}

void Perceptrons::InitializePerceptron(int maxIterations, float learningRate)
{
	this->maxIterations = maxIterations;
	this->learningRate = learningRate;

	canPerformCalculations = true;
}

void Perceptrons::Reset()
{
	SLAlgorithm::Reset();

	w0 = 0;
	weightsList.clear();
}

void Perceptrons::PerformTraining(const List<Sample>& trainSet)
{
	if (!canPerformCalculations)
		return;

	Sample temp = trainSet[0];
	RandomizeInitialWeights(temp->size() - 1); 

	for (int i = 0; i < maxIterations; i++)
	{
 		int randIndex = rand() % trainSet.size(); 
		Sample sample = trainSet[randIndex];

		int target = sample->back(); 
		int numFeatures = sample->size() - 1;

		float activation = 0 + w0;
		for (int j = 0; j < numFeatures; j++)
		{
			activation += (*sample)[j] * weightsList[j]; 
		}

		int output = activation > 0 ? 1 : 0;
		int error = target - output;

		w0 += learningRate * error;
		for (int j = 0; j < numFeatures; j++)
		{
			weightsList[j] += learningRate * error * (*sample)[j];
		}
	}
}

void Perceptrons::PerformTest(const List<Sample>& testSet)
{
	if (!canPerformCalculations)
		return;

	for (int i = 0; i < testSet.size(); i++) 
	{
		Sample sample = testSet[i];
		int expected = sample->back();
		int numFeatures = sample->size() - 1;

		float activation = 0 + w0;
		for (int j = 0; j < numFeatures; j++) 
		{
			activation += (*sample)[j] * weightsList[j];
		}

		int output = activation > 0 ? 1 : 0; 

		if (output == expected && expected == 1) 
		{
			confusionMatrix[0][0]++; // TP
		}
		else if (output != expected && expected == 1) 
		{
			confusionMatrix[0][1]++; // FN
		}
		else if (output != expected && expected == 0)
		{
			confusionMatrix[1][0]++; // FP
		}
		else if (output == expected && expected == 0) 
		{
			confusionMatrix[1][1]++; // TN
		}
	}

	CalculatePerformance();
}

void Perceptrons::PerformPredictions(const List<Sample>& unknownSet)
{
	if (!canPerformCalculations) 
		return; 

	List<int> results; 

	for (int i = 0; i < unknownSet.size(); i++)
	{
		Sample sample = unknownSet[i];
		int numFeatures = sample->size();

		float activation = 0 + w0; 
		for (int j = 0; j < numFeatures; j++) 
		{
			activation += (*sample)[j] * weightsList[j]; 
		}

		int output = activation > 0 ? 1 : 0; 
		results.push_back(output);
	}

	PrintPredictions(unknownSet, results);
}

void Perceptrons::RandomizeInitialWeights(int numFeatures)
{
	w0 = (float)(rand() % 100 + 1) / 100.f;

	for (int i = 0; i < numFeatures; i++)
	{
		float weight = (float)(rand() % 201 - 100) / 100.f;
		weightsList.push_back(weight);
	}
}
