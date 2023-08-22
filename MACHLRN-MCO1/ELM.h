#pragma once

#include "SLAlgorithm.h"
#include "EActivationFunctions.h"

class ELM : public SLAlgorithm
{
public:
	ELM();

	void InitializeELM(EActivationFunctions fucntion);
	void Reset();

	void PerformTraining(const List<Sample>& trainSet);
	void PerformTest(const List<Sample>& testSet);
	void PerformPredictions(const List<Sample>& unknownSet);

private:
	void RandomizeInitialWeights(int numFeatures);
	arma::Mat<float> ConvertDataSetIntoMatrix(const List<Sample>& trainSet, int numFeatures); 
	arma::Mat<float> ComputeHiddenLayerOutputMatrix(const arma::Mat<float>& trainSet);
	arma::Col<int> ExtractOutputDataFromDataset(const List<Sample>& trainSet);
	void EvaluateResults(const arma::Col<int>& activationsList, const arma::Col<int>& expectedList);


private:
	arma::Col<float> biasList;                      // n = hidden nodes;             or the connection (weights) from one bias node to all hidden nodes
	arma::Mat<float> hiddenWeightsMatrix;           // n = hidden nodes, m = inputs; or the intertwined connection between the inputs and the hidden nodes
	arma::Col<float> outputWeightsList;             // n = hidden nodes;             or the connection (weights) from all hidden nodes to the single output node

	int hiddenNodes;
	EActivationFunctions function;
};

