#pragma once

#include "SLAlgorithm.h"

class SLProgram
{
public:
	SLProgram();

	bool Initialize(String directory, float testSplit, int repetitions, int maxIterations, float learningRate, int metric);
	void RunProgram();

private:
	// Initialization
	void ExtractTestSet();
	void PrepareAllAlgorithms(int maxIterations, float learningRate);

	// Supervised Learning Computations
	void PerformRepeatedCrossValidation();
	void PerformTest();
	void PerformPredictions();

	// Subset Creation
	Matrix<Sample> GenerateKSubsets(int k);
	void ShuffleIndices(SortedHashMap<int, int>& shuffledIndicesMap, const unsigned int& size);
	void DistributeSamplesFromShuffledMap(Matrix<int>& distributedIndices, const SortedHashMap<int, int> shuffledIndicesMap, int subsetSize, int k);

	// Preparing Training Set and Validation Set
	List<Sample> CombineSubsetsForTraining(const Matrix<Sample>& subsetsList, int testIndex); 

	// Evaluation 
	void GetScore(List<float>& scoresList, int k);
	void ComputeAverageScoresFromMatrix(List<float>& avgScoresList, const Matrix<float>& scoresList);
	void DetermineBestProcedure(const List<float>& avgScoresList);

	// Prints
	void PrintCrossValidationResults(int i, int j, int k, SLAlgorithm* algorithm);
	void PrintEvaluation(const List<float>& avgScoresList, const Matrix<float>& scoresMatrix);
	void PrintFinalTestResults();
	String MetricToString();


private:
	bool canRunProgram;
	int defaultK;
	float testSplit;
	int repetitions;
	int maxIterations;
	float learningRate;
	int metric;

	Matrix<int> knownSet;       
	Matrix<int> unknownSet;         
	List<Sample> trainSet;
	List<Sample> testSet;

	List<SLAlgorithm*> algorithms;
	int bestAlgorithmIndex;
};

