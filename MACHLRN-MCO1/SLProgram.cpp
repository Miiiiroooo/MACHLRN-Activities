#include "pch.h"
#include "SLProgram.h"
#include "Perceptrons.h"
#include "ELM.h"
#include "ELM2.h"
#include "CSVHandler.h"


SLProgram::SLProgram()
{
	canRunProgram = false;
	defaultK = 10;
	testSplit = 0.1;
	repetitions = 0;
	learningRate = 0.2f;
	maxIterations = 100000;
	metric = 1;

	bestAlgorithmIndex = 0;
}

#pragma region Public Methods
bool SLProgram::Initialize(String directory, float testSplit, int repetitions, int maxIterations, float learningRate, int metric)
{
	CSVHandler csvHandler = CSVHandler();

	if (csvHandler.RetrieveDataFromDirectory(directory))
	{
		this->testSplit = testSplit;
		this->repetitions = repetitions;
		this->maxIterations = maxIterations; 
		this->learningRate = learningRate;
		this->metric = metric;

		unknownSet = csvHandler.GetUnknownSet();
		knownSet = csvHandler.GetKnownSet();
		ExtractTestSet();
		PrepareAllAlgorithms(maxIterations, learningRate);

		canRunProgram = true;

		return true;
	}
	else
	{
		std::cout << "Failed to retrieve data from the given directory.\n";
		std::cout << "Cannot Perform Perceptron Simulation without proper data.\n\n";
		std::cout << "---------------------------------------------------------\n\n";

		return false;
	}
}

void SLProgram::RunProgram()
{
	if (!canRunProgram)
		return;

	PerformRepeatedCrossValidation();
	PerformTest();
	PerformPredictions();
}
#pragma endregion

#pragma region Initialization
void SLProgram::ExtractTestSet()
{
	SortedHashMap<int, int> shuffledIndicesMap;
	ShuffleIndices(shuffledIndicesMap, knownSet.size());

	int testSize = knownSet.size() * testSplit;
	int count = 0; 

	for (Pair<int, int> pair : shuffledIndicesMap)
	{
		int index = pair.second; 

		if (count < testSize)
		{
			testSet.push_back(&knownSet[index]);
			count++;
		}
		else
		{
			trainSet.push_back(&knownSet[index]);
		}
	}
}

void SLProgram::PrepareAllAlgorithms(int maxIterations, float learningRate)
{
	if (algorithms.size() > 0)
	{
		for (auto& algorithm : algorithms)
		{
			free(algorithm);
		}

		algorithms.clear();
		algorithms.shrink_to_fit();
	}

	Perceptrons* perceptron = new Perceptrons(); 
	perceptron->InitializePerceptron(maxIterations, learningRate);
	algorithms.push_back(perceptron);

	ELM* elm_sigmoid = new ELM(); 
	elm_sigmoid->InitializeELM(EActivationFunctions::Sigmoid); 
	algorithms.push_back(elm_sigmoid); 

	ELM* elm_tanh = new ELM(); 
	elm_tanh->InitializeELM(EActivationFunctions::TanH);
	algorithms.push_back(elm_tanh); 

	ELM* elm_elu = new ELM(); 
	elm_elu->InitializeELM(EActivationFunctions::ELU);
	algorithms.push_back(elm_elu); 

	ELM2* elm2_sigmoid = new ELM2();
	elm2_sigmoid->InitializeELM(EActivationFunctions::Sigmoid);
	algorithms.push_back(elm2_sigmoid);

	ELM2* elm2_tanh = new ELM2(); 
	elm2_tanh->InitializeELM(EActivationFunctions::TanH);
	algorithms.push_back(elm2_tanh);

	ELM2* elm2_elu = new ELM2();
	elm2_elu->InitializeELM(EActivationFunctions::ELU);
	algorithms.push_back(elm2_elu);
}
#pragma endregion

#pragma region Supervised Learning Computations
void SLProgram::PerformRepeatedCrossValidation()
{
	if (!canRunProgram)
		return;

	List<float> avgScoresList;    // list of all average scores from the algorithms (or models)
	Matrix<float> scoresMatrix;   // row - folds (from all repetitions); col - scores of each algorithm
	float totalTime = 0.f;

	for (int i = 0; i < repetitions; i++)
	{
		Matrix<Sample> subsetsList = GenerateKSubsets(defaultK);         // row - k number of subsets; col - size of the subset; each cell represents a sample

		for (int j = 0; j < defaultK; j++)
		{
			List<Sample> trainingSetInFold = CombineSubsetsForTraining(subsetsList, j);   // contains all samples to be used for training
																					      // note: subsetsList[j] is used as the validation set
			List<float> scoresList;                                                       // used to keep track of score; each index represents the score from a given algorithm

			for (int k = 0; k < algorithms.size(); k++) 
			{ 
				auto start = std::chrono::high_resolution_clock::now();

				algorithms[k]->PerformTraining(trainingSetInFold); 
				algorithms[k]->PerformTest(subsetsList[j]);
				 
				GetScore(scoresList, k);
				PrintCrossValidationResults(i, j, k, algorithms[k]);
				algorithms[k]->Reset(); 

				auto end = std::chrono::high_resolution_clock::now();
				float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f;
				totalTime += duration;
				std::cout << i + 1 << " " << j + 1 << " " << k + 1 << "  duration: " << duration << "s\n";
			}

			scoresMatrix.push_back(scoresList); 
		}
	}

	std::cout << "\nTotal: " << totalTime;
	ComputeAverageScoresFromMatrix(avgScoresList, scoresMatrix);
	DetermineBestProcedure(avgScoresList);
	PrintEvaluation(avgScoresList, scoresMatrix);
}

void SLProgram::PerformTest()
{
	if (!canRunProgram)
		return;

	algorithms[bestAlgorithmIndex]->PerformTraining(trainSet);
	algorithms[bestAlgorithmIndex]->PerformTest(testSet);

	PrintFinalTestResults();
}

void SLProgram::PerformPredictions()
{
	List<Sample> unknownSetPointer;

	for (List<int>& unknown : unknownSet)
	{
		unknownSetPointer.push_back(&unknown);
	}

	algorithms[bestAlgorithmIndex]->PerformPredictions(unknownSetPointer);
}
#pragma endregion

#pragma region Subset Creation
Matrix<Sample> SLProgram::GenerateKSubsets(int k)
{
	int subsetSize = trainSet.size() / k; 

	SortedHashMap<int, int> shuffledIndicesMap;     // note: indices of the TRAINING SET, not the KNOWN SET 
	ShuffleIndices(shuffledIndicesMap, trainSet.size()); 

	Matrix<int> indicesWithinSubsets;               // row - subset; col - indices from TRAINING stored within the subset 
	DistributeSamplesFromShuffledMap(indicesWithinSubsets, shuffledIndicesMap, subsetSize, k); 

	Matrix<Sample> subsetsList;  

	for (int i = 0; i < k; i++) 
	{
		List<Sample> subset;  

		for (int j = 0; j < indicesWithinSubsets[i].size(); j++) 
		{
			int index = indicesWithinSubsets[i][j]; 
			subset.push_back(trainSet[index]); 
		}

		subsetsList.push_back(subset); 
	}

	return subsetsList; 
}

void SLProgram::ShuffleIndices(SortedHashMap<int, int>& shuffledIndicesMap, const unsigned int& size) 
{
	int max = size * 1000; 
	int index = 0;

	do
	{
		int random = rand() % max;
		if (shuffledIndicesMap.find(random) == shuffledIndicesMap.end())
		{
			shuffledIndicesMap[random] = index;
			index++;
		}
	} while (index < size);
}

void SLProgram::DistributeSamplesFromShuffledMap(Matrix<int>& indicesWithinSubsets, const SortedHashMap<int, int> shuffledIndicesMap, int subsetSize, int k)
{
	int remainder = shuffledIndicesMap.size() % k;  // the remaining number of samples when it is not an even split
	List<int> indicesInSubset;

	for (Pair<int, int> pair : shuffledIndicesMap)
	{
		indicesInSubset.push_back(pair.second);

		bool check1 = remainder > 0 && indicesInSubset.size() == subsetSize + 1;  // if there are some remaining number of samples, add it to the current subset
		bool check2 = remainder <= 0 && indicesInSubset.size() == subsetSize;     // if no more remainders, then proceed with the normal subset size

		if (check1 || check2) 
		{
			remainder--; 

			indicesWithinSubsets.push_back(indicesInSubset); 
			indicesInSubset.clear(); 
		}
	}
}
#pragma endregion

#pragma region Preparing Training Set and Validation Set
List<Sample> SLProgram::CombineSubsetsForTraining(const Matrix<Sample>& subsetsList, int testIndex)
{
	List<Sample> trainingSetInFold;  

	for (int i = 0; i < subsetsList.size(); i++)  
	{
		if (i != testIndex)  
		{
			trainingSetInFold.insert(trainingSetInFold.end(), subsetsList[i].begin(), subsetsList[i].end());  
		}
	}

	return trainingSetInFold;  
}
#pragma endregion

#pragma region Evaluation
void SLProgram::GetScore(List<float>& scoresList, int k)
{
	float score = 0;

	switch (metric)
	{
	case 1:
		score = algorithms[k]->GetAccuracy();
		break;

	case 2: 
		score = algorithms[k]->GetSensitivity();
		break;

	case 3:
		score = algorithms[k]->GetSpecificity();
		break;

	case 4:
		score = algorithms[k]->GetPrecision(); 
		break;

	case 5:
		score = algorithms[k]->GetRecall();
		break;

	case 6:
		score = algorithms[k]->GetFMeasure();
		break;

	default:
		score = algorithms[k]->GetAccuracy(); 
		break;
	}

	scoresList.push_back(score);
}

void SLProgram::ComputeAverageScoresFromMatrix(List<float>& avgScoresList, const Matrix<float>& scoresMatrix)
{
	List<float> totalList;

	for (int i = 0; i < scoresMatrix.size(); i++)
	{
		for (int j = 0; j < scoresMatrix[i].size(); j++)
		{
			if (i == 0)
			{
				totalList.push_back(scoresMatrix[i][j]);
			}
			else
			{
				totalList[j] += scoresMatrix[i][j]; 
			}
		}
	}

	for (int i = 0; i < totalList.size(); i++)
	{
		totalList[i] /= scoresMatrix.size();

		avgScoresList.push_back(totalList[i]);
	}
}

void SLProgram::DetermineBestProcedure(const List<float>& avgScoresList)
{
	int max = 0;

	for (int i = 0; i < avgScoresList.size(); i++)
	{
		if (avgScoresList[i] > avgScoresList[max])
		{
			max = i;
		}
	}

	bestAlgorithmIndex = max % algorithms.size();
}
#pragma endregion

#pragma region Prints
void SLProgram::PrintCrossValidationResults(int i, int j, int k, SLAlgorithm* algorithm)
{ 
	String msg = "repetitions:, " + std::to_string(i + 1); 
	msg += ", fold:, " + std::to_string(j + 1); 
	msg += ", algorithm:, " + algorithm->GetName() + "\n";  
	std::ios::openmode mode = i == 0 && j == 0 && k == 0 ? std::ios::out : std::ios::app; 
	algorithm->PrintOutConfusionMatrix("files/results/CrossValidation.csv", msg, mode);  
}

void SLProgram::PrintEvaluation(const List<float>& avgScoresList, const Matrix<float>& scoresMatrix)
{
	String message = "Score Metric :, " + MetricToString() + ",Best Algorithm : , " + algorithms[bestAlgorithmIndex]->GetName() + ", \n\n";
	message += "(repetition - fold), ";

	for (int i = 0; i < algorithms.size(); i++)
	{
		message += algorithms[i]->GetName() + ",";
	}

	message += ",\n";

	for (int i = 0; i < scoresMatrix.size(); i++)
	{
		int repetition = i / repetitions + 1;
		int fold = i % defaultK + 1;

		message += "(" + std::to_string(repetition) + " - " + std::to_string(fold) + "),";

		for (int j = 0; j < scoresMatrix[i].size(); j++)
		{
			message += std::to_string(scoresMatrix[i][j]) + ",";
		}

		message += "\n";
	}

	message += ",";

	for (int i = 0; i < avgScoresList.size(); i++)
	{
		message += std::to_string(avgScoresList[i]) + ",";
	}

	std::ios::openmode mode = std::ios::out; 
	CSVHandler handler = CSVHandler(); 
	handler.PrintOutMessage("files/results/Evaluation.csv", message, mode); 
}

void SLProgram::PrintFinalTestResults()
{
	String msg = "algorithm:, " + algorithms[bestAlgorithmIndex]->GetName() + "\n"; 
	std::ios::openmode mode = std::ios::out; 
	algorithms[bestAlgorithmIndex]->PrintOutConfusionMatrix("files/results/FinalTest.csv", msg, mode); 
}

String SLProgram::MetricToString()
{
	String string = "";

	switch (metric)
	{
	case 1:
		string = "Accuracy";
		break;

	case 2:
		string = "Sensitivity";
		break;

	case 3:
		string = "Specificity";
		break;

	case 4:
		string = "Precision";
		break;

	case 5:
		string = "Recall";
		break;

	case 6:
		string = "F-Measure";
		break;

	default:
		string = "Accuracy";
		break;
	}

	return string;
}
#pragma endregion