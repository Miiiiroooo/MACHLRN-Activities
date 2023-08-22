#include "pch.h"
#include "ELM.h"
#include "CSVHandler.h"


ELM::ELM()
{
	name = "Extreme Learning Machine";
	hiddenNodes = 1000;
	function = EActivationFunctions::Unknown;
}

void ELM::InitializeELM(EActivationFunctions function)
{
	this->function = function;

	canPerformCalculations = true;
}

void ELM::Reset()
{
	SLAlgorithm::Reset();

	biasList.fill(0.f);
	hiddenWeightsMatrix.fill(0.f);
	outputWeightsList.fill(0.f);
}

void ELM::PerformTraining(const List<Sample>& trainSet)
{
	if (!canPerformCalculations)
		return;

	// Step 0: Resize hidden nodes
	int sampleSize = trainSet.size();
	hiddenNodes = sampleSize / 2;

	// Step 1: Randomly assign weights and bias
	Sample temp = trainSet[0];
	int numFeatures = temp->size() - 1;
	RandomizeInitialWeights(numFeatures);

	// Step 2: Calculate hidden layer output
	arma::Mat<float> dataset = ConvertDataSetIntoMatrix(trainSet, numFeatures);
	arma::Mat<float> H = ComputeHiddenLayerOutputMatrix(dataset); 

	// Step 3: Calculate output weight matrix
	arma::Mat<float> Hplus = (H.t() * H).i() * H.t();
	arma::Col<int> Y = ExtractOutputDataFromDataset(trainSet); 
	outputWeightsList = Hplus * Y; 
}

void ELM::PerformTest(const List<Sample>& testSet)
{
	if (!canPerformCalculations)
		return;

	// Step 4: Use output weight matrix to make a prediction
	Sample temp = testSet[0];
	int numFeatures = temp->size() - 1;
	arma::Mat<float> dataset = ConvertDataSetIntoMatrix(testSet, numFeatures);
	arma::Mat<float> H = ComputeHiddenLayerOutputMatrix(dataset); 

	arma::Col<float> activationList = H * outputWeightsList;
	arma::Col<int> outputList = arma::Col<int>(activationList.n_elem);
	for (int i = 0; i < activationList.n_elem; i++)
	{
		outputList(i) = activationList(i) > 0 ? 1 : 0;
	}

	arma::Col<int> expectedList = ExtractOutputDataFromDataset(testSet); 
	EvaluateResults(outputList, expectedList); 
}

void ELM::PerformPredictions(const List<Sample>& unknownSet)
{
	if (!canPerformCalculations)
		return;

	Sample temp = unknownSet[0];
	int numFeatures = temp->size();
	arma::Mat<float> dataset = ConvertDataSetIntoMatrix(unknownSet, numFeatures); 
	arma::Mat<float> H = ComputeHiddenLayerOutputMatrix(dataset); 

	List<int> results; 
	arma::Col<float> activationList = H * outputWeightsList;

	for (int i = 0; i < activationList.n_elem; i++) 
	{
		int output = activationList(i) > 0 ? 1 : 0; 
		results.push_back(output);
	}

	PrintPredictions(unknownSet, results);  
}

void ELM::RandomizeInitialWeights(int numFeatures)
{
	biasList = arma::Col<float>(hiddenNodes); 
	for (int i = 0; i < hiddenNodes; ++i)
	{
		float weight = (float)(rand() % 100 + 1) / 100.f;
		biasList(i) = weight;
	}

	hiddenWeightsMatrix = arma::Mat<float>(hiddenNodes, numFeatures);
	for (int i = 0; i < hiddenNodes; ++i)
	{
		for (int j = 0; j < numFeatures; ++j)
		{
			float weight = (float)(rand() % 201 - 100) / 100.f;
			hiddenWeightsMatrix(i, j) = weight;
		}
	}
}

arma::Mat<float> ELM::ConvertDataSetIntoMatrix(const List<Sample>& trainSet, int numFeatures)
{
	int rowSize = trainSet.size();            // no. of samples
	arma::Mat<float> matrix = arma::Mat<float>(rowSize, numFeatures);

	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < numFeatures; j++)
		{
			matrix(i, j) = (*trainSet[i])[j]; 
		}
	}

	return matrix;
}

arma::Mat<float> ELM::ComputeHiddenLayerOutputMatrix(const arma::Mat<float>& trainSet)
{
	int sampleSize = trainSet.n_rows; 
	arma::Mat<float> H = arma::Mat<float>(sampleSize, hiddenNodes);

	for (int i = 0; i < sampleSize; i++)
	{
		arma::Col<float> sample = trainSet.row(i).as_col();

		for (int j = 0; j < hiddenNodes; j++)
		{
			arma::Col<float> weightVector = hiddenWeightsMatrix.row(j).as_col();

			float hiddenActivation = arma::dot(sample, weightVector);
			hiddenActivation += biasList[j];

			if (function == EActivationFunctions::Sigmoid)
			{
				H(i, j) = 1.f / (1.f + std::exp(-hiddenActivation)); 
			}
			else if (function == EActivationFunctions::TanH)
			{
				H(i, j) = std::tanh(hiddenActivation); 
			}
			else if (function == EActivationFunctions::ELU)
			{
				H(i, j) = hiddenActivation >= 0 ? hiddenActivation : (std::exp(hiddenActivation) - 1);
			}
		}
	}

	return H;
}

arma::Col<int> ELM::ExtractOutputDataFromDataset(const List<Sample>& trainSet)
{
	int rowSize = trainSet.size();
	arma::Col<int> output = arma::Col<int>(rowSize);

	for (int i = 0; i < rowSize; i++)
	{
		Sample sample = trainSet[i];
		int numFeatures = sample->size();

		output(i) = (*sample)[numFeatures - 1];
	}

	return output;
}

void ELM::EvaluateResults(const arma::Col<int>& outputList, const arma::Col<int>& expectedList) 
{
	for (int i = 0; i < expectedList.n_elem; i++)
	{
		int output = outputList(i); 
		int expected = expectedList(i);
			 
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
