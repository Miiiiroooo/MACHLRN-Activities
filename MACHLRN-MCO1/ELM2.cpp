#include "pch.h"
#include "ELM2.h"
#include "CSVHandler.h"


ELM2::ELM2()
{
	name = "Extreme Learning Machine 2";
	hiddenNodes = 1000;
	function = EActivationFunctions::Unknown; 
}

void ELM2::InitializeELM(EActivationFunctions function)
{
	this->function = function;

	canPerformCalculations = true;
}

void ELM2::Reset()
{
	SLAlgorithm::Reset();

	biasList.fill(0.f);
	hiddenWeightsMatrix.fill(0.f);
	outputWeightsList.fill(0.f);
}

void ELM2::PerformTraining(const List<Sample>& trainSet)
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
	arma::Mat<float> weightsWithBiasMatrix = hiddenWeightsMatrix; 
	weightsWithBiasMatrix.insert_cols(weightsWithBiasMatrix.n_cols, biasList); 

	arma::Mat<float> H = dataset * weightsWithBiasMatrix.t();	
	H = ApplyActivationFunction(H);

	// Step 3: Calculate output weight matrix
	arma::Mat<float> Hplus = (H.t() * H).i() * H.t();
	arma::Col<int> Y = ExtractOutputDataFromDataset(trainSet);
	outputWeightsList = Hplus * Y;
}

void ELM2::PerformTest(const List<Sample>& testSet)
{
	if (!canPerformCalculations)
		return;

	// Step 4: Use output weight matrix to make a prediction
	Sample temp = testSet[0];
	int numFeatures = temp->size() - 1;
	arma::Mat<float> dataset = ConvertDataSetIntoMatrix(testSet, numFeatures);
	arma::Mat<float> weightsWithBiasMatrix = hiddenWeightsMatrix; 
	weightsWithBiasMatrix.insert_cols(weightsWithBiasMatrix.n_cols, biasList);

	arma::Mat<float> H = dataset * weightsWithBiasMatrix.t(); 
	H = ApplyActivationFunction(H); 

	arma::Col<float> activationList = H * outputWeightsList;
	arma::Col<int> outputList = arma::Col<int>(activationList.n_elem);
	for (int i = 0; i < activationList.n_elem; i++)
	{
		outputList(i) = activationList(i) > 0 ? 1 : 0;
	}

	arma::Col<int> expectedList = ExtractOutputDataFromDataset(testSet);
	EvaluateResults(outputList, expectedList);
}

void ELM2::PerformPredictions(const List<Sample>& unknownSet)
{
	if (!canPerformCalculations)
		return;

	Sample temp = unknownSet[0];
	int numFeatures = temp->size();
	arma::Mat<float> dataset = ConvertDataSetIntoMatrix(unknownSet, numFeatures);
	arma::Mat<float> weightsWithBiasMatrix = hiddenWeightsMatrix; 
	weightsWithBiasMatrix.insert_cols(weightsWithBiasMatrix.n_cols, biasList);

	arma::Mat<float> H = dataset * weightsWithBiasMatrix.t();  
	H = ApplyActivationFunction(H);  
	 
	List<int> results; 
	arma::Col<float> activationList = H * outputWeightsList; 

	for (int i = 0; i < activationList.n_elem; i++) 
	{
		int output = activationList(i) > 0 ? 1 : 0; 
		results.push_back(output); 
	}

	PrintPredictions(unknownSet, results); 
}

void ELM2::RandomizeInitialWeights(int numFeatures)
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

arma::Mat<float> ELM2::ConvertDataSetIntoMatrix(const List<Sample>& trainSet, int numFeatures)
{
	int rowSize = trainSet.size();            // no. of samples
	arma::Mat<float> matrix = arma::Mat<float>(rowSize, numFeatures + 1);

	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < numFeatures + 1; j++)
		{
			if (j == numFeatures)
			{
				matrix(i, j) = 1;
			}
			else
			{
				matrix(i, j) = (*trainSet[i])[j];
			}
		}
	}

	return matrix;
}

arma::Mat<float> ELM2::ApplyActivationFunction(const arma::Mat<float>& H)
{
	arma::Mat<float> newH = arma::Mat<float>(H.n_rows, H.n_cols);

	for (int i = 0; i < H.n_rows; i++)
	{
		for (int j = 0; j < H.n_cols; j++)
		{
			if (function == EActivationFunctions::Sigmoid)
			{
				newH(i, j) = 1.f / (1.f + std::exp(-H(i, j))); 
			}
			else if (function == EActivationFunctions::TanH)
			{
				newH(i, j) = std::tanh(H(i, j));
			}
			else if (function == EActivationFunctions::ELU)
			{
				newH(i, j) = H(i, j) >= 0 ? H(i, j) : (std::exp(H(i, j)) - 1);
			}
		}
	}

	return newH;
}

arma::Col<int> ELM2::ExtractOutputDataFromDataset(const List<Sample>& trainSet)
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

void ELM2::EvaluateResults(const arma::Col<int>& outputList, const arma::Col<int>& expectedList)
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
