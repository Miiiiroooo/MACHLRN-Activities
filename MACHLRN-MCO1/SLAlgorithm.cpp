#include "pch.h"
#include "SLAlgorithm.h"
#include "CSVHandler.h"

SLAlgorithm::SLAlgorithm()
{
	canPerformCalculations = false;

	confusionMatrix = { {0, 0}, {0, 0} };

	accuracy = 0.f;
	recall = 0.f;
	precision = 0.f;
	f_measure = 0.f;
	sensitivity = 0.f;
	specificity = 0.f;
}

String SLAlgorithm::GetName()
{
	return name;
}

float SLAlgorithm::GetAccuracy()
{
	return accuracy;
}

float SLAlgorithm::GetRecall()
{
	return recall;
}

float SLAlgorithm::GetPrecision()
{
	return precision;
}

float SLAlgorithm::GetFMeasure()
{
	return f_measure;
}

float SLAlgorithm::GetSensitivity()
{
	return sensitivity;
}

float SLAlgorithm::GetSpecificity()
{
	return specificity;
}

void SLAlgorithm::PrintOutConfusionMatrix(String directory, String message, std::ios::openmode mode)
{
	message += " , Predicted RT, Predicted Non-RT,\n";
	message += "Actual RT, " + std::to_string(confusionMatrix[0][0]) + ", " + std::to_string(confusionMatrix[0][1]) + ",";
	message += " ,accuracy: ," + std::to_string(accuracy) + ", sensitivity: ," + std::to_string(sensitivity) + ", specificity: ," + std::to_string(specificity) + ",\n";
	message += "Actual Non-RT, " + std::to_string(confusionMatrix[1][0]) + ", " + std::to_string(confusionMatrix[1][1]) + ",";
	message += " ,precision: ," + std::to_string(precision) + ", recall: ," + std::to_string(recall) + ", f-measure: ," + std::to_string(f_measure) + ",\n\n";

	CSVHandler handler = CSVHandler(); 
	handler.PrintOutMessage(directory, message, mode);  
}

void SLAlgorithm::PrintPredictions(const List<Sample>& unknownSet, const List<int>& results)
{
	String message = "a1,a2,a3,a4,a5,a6,a7,a8,d1,d2,d3,d4,d5,u1,u4,v1,v2,v3,v4,r1,r2,r3,r4,s1,s2,s3,s4,s5,s6,s7,s8,p1,p2,p3,p4,p5,p6,p7,p8,p11,p12,RiskTaker?,\n";

	for (int i = 0; i < unknownSet.size(); i++)
	{
		Sample sample = unknownSet[i];

		for (int j = 0; j < sample->size(); j++)
		{
			int value = (*sample)[j] == -1 ? 0 : (*sample)[j];
			message += std::to_string(value) + ", "; 
		}

		message += std::to_string(results[i]) + ",\n";
	}

	String directory = "files/results/Predictions.csv";
	std::ios::openmode mode = std::ios::out;

	CSVHandler handler = CSVHandler(); 
	handler.PrintOutMessage(directory, message, mode); 
}

void SLAlgorithm::Reset()
{
	ResetMatrix(confusionMatrix); 
	confusionMatrix = { {0, 0}, {0, 0} };

	accuracy = 0.f;
	recall = 0.f;
	precision = 0.f;
	f_measure = 0.f;
	sensitivity = 0.f;
	specificity = 0.f;
}

void SLAlgorithm::ResetMatrix(Matrix<int>& matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		matrix[i].clear();
	}
	matrix.clear();
}

void SLAlgorithm::CalculatePerformance()
{
	int TP = confusionMatrix[0][0];
	int FN = confusionMatrix[0][1];
	int FP = confusionMatrix[1][0];
	int TN = confusionMatrix[1][1];

	int total = TP + FP + FN + TN; 
	int correct = TP + TN; 
	int incorrect = FP + FN; 

	accuracy = (float)correct / (float)total; 

	float denominator = 0.f;

	denominator = (float)(TP + FN);
	recall = denominator == 0 ? 0 : (float)TP / denominator;

	denominator = (float)(TP + FP);
	precision = denominator == 0 ? 0 : (float)TP / denominator; 

	denominator = (precision + recall);
	f_measure = denominator == 0 ? 0 : (2 * precision * recall / (precision + recall));

	denominator = (float)(TP + FN);
	sensitivity = denominator == 0 ? 0 : (float)TP / denominator;

	denominator = (float)(TN + FP);
	specificity = denominator == 0 ? 0 : (float)TN / denominator;
}