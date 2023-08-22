#include "pch.h"
#include "USLProgram.h"
#include "CSVHandler.h"

USLProgram::USLProgram()
{
    canRunProgram = false;

	numRows = 16;
	numCols = 16;
	numNodes = numRows * numCols;

	trainingCycles = 100000;
	learningRate = 0.5f;
	radius = 3;

	numFeatures = 0;
	k_nearest = 7;
	numClusters = 5;
	numClusterings = 10;
	maxCentroidMovement =100;

	for (int i = 0; i < numClusters; i++)
	{
		Cluster newCluster(i + 1);
		clustersList.push_back(newCluster);
	}

	labelNames = {
		{"Sex", "Male", "Female"},
		{"Age", "Age 9-11", "Age 12-17"},
		{"Economic Status", "Low Income", "Mid/High Income"},
		{"Setting", "Rural", "Urban"},
		{"Risk-Taker", "Non Risk-Taker", "Risk-Taker"},
	};
}

#pragma region Public Methods
bool USLProgram::Initialize(String directory)
{
	CSVHandler csvHandler = CSVHandler();

	if (csvHandler.RetrieveDataFromDirectory(directory))
	{
		unknownSet = csvHandler.GetUnknownSet();
		knownSet = csvHandler.GetKnownSet();

		numFeatures = knownSet[0].featuresList.size();
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

void USLProgram::RunProgram()
{
	if (!canRunProgram)
		return;

	PerformSOM(); 
	LabelNodes();
	ClusterNodes();
	DetermineProfiles(); 
	PrintMap(); 
	LabelUnknownSamples(); 
}
#pragma endregion

#pragma region Main Operations
void USLProgram::PerformSOM()
{
	InitializeNodes();

	for (int i = 0; i < trainingCycles; i++)
	{
		int index = rand() % knownSet.size();
		Sample sample = knownSet[index]; 

		int BMU_index = DetermineBMU(sample);
		UpdateLearningParameters(i);
		AdjustWeights(sample, BMU_index);
	}
}

void USLProgram::LabelNodes()
{ 
	// Upload samples to nodes
	for (int i = 0; i < knownSet.size(); i++)
	{
		Sample& sample = knownSet[i];
		int index = DetermineBMU(sample);

		Node& node = nodesList[index];  
		node.samplesList.push_back(sample); 
	}

	// Label nodes
	for (int i = 0; i < numNodes; i++)
	{
		Node& node = nodesList[i];
		GetKNearestSamples(node, i);
		node.DetermineDominantLabels();
	}
}

void USLProgram::ClusterNodes()
{
	List<List<Cluster>> clusteringsList;

	for (int i = 0; i < numClusterings; i++)
	{
		List<Cluster> newClustering;
		InitializeClusters(newClustering);

		bool hasFinalizeCentroids = false;
		int itr = 0;

		while (!hasFinalizeCentroids && itr < maxCentroidMovement) 
		{
			for (int i = 0; i < numClusters; i++) 
			{
				Cluster& cluster = newClustering[i];
				cluster.nodesList.clear(); 
			}

			for (int i = 0; i < numNodes; i++) 
			{
				AssignNodesToClusters(newClustering, i);  
			} 

			hasFinalizeCentroids = RecalculateCentroids(newClustering);  
			itr++; 
		}

		clusteringsList.push_back(newClustering);
	}

	DetermineBestCluster(clusteringsList);
	for (int i = 0; i < clustersList.size(); i++)
	{
		Cluster& cluster = clustersList[i];
		for (int j = 0; j < cluster.nodesList.size(); j++)
		{
			Node* node = cluster.nodesList[j];
			node->clusterID = cluster.id;
		}
	}
}

void USLProgram::DetermineProfiles()
{
	// local profile
	for (int i = 0; i < clustersList.size(); i++)
	{
		Cluster& cluster = clustersList[i];
		cluster.DetermineLocalProfile(labelNames);
	}

	// global profile
	for (int i = 0; i < labelNames.size(); i++)
	{
		List<String> labels = labelNames[i];
		Dictionary<String, int> characteristic;

		for (int j = 1; j < labels.size(); j++)
		{
			characteristic[labels[j]] = 0;
		}

		globalProfile[labels[0]] = characteristic;
	}

	for (int i = 0; i < labelNames.size(); i++)
	{
		List<String> labels = labelNames[i];
		Dictionary<String, int>& characteristic = globalProfile[labels[0]];

		for (int j = 0; j < nodesList.size(); j++)
		{
			Node& node = nodesList[j];

			int value = node.dominantLabelsList[i];
			String key = labels[value + 1];
			characteristic[key] += 1;
		}
	}
}

void USLProgram::PrintMap()
{
	String directory = "MACHLRN-Unity/Assets/Resources/nodes.csv";
	std::ofstream outFile; 
	outFile.open(directory, std::ios::out);  

	if (outFile) 
	{
		PrintNodes(outFile);
	}
	else
	{
		std::cout << "Cannot open file at: " << directory << "\n"; 
		std::cout << "Cannot print the nodes of the program" << "\n"; 
	}
	outFile.close(); 


	directory = "MACHLRN-Unity/Assets/Resources/profiles.csv";
	outFile.open(directory, std::ios::out); 

	if (outFile)
	{
		PrintProfiles(outFile); 
	}
	else
	{
		std::cout << "Cannot open file at: " << directory << "\n"; 
		std::cout << "Cannot print the profiles of the program" << "\n"; 
	}
	outFile.close(); 
}

void USLProgram::LabelUnknownSamples()
{

}
#pragma endregion

#pragma region SOM
void USLProgram::InitializeNodes()
{
	for (int i = 0; i < numNodes; i++)
	{
		Node newNode = Node();
		newNode.weights = Vector<float>(numFeatures);

		for (int j = 0; j < numFeatures; j++)
		{
			float value = (rand() % 101) / 100.f;
			newNode.weights(j) = value;
		}

		nodesList.push_back(newNode);
	}
}

int USLProgram::DetermineBMU(const Sample& sample)
{
	float closestDistance = 10000000.f;
	int closestIndex = 0;

	for (int i = 0; i < numNodes; i++)
	{
		float distance = GetDistance(sample.featuresList, nodesList[i].weights);

		if (distance < closestDistance) 
		{
			closestDistance = distance; 
			closestIndex = i;
		}
	}

	return closestIndex;
}

template <typename T, typename U>
float USLProgram::GetDistance(const Vector<T>& A, const Vector<U>& B)
{
	float distance = 0;
	Vector<float> difference = A - B; 

	for (int j = 0; j < numFeatures; j++)
	{
		distance += std::pow(difference(j), 2); 
	}

	distance = std::sqrt(distance); 

	return distance;
}

void USLProgram::UpdateLearningParameters(int trainingCycle)
{
	if (trainingCycle < 50000)
	{
		learningRate = 0.5f;
		radius = 3;
	}
	else if (trainingCycle < 75000)
	{
		learningRate = 0.25f;
		radius = 2;
	}
	else
	{
		learningRate = 0.1f;
		radius = 1;
	}
}

void USLProgram::AdjustWeights(const Sample& sample, int BMU_index)
{
	int BMU_row = BMU_index / numRows;
	int BMU_col = BMU_index % numCols;

	for (int i = BMU_row - radius; i <= BMU_row + radius; i++)
	{
		for (int j = BMU_col - radius; j <= BMU_col + radius; j++)
		{
			if ((i >= 0 && i < numRows) && (j >= 0 && j < numCols))
			{
				int nodeIndex = i * numRows + j;
				Node& node = nodesList[nodeIndex];

				float distance = std::sqrt(std::pow(i - BMU_row, 2) + std::pow((j - BMU_col), 2));
				float gaussian_value = std::exp(-std::pow(distance, 2) / std::pow(radius, 2));
				node.weights += RoundOfVector(learningRate * gaussian_value * (sample.featuresList - node.weights));
			}
		}
	}
}
#pragma endregion

#pragma region Labelling
void USLProgram::GetKNearestSamples(Node& node, int nodeIndex) 
{
	int nodeRow = nodeIndex / numRows; 
	int nodeCol = nodeIndex % numCols; 
	int r = 0;

	while (node.nearestSamplesList.size() < k_nearest || r >= 4)
	{
		List<Sample> samplesList;

		// Get samples from outer boxes
		for (int i = nodeRow - r; i <= nodeRow + r; i++) 
		{
			for (int j = nodeCol - r; j <= nodeCol + r; j++) 
			{
				if (i < 0 || i >= numRows || j < 0 || j >= numCols)
				{
					continue;
				}

				if (i != nodeRow - r && i != nodeRow + r && j != nodeCol - r && j != nodeCol + r)
				{
					continue;
				}

				int neighborIndex = i * numRows + j; 
				Node& neighbor = nodesList[neighborIndex]; 
				samplesList.insert(samplesList.end(), neighbor.samplesList.begin(), neighbor.samplesList.end());
			}
		}

		// Add samples depending on the list size
		if (node.nearestSamplesList.size() + samplesList.size() <= k_nearest)
		{
			node.nearestSamplesList.insert(node.nearestSamplesList.end(), samplesList.begin(), samplesList.end()); 
		}
		else
		{
			InsertRemainingNodes(node, samplesList);
		}

		r++;
	}
	
}

void USLProgram::InsertRemainingNodes(Node& node, List<Sample>& samplesList)
{
	List<Pair<Sample, float>> distanceList;

	// Get all their distances
	for (Sample& sample : samplesList)
	{
		float distance = GetDistance(sample.featuresList, node.weights);
		distanceList.push_back(Pair<Sample, float>(sample, distance));
	}

	// Sort
	std::sort(distanceList.begin(),
		distanceList.end(),
		[](Pair<Sample, float>& A, Pair<Sample, float>& B) {
			return A.second < B.second;
		});

	// Insert remaining nodes
	int missing = k_nearest - node.nearestSamplesList.size();
	for (int j = 0; j < missing; j++)
	{
		node.nearestSamplesList.push_back(distanceList[j].first);
	}
}
#pragma endregion

#pragma region Clustering
void USLProgram::InitializeClusters(List<Cluster>& newClustering)
{
	List<int> indicesList; 

	for (int i = 0; i < numClusters; i++)
	{
		// get random node
		int index = 0; 
		do
		{ 
			index = rand() % nodesList.size(); 
		} while (std::find(indicesList.begin(), indicesList.end(), index) != indicesList.end()); 
		indicesList.push_back(index); 

		// initialize cluster  
		Cluster newCluster(i + 1);
		newCluster.centroid = Vector<float>(numFeatures);  

		for (int j = 0; j < numFeatures; j++)
		{
			newCluster.centroid(j) = (float)nodesList[index].weights(j);
		}

		newClustering.push_back(newCluster); 
	}
}

void USLProgram::AssignNodesToClusters(List<Cluster>& newClustering, int nodeIndex) 
{
	int closestIndex = 0; 
	float closestDistance = 10000000.f; 
	Node& node = nodesList[nodeIndex]; 

	for (int i = 0; i < newClustering.size(); i++) 
	{
		float distance = GetDistance(node.weights, newClustering[i].centroid);

		if (distance < closestDistance) 
		{
			closestIndex = i; 
			closestDistance = distance; 
		}
	}
 
	newClustering[closestIndex].nodesList.push_back(&node);  
}

bool USLProgram::RecalculateCentroids(List<Cluster>& newClustering)
{
	bool isEqual = true;

	for (int i = 0; i < newClustering.size(); i++) 
	{
		Cluster& cluster = newClustering[i];
		Vector<float> newCentroid(numFeatures);
		newCentroid.fill(0.f);

		for (int j = 0; j < cluster.nodesList.size(); j++)
		{
			Node* node = cluster.nodesList[j];

			for (int k = 0; k < numFeatures; k++)
			{
				newCentroid(k) += node->weights(k);
			}
		}

		newCentroid /= cluster.nodesList.size();
		isEqual = isEqual && arma::approx_equal(newCentroid, cluster.centroid, "abs_diff", 0.001f);

		cluster.centroid = newCentroid;
	}

	return isEqual;
}
#pragma endregion

#pragma region DBI Scoring
void USLProgram::DetermineBestCluster(const List<List<Cluster>>& clusteringsList)  
{
	float lowestScore = 1000000.f;
	int bestIndex = 0;

	for (int i = 0; i < clusteringsList.size(); i++)
	{
		List<Cluster> clustering = clusteringsList[i];
		float score = CalculateDBIScore(clustering);

		if (score < lowestScore)
		{
			lowestScore = score;
			bestIndex = i;
		}
	}

	clustersList = clusteringsList[bestIndex];
}

float USLProgram::CalculateDBIScore(const List<Cluster>& clustering)
{
	float score = 0; 

	for (int i = 0; i < clustering.size(); i++) 
	{
		score += GetHighestSimilarityScore(clustering[i], clustering); 
	}

	score /= clustering.size(); 

	return score; 
}

float USLProgram::GetHighestSimilarityScore(const Cluster& clusterToEvaluate, const List<Cluster>& clusterings)
{
	List<float> similarityScoresList; 

	for (Cluster cluster : clusterings) 
	{
		if (cluster.id != clusterToEvaluate.id) 
		{
			similarityScoresList.push_back(GetSimilarityScore(cluster, clusterToEvaluate)); 
		}
	}

	std::sort(similarityScoresList.begin(), similarityScoresList.end(), std::greater<float>()); 

	return similarityScoresList[0];
}

float USLProgram::GetSimilarityScore(const Cluster& cluster1, const Cluster& cluster2)
{
	// Get Intra Cluster Distance Score of te First Cluster
	float interCluster1 = 0; 
	for (Node* node : cluster1.nodesList)
	{
		interCluster1 += GetDistance(cluster1.centroid, node->weights);
	}
	interCluster1 /= cluster1.nodesList.size();

	// Get Intra Cluster Distance Score of te Second Cluster
	float interCluster2 = 0; 
	for (Node* node : cluster2.nodesList) 
	{
		interCluster2 += GetDistance(cluster2.centroid, node->weights); 
	}
	interCluster2 /= cluster2.nodesList.size(); 

	// Get the Distance Between Both of the Clusters
	float betweenClusters = GetDistance(cluster1.centroid, cluster2.centroid);

	return ((interCluster1 + interCluster2) / betweenClusters); 
}
#pragma endregion

#pragma region Print
void USLProgram::PrintNodes(std::ofstream& outFile)
{
	String msg = "";

	for (int i = 0; i < numRows; i++) 
	{
		for (int j = 0; j < numCols; j++) 
		{
			int index = i * numRows + j; 
			Node& node = nodesList[index]; 

			msg += std::to_string(node.clusterID) + ",";
		}
		msg += "\n"; 
	}

	outFile << msg;
}

void USLProgram::PrintProfiles(std::ofstream& outFile)
{
	String msg = "Global Profile:\n";

	for (int i = 0; i < labelNames.size(); i++) 
	{
		List<String> labels = labelNames[i]; 
		String mainKey = labels[0];
		int total = 0;

		for (int j = 1; j < labels.size(); j++) 
		{
			total += globalProfile[mainKey][labels[j]];
		}

		for (int j = 1; j < labels.size(); j++)
		{
			float percent = ((float)globalProfile[mainKey][labels[j]] / (float)total) * 100.f;
			percent = (int)(percent * 100) / 100.f;

			msg += labels[j] + "," + std::to_string(percent) + "%,";
		}
		msg += "\n";
	}

	for (int i = 0; i < clustersList.size(); i++)
	{
		Cluster& cluster = clustersList[i];
		msg += "Cluster" + std::to_string(cluster.id) + " Profile:\n";

		for (int j = 0; j < labelNames.size(); j++) 
		{
			List<String> labels = labelNames[j]; 
			String mainKey = labels[0];
			int total = 0;

			for (int k = 1; k < labels.size(); k++)
			{
				total += cluster.profile[mainKey][labels[k]];
			}

			for (int k = 1; k < labels.size(); k++)
			{
				float percent = ((float)cluster.profile[mainKey][labels[k]] / (float)total) * 100.f;
				percent = (int)(percent * 100) / 100.f;

				msg += labels[k] + "," + std::to_string(percent) + "%,";
			}
			msg += "\n";
		}
	}

	outFile << msg;
}
#pragma endregion