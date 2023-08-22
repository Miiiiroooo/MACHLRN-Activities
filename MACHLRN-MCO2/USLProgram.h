#pragma once

#include "Node.h"
#include "Cluster.h"

class USLProgram
{
public:
	USLProgram();

	bool Initialize(String directory);
	void RunProgram();

private:
	// Main Operations
	void PerformSOM();
	void LabelNodes();
	void ClusterNodes();
	void DetermineProfiles();
	void PrintMap();
	void LabelUnknownSamples();

	// SOM
	void InitializeNodes();
	int DetermineBMU(const Sample& sample);
	template <typename T, typename U> 
	float GetDistance(const Vector<T>& A, const Vector<U>& B);
	void UpdateLearningParameters(int trainingCycle);
	void AdjustWeights(const Sample& sample, int BMU_index);
	
	// Labelling 
	void GetKNearestSamples(Node& node, int nodeIndex);
	void InsertRemainingNodes(Node& node, List<Sample>& samplesList);

	// Clustering
	void InitializeClusters(List<Cluster>& newClustering);
	void AssignNodesToClusters(List<Cluster>& newClustering, int nodeIndex);
	bool RecalculateCentroids(List<Cluster>& newClustering);
	
	// DBI Scoring
	void DetermineBestCluster(const List<List<Cluster>>& clusteringsList);
	float CalculateDBIScore(const List<Cluster>& clustering);
	float GetHighestSimilarityScore(const Cluster& clusterToEvaluate, const List<Cluster>& clusterings);
	float GetSimilarityScore(const Cluster& cluster1, const Cluster& cluster2);
	
	// Print
	void PrintNodes(std::ofstream& outFile);
	void PrintProfiles(std::ofstream& outFile);


private:
	bool canRunProgram;

	int numRows;
	int numCols;
	int numNodes;

	int trainingCycles;
	float learningRate;
	int radius;

	int numFeatures;
	int k_nearest;
	int numClusters;
	int numClusterings;
	int maxCentroidMovement;

	List<Sample> unknownSet;
	List<Sample> knownSet;
	Profile globalProfile;
	List<List<String>> labelNames;

	List<Node> nodesList;
	List<Cluster> clustersList;
};

