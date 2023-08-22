#pragma once


class KMeansSimulator
{
public:
	KMeansSimulator();

	bool Initialize(int numFeatures, int maxIterations, std::string directory);
	void PerformClustering();
	void PrintResult();

private:
	void ConvertRawSamplesIntoPairs(const Matrix<float>& rawSamples);

	void RandomizeInitialCentroids(const int& k, List<Vector2f>& centroidsList, List<Cluster>& clusterings);
	void CreateClustersFromSamples(const List<Vector2f>& centroidsList, List<Cluster>& clusterings);
	void RecalculateCentroids(List<Cluster>& clusterings);
	
	void GetBestCluster(const List<List<Cluster>>& clusteringsList);
	float CalculateDBIScore(const List<Cluster>& clustering);
	float GetHighestSimilarityScore(const Cluster& clusterToEvaluate, const List<Cluster>& clusterings);
	float GetSimilarityScore(const Cluster& cluster1, const Cluster& cluster2);


private:
	bool canPerformSimulation;
	int numFeatures;
	int maxIterations;

	List<int> kList;
	List<Vector2f> samplesList;
	List<ClusteringsWithScore> dbiScoresList;
};