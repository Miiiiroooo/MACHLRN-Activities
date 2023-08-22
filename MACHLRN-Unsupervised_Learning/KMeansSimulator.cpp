#include "pch.h"
#include "KMeansSimulator.h"
#include "CSVHandler.h"

KMeansSimulator::KMeansSimulator()
{
	this->canPerformSimulation = false;
	this->numFeatures = 2;
	this->maxIterations = 5;
}

bool KMeansSimulator::Initialize(int numFeatures, int maxIterations, std::string directory)
{
	CSVHandler csvHandler = CSVHandler();

	if (csvHandler.RetrieveDataFromDirectory(directory, numFeatures))
	{
		ConvertRawSamplesIntoPairs(csvHandler.GetSamples());
		this->numFeatures = numFeatures; 
		this->maxIterations = maxIterations;

		kList.push_back(2);
		kList.push_back(3);

		canPerformSimulation = true;

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

void KMeansSimulator::PerformClustering()
{
	List<List<Cluster>> clusteringsList;

	for (int k : kList)
	{ 
		for (int i = 0; i < maxIterations; i++) 
		{
			List<Vector2f> centroidsList;
			List<Cluster> clusterings;

			RandomizeInitialCentroids(k, centroidsList, clusterings);
			CreateClustersFromSamples(centroidsList, clusterings);
			RecalculateCentroids(clusterings);

			clusteringsList.push_back(clusterings);
		}
	}

	GetBestCluster(clusteringsList);
}

void KMeansSimulator::PrintResult()
{
	std::cout << "---------------------------------------------------------\n\n";
	std::cout << "The final clustering is as follows:\n";

	for (Cluster bestCluster : dbiScoresList[0].clustering)
	{
		std::cout << "Centroid: (" << bestCluster.centroid.x << ", " << bestCluster.centroid.y << ")\nSamples:\n";

		for (Vector2f point : bestCluster.samplesList)
		{
			std::cout << "(" << point.x << ", " << point.y << ")\n";
		}

		std::cout << "\n";
	}
}

void KMeansSimulator::ConvertRawSamplesIntoPairs(const Matrix<float>& rawSamples)
{
	for (List<float> sample : rawSamples) 
	{
		Vector2f newSample = Vector2f(sample[0], sample[1]); 
		samplesList.push_back(newSample);
	}
}

void KMeansSimulator::RandomizeInitialCentroids(const int& k, List<Vector2f>& centroidsList, List<Cluster>& clusterings)
{
	List<int> indicesList;
	std::cout << "---------------------------------------------------------\n\n";
	std::cout << "At K = " << k << "\nInitial Centroid List:\n";

	do
	{
		int randIndex = rand() % samplesList.size(); 

		if (std::find(indicesList.begin(), indicesList.end(), randIndex) == indicesList.end()) 
		{
			centroidsList.push_back(samplesList[randIndex]);  
			indicesList.push_back(randIndex); 

			Cluster newCluster = Cluster(samplesList[randIndex], List<Vector2f>());
			clusterings.push_back(newCluster);
		}
	} while (centroidsList.size() != k); 

	for (int i = 0; i < centroidsList.size(); i++) 
	{
		std::cout << "(" << centroidsList[i].x << ", " << centroidsList[i].y << "), "; 
	}
}

void KMeansSimulator::CreateClustersFromSamples(const List<Vector2f>& centroidsList, List<Cluster>& clusterings)
{
	for (Vector2f point : samplesList)  
	{
		List<DistanceFromCentroid> distanceFromCentroidsList; 

		for (Vector2f centroid : centroidsList)
		{
			float squaredDistance = std::pow((point.x - centroid.x), 2) + std::pow((point.y - centroid.y), 2);

			DistanceFromCentroid newInstance = DistanceFromCentroid(std::sqrt(squaredDistance), centroid);
			distanceFromCentroidsList.push_back(newInstance);
		}

		std::sort(distanceFromCentroidsList.begin(), 
			distanceFromCentroidsList.end(), 
			[](DistanceFromCentroid& x, DistanceFromCentroid& y) {
				return (x.distance < y.distance); 
			});

		Vector2f closestCentroid = distanceFromCentroidsList[0].centroid;

		for (int i = 0; i < clusterings.size(); i++)
		{
			if (closestCentroid == clusterings[i].centroid)
			{
				clusterings[i].samplesList.push_back(point);

				break;
			}
		}
	}
}

void KMeansSimulator::RecalculateCentroids(List<Cluster>& clusterings)
{
	std::cout << "\n\nFinal Centroid List:\n";

	for (int i = 0; i < clusterings.size(); i++)
	{
		Cluster cluster = clusterings[i];
		float avgX = 0;
		float avgY = 0;

		for (Vector2f point : cluster.samplesList)
		{
			avgX += point.x;
			avgY += point.y;
		}

		avgX /= cluster.samplesList.size();
		avgY /= cluster.samplesList.size();

		Vector2f newCentroid = Vector2f(avgX, avgY);
		clusterings[i].centroid = newCentroid;
	}

	for (int i = 0; i < clusterings.size(); i++)
	{
		std::cout << "(" << clusterings[i].centroid.x << ", " << clusterings[i].centroid.y << "), ";
	}

	std::cout << "\n\n";
}

void KMeansSimulator::GetBestCluster(const List<List<Cluster>>& clusteringsList)
{
	for (List<Cluster> clustering : clusteringsList)
	{
		float score = CalculateDBIScore(clustering);
		ClusteringsWithScore newInstance = ClusteringsWithScore(clustering, score);
		dbiScoresList.push_back(newInstance);
	}

	std::sort(dbiScoresList.begin(),
		dbiScoresList.end(),
		[](ClusteringsWithScore& x,ClusteringsWithScore& y) {
			return (x.dbiScore < y.dbiScore);
		});
}

float KMeansSimulator::CalculateDBIScore(const List<Cluster>& clustering)
{
	float score = 0;

	for (int i = 0; i < clustering.size(); i++)
	{
		score += GetHighestSimilarityScore(clustering[i], clustering);
	}

	score /= clustering.size();

	return score;
}

float KMeansSimulator::GetHighestSimilarityScore(const Cluster& clusterToEvaluate, const List<Cluster>& clusterings)
{
	List<float> similarityScoresList;

	for (Cluster cluster : clusterings)
	{
		if (cluster != clusterToEvaluate)
		{
			similarityScoresList.push_back(GetSimilarityScore(cluster, clusterToEvaluate));
		}
	}

	std::sort(similarityScoresList.begin(), similarityScoresList.end(), std::greater<float>());

	return similarityScoresList[0];
}

float KMeansSimulator::GetSimilarityScore(const Cluster& cluster1, const Cluster& cluster2)
{
	// Get Intra Cluster Distance Score of te First Cluster
	float interCluster1 = 0;
	for (Vector2f point : cluster1.samplesList)
	{
		float squaredDistance = std::pow((point.x, cluster1.centroid.x), 2) + std::pow((point.y, cluster1.centroid.y), 2);
		interCluster1 += std::sqrt(squaredDistance); 
	}
	interCluster1 /= cluster1.samplesList.size();

	// Get Intra Cluster Distance Score of te Second Cluster
	float interCluster2 = 0;
	for (Vector2f point : cluster2.samplesList)
	{
		float squaredDistance = std::pow((point.x, cluster2.centroid.x), 2) + std::pow((point.y, cluster2.centroid.y), 2);
		interCluster2 += std::sqrt(squaredDistance);
	}
	interCluster2 /= cluster2.samplesList.size();

	// Get the Distance Between Both of the Clusters
	float squaredDistance = std::pow((cluster1.centroid.x, cluster2.centroid.x), 2) + std::pow((cluster1.centroid.y, cluster2.centroid.y), 2);
	float betweenClusters = std::sqrt(squaredDistance);

	return ((interCluster1 + interCluster2) / betweenClusters);
}