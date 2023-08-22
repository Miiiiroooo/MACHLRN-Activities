#pragma once

struct Node
{
	Vector<float> weights;

	List<Sample> samplesList;
	List<Sample> nearestSamplesList;
	List<int> dominantLabelsList;

	int clusterID;

	void DetermineDominantLabels();
};

