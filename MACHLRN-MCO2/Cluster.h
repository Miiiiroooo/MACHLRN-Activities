#pragma once

#include "Node.h"

struct Cluster
{
	int id;
	Vector<float> centroid;
	List<Node*> nodesList;
	Profile profile;

	Cluster(int id);

	void DetermineLocalProfile(const List<List<String>>& labelNames);
};

