#include "pch.h"
#include "Cluster.h"

Cluster::Cluster(int id) : id(id)
{
	
}

void Cluster::DetermineLocalProfile(const List<List<String>>& labelNames) 
{
	for (int i = 0; i < labelNames.size(); i++) 
	{
		List<String> labels = labelNames[i];
		Dictionary<String, int> characteristic; 

		for (int j = 1; j < labels.size(); j++)
		{
			characteristic[labels[j]] = 0;
		}

		profile[labels[0]] = characteristic; 
	}

	for (int i = 0; i < labelNames.size(); i++)
	{
		List<String> labels = labelNames[i]; 
		Dictionary<String, int>& characteristic = profile[labels[0]];

		for (int j = 0; j < nodesList.size(); j++)
		{
			Node* node = nodesList[j]; 

			int value = node->dominantLabelsList[i];
			String key = labels[value + 1];
			characteristic[key] += 1;
		}
	}
}
