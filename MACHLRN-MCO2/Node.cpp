#include "pch.h"
#include "Node.h"


void Node::DetermineDominantLabels()
{
	List<Dictionary<int, int>> labels(5);

	// tally the labels
	for (Sample& sample : nearestSamplesList)
	{
		for (int i = 0; i < sample.labelsList.size(); i++)
		{
			int value = sample.labelsList[i];
			if (labels[i].find(value) == labels[i].end())
			{
				labels[i][value] = 1;
			}
			else
			{
				labels[i][value]++;
			}
		}
	}

	// get max for each label
	for (int i = 0; i < labels.size(); i++)
	{
		int max = 0;
		int index = 0;
		for (Pair<const int, int>& pair : labels[i])
		{
			if (pair.second > max)
			{
				max = pair.second;
				index = pair.first;
			}
		}
		dominantLabelsList.push_back(index);
	}
}