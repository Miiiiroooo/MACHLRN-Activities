#pragma once

struct Sample
{
	const Vector<int> featuresList;  
	const List<int> labelsList;

	Sample(const Vector<int>& featuresList, const List<int>& labelsList);
	Sample& operator=(const Sample& s);
};

