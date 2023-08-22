#include "pch.h"
#include "Sample.h"


Sample::Sample(const Vector<int>& featuresList, const List<int>& labelsList)
	: featuresList(featuresList), labelsList(labelsList)
{

}

Sample& Sample::operator=(const Sample& s)
{
	Sample sample(s.featuresList, s.labelsList);
	return sample;
}
