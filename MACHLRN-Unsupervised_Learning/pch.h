#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>



template <typename T> using List = std::vector<T>;
template <typename T> using Matrix = std::vector<std::vector<T>>;
template <typename T, typename U> using Pair = std::pair<T, U>;


struct Vector2f
{
	float x;
	float y;

	Vector2f() : x(0), y(0)
	{

	}

	Vector2f(const float& x, const float& y) : x(x), y(y)
	{

	}

	void operator=(const Vector2f& RHS)
	{
		this->x = RHS.x;
		this->y = RHS.y;
	}

	Vector2f operator+(const Vector2f& RHS)
	{
		Vector2f newVec;
		newVec.x = this->x + RHS.x;
		newVec.y = this->y + RHS.y;
		return newVec;
	}

	Vector2f operator-(const Vector2f& RHS)
	{
		Vector2f newVec;
		newVec.x = this->x - RHS.x;
		newVec.y = this->y - RHS.y;
		return newVec;
	}

	Vector2f operator*(const Vector2f& RHS)
	{
		Vector2f newVec;
		newVec.x = this->x * RHS.x;
		newVec.y = this->y * RHS.y;
		return newVec;
	}

	Vector2f operator/(const Vector2f& RHS)
	{
		Vector2f newVec;
		newVec.x = RHS.x != 0 ? this->x / RHS.x : throw std::overflow_error("Divide by zero exception");
		newVec.y = RHS.y != 0 ? this->y / RHS.y : throw std::overflow_error("Divide by zero exception");
		return newVec;
	}

	bool operator==(const Vector2f& RHS)
	{
		if (this->x != RHS.x || this->y != RHS.y)
			return false;

		return true;
	}

	bool operator!=(const Vector2f& RHS)
	{
		if (this->x == RHS.x && this->y == RHS.y)
			return false;

		return true;
	}
};

struct Cluster
{
	Vector2f centroid;
	List<Vector2f> samplesList;

	Cluster()
	{
		centroid = Vector2f();
		samplesList.clear();
	}

	Cluster(const Vector2f& centroid, const List<Vector2f>& samplsList) : centroid(centroid), samplesList(samplesList)
	{

	}

	void operator=(const Cluster& RHS)
	{
		this->centroid = RHS.centroid;
		this->samplesList = RHS.samplesList;
	}

	bool operator!=(const Cluster& RHS)
	{
		if (this->centroid == RHS.centroid)
			return false;

		if (this->samplesList.size() == RHS.samplesList.size())
		{
			for (int i = 0; i < this->samplesList.size(); i++)
			{
				if (this->samplesList[i] == RHS.samplesList[i])
					return false;
			}
		}		

		return true;
	}
};

struct ClusteringsWithScore
{
	List<Cluster> clustering;
	float dbiScore;

	ClusteringsWithScore()
	{
		clustering = List<Cluster>();
		dbiScore = 0;
	}

	ClusteringsWithScore(const List<Cluster>& clustering, const float& dbiScore) : clustering(clustering), dbiScore(dbiScore)
	{

	}
};


struct DistanceFromCentroid
{
	float distance;
	Vector2f centroid;

	DistanceFromCentroid()
	{
		distance = 0;
		centroid = Vector2f();
	}

	DistanceFromCentroid(const float& distance, const Vector2f& centroid) : distance(distance), centroid(centroid)
	{

	}

	void operator=(const DistanceFromCentroid& RHS)
	{
		this->distance = RHS.distance;
		this->centroid = RHS.centroid;
	}

	bool operator<(const DistanceFromCentroid& RHS) const
	{
		return (this->distance < RHS.distance);
	}
};