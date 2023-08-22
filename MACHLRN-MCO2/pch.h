#pragma once

// standard and imported libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>
#include <armadillo>


// templates
template <typename T> using List = std::vector<T>;                            // static list; but main purpose is for computations
template <typename T> using Vector = arma::Col<T>;                            // dynamic list; but with smaller memory size
template <typename T> using Matrix = arma::Mat<T>;                            // static matrix; but mainly for computations
template <typename T, typename U> using Pair = std::pair<T, U>;       
template <typename T, typename U> using Dictionary = std::unordered_map<T, U>;
template <typename T, typename U> using SortedDictionary = std::map<T, U>;


// typedefs
typedef std::string String;
typedef Dictionary<String, Dictionary<String, int>> Profile;                  // 1st Dictionary - contains the list of all the characteristics; 
																			  //                - first value: name of the characteristic (e.g. Sex)
																			  //				- second value: demographic of that characteristic
                                                                              // 2nd Dictionary - contains the list of all subpopulations within a demographic
																			  //                - first value: name of the specific characteristic (e.g. Male)
																			  //                - second value: number of people that share the same characteristic

// self-made scripts
#include "Sample.h"
 

// global methods
template<typename Base, typename T>
inline bool InstanceOf(const T* ptr) {
	return dynamic_cast<const Base*>(ptr) != nullptr;
}

template<typename T>
inline float RoundOf(const T& value) {
	int integer = value * 100;
	return integer / 100.f;
}

inline Vector<float> RoundOfVector(Vector<float> floatVec) {
	Vector<int> intVec = Vector<int>(floatVec.n_elem);
	for (int i = 0; i < floatVec.n_elem; i++)
	{
		intVec(i) = floatVec(i) * 10000;
		floatVec(i) = intVec(i) / 10000.f;
	}
	return floatVec;
}