#pragma once


#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>
#include <armadillo>


template <typename T> using List = std::vector<T>;
template <typename T> using Matrix = std::vector<std::vector<T>>;
template <typename T, typename U> using Pair = std::pair<T, U>;
template <typename T, typename U> using SortedHashMap = std::map<T, U>;


template<typename Base, typename T>
inline bool instanceof(const T* ptr) {
	return dynamic_cast<const Base*>(ptr) != nullptr;
}


typedef std::string String;
typedef List<int>* Sample;