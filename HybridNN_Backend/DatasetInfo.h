#pragma once
#ifndef DATASETINFO_H
#define DATASETINFO_H
#include <list>
#include <vector>

using namespace std;

class DatasetInfo {
private:
	vector<vector <float>> data;
	float entropy;
	list<float> tokens;
public:
	DatasetInfo();
	DatasetInfo(vector<vector<float>> data);
	vector<vector<float>> getData();
	void initTokensAndEntropy();
	list<float> getTokens();
};

#endif 