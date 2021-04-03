#pragma once
#ifndef DATASETINFO_H
#define DATASETINFO_H
#include <list>
#include <vector>

using namespace std;

enum Type {CONTINUOUS=1, CATEGORICAL=0, NOT_AVAILABLE=-1};

class DatasetInfo {
private:
	vector<vector <float>> data;
	float entropy;
	vector<Type> types;
	list<float> tokens;
public:
	DatasetInfo();
	DatasetInfo(vector<vector<float>> data, vector<Type> types);
	vector<vector<float>> getData();
	vector<Type> getTypes();
	void initTokensAndEntropy();
	list<float> getTokens();
	float getEntropy();
	void sort(static int featureOrder);
	void print();
	void setDatasetType(int featureOrder, Type type);
};

#endif 