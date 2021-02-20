#pragma once
#ifndef DATASETINFO_H
#define DATASETINFO_H
#include <list>

using namespace std;

class DatasetInfo {
private:
	float* data;
	float entropy;
	list<float> tokens;
	int columnSize;
	int rowSize;
public:
	DatasetInfo();
	DatasetInfo(float* data, int columnSize, int rowSize);
	int getColumnSize();
	int getRowSize();
	float getItem(int posX, int posY);
	void initTokensAndEntropy();
	list<float> getTokens();
};

#endif 