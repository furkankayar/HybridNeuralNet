#include "DatasetInfo.h"
#include <list> 
#include <map>
#include <iostream>

using namespace std;

DatasetInfo::DatasetInfo(float* data, int columnSize, int rowSize) : 
	data(data), 
	columnSize(columnSize),
	rowSize(rowSize){
	this->initTokensAndEntropy();
}
DatasetInfo::DatasetInfo() : DatasetInfo(NULL, 0, 0) {}

int DatasetInfo::getColumnSize() {
	return this->columnSize;
}
int DatasetInfo::getRowSize() {
	return this->rowSize;
}
float DatasetInfo::getItem(int posX, int posY) {
	return this->data[posX * this->rowSize + posY];
}

void DatasetInfo::initTokensAndEntropy() {
	if (this->columnSize <= 0 || this->rowSize <= 0) {
		throw exception("Dataset is empty");
	}
	this->entropy = 0;
	this->tokens = {};
	map<float, int> frequencies;
	int target_feature_index = this->getColumnSize() - 1;
	for (size_t row = 0; row < this->getRowSize(); row++) {
		frequencies[this->getItem(target_feature_index, row)]++;
	}
	map<float, int>::iterator it = frequencies.begin();
	while (it != frequencies.end()) {
		float p = (float)it->second / this->getRowSize();
		if (p > 0) {
			this->entropy -= p * (log(p) / log(2));
		}
		this->tokens.push_back(it->first);
		it++;
	}
	cout << "entropy: " << this->entropy << endl;
	cout << "tokens: ";
	for (float token : tokens) {
		cout << token << " ";
	}
	cout << endl;
}
list<float> DatasetInfo::getTokens() {
	return this->tokens;
}
