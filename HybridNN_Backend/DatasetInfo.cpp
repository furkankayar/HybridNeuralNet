#include "DatasetInfo.h"
#include <list> 
#include <map>
#include <iostream>
#include <algorithm>

using namespace std;

DatasetInfo::DatasetInfo(vector<vector <float>> data, vector<Type> types):
	data(data),
	types(types){
	this->initTokensAndEntropy();
}
DatasetInfo::DatasetInfo() : DatasetInfo(vector<vector <float>>(0), vector<Type>(0)) {}

vector<vector <float>> DatasetInfo::getData() {
	return this->data;
}

vector<Type> DatasetInfo::getTypes() {
	return this->types;
}

void DatasetInfo::initTokensAndEntropy() {
	if (this->data.size() <= 0 || this->data[0].size() <= 0) {
		throw exception("Dataset is empty");
		this->entropy = 0;
		this->tokens = {};
		return;
	}
	this->entropy = 0;
	this->tokens = {};
	map<float, int> frequencies;
	int target_feature_index = this->data[0].size() - 1;
	for (size_t row = 0; row < this->data.size(); row++) {
		frequencies[this->data[row][target_feature_index]]++; 
	}
	map<float, int>::iterator it = frequencies.begin();
	while (it != frequencies.end()) {
		float p = (float)it->second / this->data.size();
		if (p > 0) {
			this->entropy -= p * (log(p) / log(2));
		}
		this->tokens.push_back(it->first);
		it++;
	}
	/*cout << "entropy: " << this->entropy << endl;
	cout << "tokens: ";
	for (float token : tokens) {
		cout << token << " ";
	}
	cout << endl;*/
}

list<float> DatasetInfo::getTokens() {
	return this->tokens;
}

float DatasetInfo::getEntropy() {
	return this->entropy;
}

void DatasetInfo::print() {
	cout << "Dataset: " << endl;
	for (unsigned int i = 0; i < this->data.size(); i++) {
		cout << i << " - ";
		for (unsigned int j = 0; j < this->data[0].size(); j++) {
			cout << this->data[i][j] << " ";
		}
		cout << endl;
	}
}


void DatasetInfo::setDatasetType(int featureOrder, Type type) {
	this->types[featureOrder] = type;
}

