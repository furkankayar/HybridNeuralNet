#include "DatasetInfo.h"
#include <list> 
#include <map>
#include <iostream>

using namespace std;

DatasetInfo::DatasetInfo(vector<vector <float>> data):
	data(data){
	this->initTokensAndEntropy();
}
DatasetInfo::DatasetInfo() : DatasetInfo(vector<vector <float>>(0)) {}

vector<vector <float>> DatasetInfo::getData() {
	return this->data;
}

void DatasetInfo::initTokensAndEntropy() {
	if (this->data.size() <= 0 || this->data[0].size() <= 0) {
		throw exception("Dataset is empty");
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
