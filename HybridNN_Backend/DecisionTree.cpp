#include "Node.h"
#include "DatasetInfo.h"
#include "DecisionTree.h"

#include <iostream>
#include <algorithm>
#include <map>

using namespace std;

DecisionTree::DecisionTree() : DecisionTree(NULL){}

DecisionTree::DecisionTree(Node* root) {
	this->root = root;
}

Node* DecisionTree::getRoot() {
	return this->root;
}

void DecisionTree::setRoot(Node* root) {
	this->root = root;
}

void DecisionTree::splitNode() {
	static int featureOrder = this->root->getSelectiveFeatureOrder();
	this->root->getDataset()->sort(featureOrder);
	//this->root->getDataset()->print();
	
	vector<vector <float>> data = this->root->getDataset()->getData();

	unsigned long Y = this->root->getDataset()->getData().size();
	unsigned short X = this->root->getDataset()->getData()[0].size();
	unsigned short targetFeature = X - 1;

	list<float> thresholds;
	for (unsigned long i = 0; i < Y - 1; i++) {
		if (data[i][targetFeature] != data[i + 1][targetFeature]) {
			thresholds.push_back((data[i][featureOrder] + data[i + 1][featureOrder]) / 2);
		}
	}
	thresholds.unique();
	this->calculateBestInformationGainContinousFeature(thresholds, featureOrder);
}

float DecisionTree::calculateBestInformationGainContinousFeature(list<float> thresholds, int featureOrder) {
	list<float> remainders;
	vector<vector <float>> data = this->root->getDataset()->getData();
	int totalInstanceCount = data.size();
	int targetFeature = data[0].size() - 1;
	float bestInformationGain = 0.0f;
	float bestThreshold = 0.0f;
	for (float threshold : thresholds) {
		int totalInstanceLT = 0;
		int totalInstanceGTE = 0;
		map<float, int> itemsLT;
		map<float, int> itemsGTE;
		for (unsigned long i = 0; i < totalInstanceCount; i++) {
			if (data[i][featureOrder] < threshold) {
				itemsLT[data[i][targetFeature]]++;
				totalInstanceLT++;
			}
			else {
				itemsGTE[data[i][targetFeature]]++;
				totalInstanceGTE++;
			}
		}
		float partitionEntropyLT = 0.0f;
		map<float, int>::iterator itLT = itemsLT.begin();
		while (itLT != itemsLT.end()) {
			partitionEntropyLT -= (float)itLT->second / totalInstanceLT * (log((float)itLT->second / totalInstanceLT) / log(2));
			itLT++;
		}

		float partitionEntropyGTE = 0.0f;
		map<float, int>::iterator itGTE = itemsGTE.begin();
		while (itGTE != itemsGTE.end()) {
			partitionEntropyGTE -= (float)itGTE->second / totalInstanceGTE * (log((float)itGTE->second / totalInstanceGTE) / log(2));
			itGTE++;
		}
		float informationGain = this->root->getDataset()->getEntropy() - (((float)totalInstanceLT / totalInstanceCount * partitionEntropyLT) + ((float)totalInstanceGTE / totalInstanceCount * partitionEntropyGTE));
		if (informationGain > bestInformationGain) {
			bestInformationGain = informationGain;
			bestThreshold = threshold;
		}
	}
	cout << "Best info gain: " << bestInformationGain << endl;
	cout << "Best threshold: " << bestThreshold << endl;
	return bestInformationGain;
}