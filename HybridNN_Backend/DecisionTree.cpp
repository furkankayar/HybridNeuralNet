#include "Edge.h"
#include "Node.h"
#include "DatasetInfo.h"
#include "DecisionTree.h"

#include <iostream>
#include <algorithm>
#include <map>
#include <utility>
#include <numeric>

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

void DecisionTree::splitRootNode() {
	int featureOrder = this->root->getSelectiveFeatureOrder();
	vector<Type> types = this->root->getDataset()->getTypes();
	
	if (types[featureOrder] == CATEGORICAL) {
		this->root->getDataset()->setDatasetType(featureOrder, NOT_AVAILABLE);
		cout << this->calculateBestInformationGainCategoricalFeature(this->root, featureOrder) << endl;
	}
	else if (types[featureOrder] == CONTINUOUS) {
		this->root->getDataset()->setDatasetType(featureOrder, NOT_AVAILABLE);
		this->calculateBestInformationGainContinuousFeature(this->root, featureOrder);
		this->splitContinuous(this->root, featureOrder);
		/*
		for (Edge* edge : this->root->getEdges()) {
			cout << calculateBestFeatureOrder(edge->getTarget()) << endl;			
		}
		*/
	}

}

void DecisionTree::printTree(Node* node) {
	if (node->getEdges().size() == 0) {
		node->getDataset()->print();
		return;
	}
	
	for (Edge* edge : node->getEdges()) {
		printTree(edge->getTarget());
	}
}

void DecisionTree::buildTree(Node* node) {

	if (isAllSameClass(node)) {
		return;
	}
	
	vector<Type> types = node->getDataset()->getTypes();
	int bestFeatureOrder = calculateBestFeatureOrder(node); 
	if (types[bestFeatureOrder] == CATEGORICAL){
		node->getDataset()->setDatasetType(bestFeatureOrder, NOT_AVAILABLE);
	}
	else if (types[bestFeatureOrder] == CONTINUOUS) {
		node->getDataset()->setDatasetType(bestFeatureOrder, NOT_AVAILABLE);
		splitContinuous(node, bestFeatureOrder);
	}

	for (Edge* edge : node->getEdges()) {
		buildTree(edge->getTarget());
	}
}

bool DecisionTree::isAllSameClass(Node* node) {
	vector<vector<float>> dataset = node->getDataset()->getData();
	int target = dataset[0].size() - 1;
	float firstClass = dataset[0][target];
	for (long i = 1; i < dataset.size(); i++) {
		if (firstClass != dataset[i][target]) {
			return false;
		}
	}
	return true;
}

int DecisionTree::calculateBestFeatureOrder(Node* node) {
	vector<vector <float>> data = node->getDataset()->getData();
	vector<Type> types = node->getDataset()->getTypes();
	float bestInfoGain = 0.0f;
	int bestOrder = -1;
	for (size_t i = 0; i < types.size() - 1; i++) {
		if (types[i] == NOT_AVAILABLE) {
			continue;
		} 
		else if (types[i] == CONTINUOUS) {
			float infoGain = calculateBestInformationGainContinuousFeature(node, i);
			if (infoGain >= bestInfoGain) {
				bestInfoGain = infoGain;
				bestOrder = i;
			}
		}
		else if (types[i] == CATEGORICAL) {
			float infoGain = calculateBestInformationGainCategoricalFeature(node, i);
			if (infoGain >= bestInfoGain) {
				bestInfoGain = infoGain;
				bestOrder = i;
			}
		}
	}
	if (types[bestOrder] == CONTINUOUS) { //TODO burayý düzenleyebilirsin. bu fonksiyon node'un içerisindeki LT GTE gibi deðerleri atadýðý için yukarýda en iyisini bulurken bu deðerler en son order için hesaplanmýþ halde kalýyor. bu yüzden burada tekrardan best order için yeniden hesaplanýp atanmasý gerekiyor.
		calculateBestInformationGainContinuousFeature(node, bestOrder);
	}
	return bestOrder;
}

void DecisionTree::splitContinuous(Node* node, int featureOrder) {

	vector<vector <float>> data = node->getDataset()->getData();
	float threshold = node->getThreshold();
	vector<vector <float>> dataLT(node->getNumberOfLT(), vector<float>(data[0].size()));
	vector<vector <float>> dataGTE(node->getNumberOfGTE(), vector<float>(data[0].size()));
	unsigned long pointerLT = 0;
	unsigned long pointerGTE = 0;

	for (unsigned long i = 0; i < data.size(); i++) {
		if (data[i][featureOrder] < threshold) {
			dataLT[pointerLT++] = data[i];
		}
		else {
			dataGTE[pointerGTE++] = data[i];
		}
	}

	Node* nodeLT = new Node(new DatasetInfo(dataLT, node->getDataset()->getTypes()));
	Node* nodeGTE = new Node(new DatasetInfo(dataGTE, node->getDataset()->getTypes()));
	node->addEdge(new Edge(nodeLT));
	node->addEdge(new Edge(nodeGTE));
}

float DecisionTree::calculateBestInformationGainContinuousFeature(Node* node, int featureOrder) {
	vector<vector <float>> data = node->getDataset()->getData();
	list<float> thresholds = calculateThresholds(node, featureOrder);
	unsigned long totalInstanceCount = data.size();
	int targetFeature = data[0].size() - 1;
	float bestInformationGain = 0.0f;
	float bestThreshold = 0.0f;
	int bestTotalInstanceLT = 0;
	int bestTotalInstanceGTE = 0;
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
		float informationGain = node->getDataset()->getEntropy() - (((float)totalInstanceLT / totalInstanceCount * partitionEntropyLT) + ((float)totalInstanceGTE / totalInstanceCount * partitionEntropyGTE));
		if (informationGain > bestInformationGain) {
			bestInformationGain = informationGain;
			bestThreshold = threshold;
			bestTotalInstanceLT = totalInstanceLT;
			bestTotalInstanceGTE = totalInstanceGTE;
		}
	}

	node->setThreshold(bestThreshold);
	node->setNumberOfGTE(bestTotalInstanceGTE);
	node->setNumberOfLT(bestTotalInstanceLT);
	return bestInformationGain;
}

list<float> DecisionTree::calculateThresholds(Node* node, int featureOrder) {
	node->getDataset()->sort(featureOrder);

	vector<vector <float>> data = node->getDataset()->getData();

	unsigned long Y = node->getDataset()->getData().size();
	unsigned short X = node->getDataset()->getData()[0].size();
	unsigned short targetFeature = X - 1;

	list<float> thresholds;
	for (unsigned long i = 0; i < Y - 1; i++) {
		if (data[i][targetFeature] != data[i + 1][targetFeature]) {
			thresholds.push_back((data[i][featureOrder] + data[i + 1][featureOrder]) / 2);
		}
	}
	thresholds.unique();

	return thresholds;
}

float DecisionTree::calculateBestInformationGainCategoricalFeature(Node* node, int featureOrder) {
	vector<vector <float>> data = node->getDataset()->getData();
	list<float> tokens = node->getDataset()->getTokens();
	unsigned long totalInstanceCount = data.size();
	int targetFeature = data[0].size() - 1;
	
	map<float, map<float, int>> items;
	for (unsigned long i = 0; i < totalInstanceCount; i++) {
		items[data[i][featureOrder]][data[i][targetFeature]]++;
	}

	map<float, map<float, int>>::iterator items_it = items.begin();
	float remainder = 0.0f;
	while (items_it != items.end()) {
		int total = accumulate(begin(items[items_it->first]), end(items[items_it->first]), 0,
			[](const size_t previous, const pair<const float, size_t>& p) {
				return previous + p.second; 
			});
		float entropy = 0.0f;
		map<float, int>::iterator values_it = items_it->second.begin();
		while (values_it != items_it->second.end()) {
			entropy -= ((float)values_it->second / (float)total) * log((float)values_it->second / (float)total) / log(2);
			values_it++;
		}
		remainder += ((float)total / (float)totalInstanceCount) * entropy;
		items_it++;
	}
	
	return node->getDataset()->getEntropy() - remainder;
}