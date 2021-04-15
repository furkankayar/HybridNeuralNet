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
	this->maxTreeDepth = 1;
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
	this->root->setName("R");
	if (types[featureOrder] == CATEGORICAL) {
		this->root->getDataset()->setDatasetType(featureOrder, NOT_AVAILABLE);
		this->splitCategorical(this->root, featureOrder);
	}
	else if (types[featureOrder] == CONTINUOUS) {
		//this->root->getDataset()->setDatasetType(featureOrder, NOT_AVAILABLE);
		this->calculateBestInformationGainContinuousFeature(this->root, featureOrder);
		this->splitContinuous(this->root, featureOrder);
	}

}

void DecisionTree::printTree(Node* node) {
	if (node->getEdges().size() == 0) {
		cout << "\n------" << endl;
		cout << "LEAF" << endl;
		cout << "NAME " << node->getName() << endl;
		//cout << "FEATURE " << node->getSelectiveFeatureOrder() << endl;
		//cout << "THRESHOLD " << node->getThreshold() << endl;
		cout << "LEVEL " << node->getLevel() << endl;
		cout << "CLASS " << node->getClass() << endl;
		node->getDataset()->print();
		return;
	}
	
	cout << "\n------" << endl;
	cout << "INTERNAL" << endl;
	cout << "NAME " << node->getName() << endl;
	cout << "FEATURE " << node->getSelectiveFeatureOrder() << endl;
	cout << "THRESHOLD " << node->getThreshold() << endl;
	cout << "LEVEL " << node->getLevel() << endl;
	node->getDataset()->print();
	for (Edge* edge : node->getEdges()) {
		cout << "\n----" << endl;
		cout << "FROM " << node->getName() << " TO " << edge->getTarget()->getName() << " WEIGHT: " << edge->getInfoGain() << endl;
		cout << "----" << endl;
		printTree(edge->getTarget());
	}
}

void DecisionTree::getNodesWithLevel(Node* node, int level, list<Node*>& nodes) {
	if (node->getLevel() == level) {
		nodes.push_back(node);
		return;
	}

	for (Edge* edge : node->getEdges()) {
		getNodesWithLevel(edge->getTarget(), level, nodes);
	}
}

void DecisionTree::moveLeafNodes(Node* node, int newLevel) {
	if (node->getEdges().size() == 0) {
		node->setLevel(newLevel);
	}

	for (Edge* edge : node->getEdges()) {
		moveLeafNodes(edge->getTarget(), newLevel);
	}
}

void DecisionTree::buildTree(Node* node, int depth) {

	node->setLevel(depth);

	if (depth > this->maxTreeDepth) {
		this->maxTreeDepth = depth;
	}

	if (isAllSameClass(node) || depth == 4) {
		calculateBestFeatureOrder(node);
		return;
	}

	vector<Type> types = node->getDataset()->getTypes();
	int bestFeatureOrder = calculateBestFeatureOrder(node); 
	node->setSelectiveFeatureOrder(bestFeatureOrder);

	if (bestFeatureOrder == -1) {
		return;
	}

	if (types[bestFeatureOrder] == CATEGORICAL){
		node->getDataset()->setDatasetType(bestFeatureOrder, NOT_AVAILABLE);
		splitCategorical(node, bestFeatureOrder);
	}
	else if (types[bestFeatureOrder] == CONTINUOUS) {
		//node->getDataset()->setDatasetType(bestFeatureOrder, NOT_AVAILABLE);
		splitContinuous(node, bestFeatureOrder);
	}

	int count = 0;
	for (Edge* edge : node->getEdges()) {
		Node* child = edge->getTarget();
		child->setName(node->getName() + "-" + to_string(count));
		buildTree(child, depth + 1);
		count++;
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
	float bestInfoGain = -1.0f;
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

	Edge* edge = findEdge(this->root, node);
	if (edge != NULL) {
		edge->setInfoGain(bestInfoGain);
	}

	return bestOrder;
}

Edge* DecisionTree::findEdge(Node* node, Node* target) {

	if (node->getEdges().size() == 0) {
		return NULL;
	}

	Edge* e = NULL;
	for (Edge* edge : node->getEdges()) {
		if (edge->getTarget() == target) {
			return edge;
		}
		Edge* temp = findEdge(edge->getTarget(), target);
		if (temp != NULL) {
			e = temp;
		}
	}

	return e;
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

void DecisionTree::splitCategorical(Node* node, int featureOrder) {
	vector<vector <float>> data = node->getDataset()->getData();

	list<float> usedTokens;
	for (unsigned long i = 0; i < data.size(); i++) {
		if (find(usedTokens.begin(), usedTokens.end(), data[i][featureOrder]) == usedTokens.end()) {
			float item = data[i][featureOrder];
			vector<vector <float>> sub_data;
			copy_if(data.begin(), data.end(), back_inserter(sub_data), [featureOrder, item](vector<float> i) {return i[featureOrder] == item; });
			usedTokens.push_back(item);
			Node* newNode = new Node(new DatasetInfo(sub_data, node->getDataset()->getTypes()));
			node->addEdge(new Edge(newNode));
		}
	}

}

float DecisionTree::calculateBestInformationGainContinuousFeature(Node* node, int featureOrder) {
	vector<vector <float>> data = node->getDataset()->getData();
	list<float> thresholds = calculateThresholds(node, featureOrder);
	unsigned long totalInstanceCount = data.size();
	int targetFeature = data[0].size() - 1;
	float bestInformationGain = -1.0f;
	float bestThreshold = -1.0f;
	int bestTotalInstanceLT = -1;
	int bestTotalInstanceGTE = -1;
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
		if (informationGain >= bestInformationGain) {
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

	vector<vector <float>> data = node->getDataset()->getData();
	std::sort(data.begin(), data.end(),
		[&](const vector<float>& a, const vector<float>& b) {
			return a[featureOrder] < b[featureOrder];
		});

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

void DecisionTree::setMaxTreeDepth(int maxTreeDepth) {
	this->maxTreeDepth = maxTreeDepth;
}

int DecisionTree::getMaxTreeDepth() {
	return this->maxTreeDepth;
}

void DecisionTree::initializeNonAssignedWeights() {
	float min = 1.0f;
	float max = 0.0f;
	srand(time(0));
	getMinAndMaxWeights(this->root, &min, &max);
	assignRandomWeights(this->root, min, max);

}

void DecisionTree::getMinAndMaxWeights(Node* node, float* min, float* max) {
	if (node->getEdges().size() == 0) {
		return;
	}

	for (Edge* edge : node->getEdges()) {
		if (edge->getInfoGain() >= 0.0f && edge->getInfoGain() < *min) {
			*min = edge->getInfoGain();
		}
		else if (edge->getInfoGain() >= 0.0f && edge->getInfoGain() > *max) {
			*max = edge->getInfoGain();
		}
		getMinAndMaxWeights(edge->getTarget(), min, max);
	}
}

void DecisionTree::assignRandomWeights(Node* node, float min, float max) {
	if (node->getEdges().size() == 0) {
		return;
	}

	for (Edge* edge : node->getEdges()) {
		if (edge->getInfoGain() < min) {
			float random = ((float)rand()) / (float)RAND_MAX;
			edge->setInfoGain(min + random * (max - min));
		}
		assignRandomWeights(edge->getTarget(), min, max);
	}
}
