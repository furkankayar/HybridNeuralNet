#include "Node.h"
#include "DatasetInfo.h"
#include "DecisionTree.h"

#include <iostream>

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
	cout << this->root->getSelectiveFeatureOrder() << endl;
}