#pragma once
#ifndef DECISIONTREE_H
#define DECISIONTREE_H

class Node;
class DatasetInfo;

class DecisionTree {
private:
	Node* root;
public:
	DecisionTree();
	DecisionTree(Node* root);
	Node* getRoot();
	void setRoot(Node* root);
	void splitNode();
	float calculateBestInformationGainContinousFeature(list<float> thresholds, int featureOrder);
};

#endif