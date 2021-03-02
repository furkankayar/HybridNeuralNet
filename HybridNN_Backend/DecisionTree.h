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
	void splitRootNode();
	float calculateBestInformationGainContinuousFeature(Node* node, int featureOrder);
	float calculateBestInformationGainCategoricalFeature(Node* node, int featureOrder);
	void splitContinuous(Node* node, int featureOrder);
	list<float> calculateThresholds(Node* node, int featureOrder);
};

#endif