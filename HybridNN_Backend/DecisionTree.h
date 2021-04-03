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
	int calculateBestFeatureOrder(Node* node);
	list<float> calculateThresholds(Node* node, int featureOrder);
	void buildTree(Node* node);
	bool isAllSameClass(Node* node);
	void printTree(Node* node);
};

#endif