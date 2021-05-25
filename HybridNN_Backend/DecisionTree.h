#pragma once
#ifndef DECISIONTREE_H
#define DECISIONTREE_H

class Node;
class Edge;
class DatasetInfo;

class DecisionTree {
private:
	Node* root;
	int maxTreeDepth;
	int acceptableMaxDepth;
public:
	DecisionTree();
	DecisionTree(Node* root, int acceptableMaxDepth);
	Node* getRoot();
	void setRoot(Node* root);
	int getMaxTreeDepth();
	void setMaxTreeDepth(int maxTreeDepth);
	void splitRootNode();
	float calculateBestInformationGainContinuousFeature(Node* node, int featureOrder);
	float calculateBestInformationGainCategoricalFeature(Node* node, int featureOrder);
	void splitContinuous(Node* node, int featureOrder);
	void splitCategorical(Node* node, int featureOrder);
	int calculateBestFeatureOrder(Node* node);
	list<float> calculateThresholds(Node* node, int featureOrder);
	void buildTree(Node* node, int depth);
	bool isAllSameClass(Node* node);
	void printTree(Node* node);
	void moveLeafNodes(Node* node, int newLevel);
	void getNodesWithLevel(Node* node, int level, list<Node*>& nodes);
	Edge* findEdge(Node* node, Node* target);
	void getMinAndMaxWeights(Node* node, float* min, float* max);
	void initializeNonAssignedWeights();
	void assignRandomWeights(Node* node, float min, float max);
};

#endif