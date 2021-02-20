#include "Edge.h"
#include "DatasetInfo.h"
#include "Node.h"

Node::Node() : Node(NULL, -1, -1) {}

Node::Node(DatasetInfo* dataset, int selectiveFeatureOder) : Node(dataset, selectiveFeatureOder, -1){}

Node::Node(DatasetInfo* dataset, int selectiveFeatureOrder, float threshold){
	this->dataset = dataset;
	this->selectiveFeatureOrder = selectiveFeatureOrder;
	this->threshold = threshold;
}

list<Edge> Node::getEdges() {
	return this->edges;
}

DatasetInfo* Node::getDataset() {
	return this->dataset;
}

int Node::getSelectiveFeatureOrder() {
	return this->selectiveFeatureOrder;
}

void Node::setSelectiveFeatureOrder(int selectiveFeatureOrder) {
	this->selectiveFeatureOrder = selectiveFeatureOrder;
}

float Node::getThreshold() {
	return this->threshold;
}

void Node::setThreshold(float threshold) {
	this->threshold = threshold;
}