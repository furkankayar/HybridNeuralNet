#include "Edge.h"
#include "DatasetInfo.h"
#include "Node.h"

Node::Node() : Node(NULL, 0, 0, -1, -1) {}

Node::Node(float* data, int X, int Y, int selectiveFeatureOrder, float threshold){
	this->dataset = new DatasetInfo(data, X, Y);
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