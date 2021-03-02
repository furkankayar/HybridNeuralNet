#include "Edge.h"
#include "DatasetInfo.h"
#include "Node.h"

Node::Node() : Node(NULL, -1, -1) {}

Node::Node(DatasetInfo* dataset) : Node(dataset, -1, -1) {}

Node::Node(DatasetInfo* dataset, int selectiveFeatureOder) : Node(dataset, selectiveFeatureOder, -1){}

Node::Node(DatasetInfo* dataset, int selectiveFeatureOrder, float threshold){
	this->dataset = dataset;
	this->selectiveFeatureOrder = selectiveFeatureOrder;
	this->threshold = threshold;
	this->numberOfLT = 0;
	this->numberOfGTE = 0;
}

list<Edge*> Node::getEdges() {
	return this->edges;
}

void Node::addEdge(Edge* edge) {
	this->edges.push_back(edge);
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

int Node::getNumberOfGTE() {
	return this->numberOfGTE;
}

int Node::getNumberOfLT() {
	return this->numberOfLT;
}

void Node::setNumberOfGTE(int numberOfGTE) {
	this->numberOfGTE = numberOfGTE;
}

void Node::setNumberOfLT(int numberOfLT) {
	this->numberOfLT = numberOfLT;
}
