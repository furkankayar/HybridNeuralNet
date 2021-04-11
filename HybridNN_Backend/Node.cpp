#include "Edge.h"
#include "DatasetInfo.h"
#include "Node.h"
#include <map>
#include <algorithm>

Node::Node() : Node(NULL, -1, -1) {}

Node::Node(DatasetInfo* dataset) : Node(dataset, -1, -1) {}

Node::Node(DatasetInfo* dataset, int selectiveFeatureOder) : Node(dataset, selectiveFeatureOder, -1){}

Node::Node(DatasetInfo* dataset, int selectiveFeatureOrder, float threshold){
	this->dataset = dataset;
	this->selectiveFeatureOrder = selectiveFeatureOrder;
	this->threshold = threshold;
	this->numberOfLT = 0;
	this->numberOfGTE = 0;
	this->level = 0;
	this->name = "";
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

void Node::removeDataset() {
	//free(this->dataset);
}


int Node::getLevel() {
	return this->level;
}

void Node::setLevel(int level) {
	this->level = level;
}

string Node::getName() {
	return this->name;
}

void Node::setName(string name) {
	this->name = name;
}

float Node::getClass() {
	int targetFeature = this->getDataset()->getData()[0].size() - 1;
	map<float, int> counters;
	for (unsigned long i = 0; i < this->getDataset()->getData().size(); i++) {
		counters[this->getDataset()->getData()[i][targetFeature]]++;
	}
	return max_element(counters.begin(), counters.end(), [](const pair<float, int>& a, const pair<float, int>& b)->bool {return a.second < b.second; })->first;
}

