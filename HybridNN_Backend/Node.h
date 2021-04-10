#pragma once
#ifndef NODE_H
#define NODE_H
#include <list>
#include <string>

using namespace std;

class DatasetInfo;
class Edge;

class Node {
private:
	list<Edge*> edges;
	DatasetInfo* dataset;
	string name; 
	int level;
	int selectiveFeatureOrder;
	float threshold;
	int numberOfLT;
	int numberOfGTE;

public:
	Node();
	Node(DatasetInfo* dataset);
	Node(DatasetInfo* dataset, int selectiveFeatureOrder);
	Node(DatasetInfo* dataset, int selectiveFeatureOrder, float threshold);
	list<Edge*> getEdges();
	void addEdge(Edge* edge);
	DatasetInfo* getDataset();
	int getSelectiveFeatureOrder();
	void setSelectiveFeatureOrder(int selectiveFeatureOrder);
	float getThreshold();
	void setThreshold(float threshold);
	int getNumberOfLT();
	int getNumberOfGTE();
	void removeDataset();
	void setNumberOfLT(int numbeOfLT);
	void setNumberOfGTE(int numberOfGTE);
	int getLevel();
	void setLevel(int level);
	string getName();
	void setName(string name);
	float getClass();

};

#endif