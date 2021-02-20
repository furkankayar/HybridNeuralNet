#pragma once
#ifndef NODE_H
#define NODE_H
#include <list>

using namespace std;

class DatasetInfo;
class Edge;

class Node {
private:
	list<Edge> edges;
	DatasetInfo* dataset; 
	int selectiveFeatureOrder;
	float threshold;
public:
	Node();
	Node(float* data, int X, int Y, int selectiveFeatureOder, float threshold);
	list<Edge> getEdges();
	DatasetInfo* getDataset();
	int getSelectiveFeatureOrder();
	void setSelectiveFeatureOrder(int selectiveFeatureOrder);
	float getThreshold();
	void setThreshold(float threshold);
};

#endif