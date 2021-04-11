#pragma once
#ifndef NNET_H
#define NNET_H
#include <vector>

using namespace std;

class Layer;

class NNet {
private:
	vector<Layer*> layers;

public:
	NNet(int layerNum);
	vector<Layer*> getLayers();
	Layer* getInputLayer();
	Layer* getOutputLayer();
	Layer* findOrCreateLayerWithIndex(int index);
	void mapTree(DecisionTree* dtree, int maxTreeDepth);
};

#endif
