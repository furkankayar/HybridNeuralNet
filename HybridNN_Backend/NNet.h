#pragma once
#ifndef NNET_H
#define NNET_H
#include <vector>

using namespace std;

class Layer;
enum LayerType;

class NNet {
private:
	vector<Layer*> layers;

public:
	NNet(int layerNum);
	NNet(vector<Layer*> layers);
	vector<Layer*> getLayers();
	Layer* getInputLayer();
	Layer* getOutputLayer();
	Layer* findOrCreateLayerWithIndex(LayerType type, int index);
};

#endif
