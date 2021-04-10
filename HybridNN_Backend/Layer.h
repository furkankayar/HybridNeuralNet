#pragma once
#ifndef LAYER_H
#define LAYER_H
#include <list> 

using namespace std;

class Neuron;

enum LayerType{
	INPUT=0,
	HIDDEN=1,
	OUTPUT=2,
	NOT_IN_USE=-1
};

class Layer {

private:
	LayerType type;
	int layerIndex;
	list<Neuron*> neurons; 

public:
	Layer();
	Layer(LayerType type);
	Layer(LayerType type, int layerIndex);
	LayerType getType();
	void setType(LayerType type);
	int getLayerIndex();
	void setLayerIndex(int layerIndex);
	list<Neuron*> getNeurons();
};

#endif