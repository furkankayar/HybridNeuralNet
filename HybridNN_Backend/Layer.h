#pragma once
#ifndef LAYER_H
#define LAYER_H
#include <list> 

using namespace std;

class Neuron;

class Layer {

private:
	int layerIndex;
	list<Neuron*> neurons; 

public:
	Layer();
	Layer(int layerIndex);
	int getLayerIndex();
	void setLayerIndex(int layerIndex);
	list<Neuron*> getNeurons();
	void insertNeuronWithClass(float clazz);
	void insertNeuronWithFeature(int feature);
};

#endif