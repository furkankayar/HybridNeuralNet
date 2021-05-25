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
	list<Neuron*> dummyNeurons;

public:
	Layer();
	Layer(int layerIndex);
	int getLayerIndex();
	void setLayerIndex(int layerIndex);
	void sortNeurons();
	list<Neuron*> getNeurons();
	list<Neuron*> getDummyNeurons();
	Neuron* insertNeuronWithClass(float clazz);
	Neuron* insertNeuronWithFeature(int feature);
	Neuron* getNeuronWithClass(float clazz);
	Neuron* getNeuronWithFeatureOrder(int feature);
	Neuron* insertDummyNeuron(float forClass);
	Neuron* getDummyNeuron(float forClass);
};

#endif