#include "Neuron.h"
#include "Layer.h"
#include <iostream>

using namespace std;

Layer::Layer() : Layer(-1) {}

Layer::Layer(int layerIndex){
	this->layerIndex = layerIndex;
	this->neurons = list<Neuron*>();
}


int Layer::getLayerIndex() {
	return this->layerIndex;
}

void Layer::setLayerIndex(int layerIndex) {
	this->layerIndex = layerIndex;
}

list<Neuron*> Layer::getNeurons() {
	return this->neurons;
}

void Layer::insertNeuronWithClass(float clazz) {
	for (Neuron* neuron : this->neurons) {
		if (neuron->getClass() == clazz) {
			return;
		}
	}
	this->neurons.push_back(new Neuron(clazz));
}

Neuron* Layer::insertNeuronWithFeature(int feature) {

	for (Neuron* nr : this->neurons) {
		if (nr->getSelectedFeature() == feature) {
			return nr;
		}
	}
	
	Neuron* newNeuron = new Neuron(feature);
	this->neurons.push_back(newNeuron);
	return newNeuron;
}

Neuron* Layer::getNeuronWithClass(float clazz) {
	for (Neuron* neuron : this->neurons) {
		if (neuron->getClass() == clazz) {
			return neuron;
		}
	}

	return nullptr;
}

Neuron* Layer::getNeuronWithFeatureOrder(int feature) {
	for (Neuron* neuron : this->neurons) {
		if (neuron->getSelectedFeature() == feature) {
			return neuron;
		}
	}

	return nullptr;
}
