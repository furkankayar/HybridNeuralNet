#include "Neuron.h"
#include "Layer.h"
#include <iostream>
#include <algorithm>

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

void Layer::sortNeurons() {
	this->neurons.sort([](Neuron* a, Neuron* b) {
		if (a->getClass() == -1 && b->getClass() == -1) {
			return a->getSelectedFeature() < b->getSelectedFeature();
		}
		else {
			return a->getClass() < b->getClass();
		}
	});

	this->dummyNeurons.sort([](Neuron* a, Neuron* b) {
		return a->getClass() < b->getClass();
	});

	for (Neuron* neuron : this->neurons) {
		neuron->sortSynapses();
	}

	for (Neuron* neuron : this->dummyNeurons) {
		neuron->sortSynapses();
	}
}

list<Neuron*> Layer::getNeurons() {
	return this->neurons;
}

list<Neuron*> Layer::getDummyNeurons() {
	return this->dummyNeurons;
}

Neuron* Layer::insertNeuronWithClass(float clazz) {
	for (Neuron* neuron : this->neurons) {
		if (neuron->getClass() == clazz) {
			return neuron;
		}
	}
	Neuron* neuron = new Neuron(clazz);
	this->neurons.push_back(neuron);
	return neuron; 
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

Neuron* Layer::insertDummyNeuron(float forClass) {
	for (Neuron* neuron : this->dummyNeurons) {
		if (neuron->getClass() == forClass) {
			return neuron;
		}
	}

	Neuron* neuron = new Neuron(forClass);
	this->dummyNeurons.push_back(neuron);
	return neuron;
}

Neuron* Layer::getDummyNeuron(float forClass) {
	for (Neuron* neuron : this->dummyNeurons) {
		if (neuron->getClass() == forClass) {
			return neuron;
		}
	}

	return nullptr;
}

