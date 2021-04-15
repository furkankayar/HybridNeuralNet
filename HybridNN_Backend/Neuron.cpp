#include "Synapse.h"
#include "Neuron.h"
#include <iostream>

Neuron::Neuron() : Neuron(-1) {}

Neuron::Neuron(int selectedFeature) :
	selectedFeature(selectedFeature),
	clazz(-1.0f){}

Neuron::Neuron(float clazz) :
	clazz(clazz),
	selectedFeature(-1){}

int Neuron::getSelectedFeature() {
	return this->selectedFeature;
}

void Neuron::setSelectedFeature(int selectedFeature) {
	this->selectedFeature = selectedFeature;
}

float Neuron::getClass() {
	return this->clazz;
}

void Neuron::setClass(float clazz) {
	this->clazz = clazz;
}

list<Synapse*> Neuron::getSynapses() {
	return this->synapses;
}

void Neuron::addSynapse(float weight, Neuron* target) {
	for (Synapse* sn : this->synapses) {
		if (sn->getTarget() == target) {
			float curW = sn->getCount() * sn->getWeight();
			sn->setWeight((curW + weight) / (sn->getCount() + 1));
			sn->setCount(sn->getCount() + 1);
			return;
		}
	}
	this->synapses.push_back(new Synapse(weight, target));
}


