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

void Neuron::sortSynapses() {
	this->synapses.sort([](Synapse* a, Synapse* b) {
		if (a->getTarget()->getClass() == -1 && b->getTarget()->getClass() == -1) {
			return a->getTarget()->getSelectedFeature() < b->getTarget()->getSelectedFeature();
		}
		else if (a->getTarget()->getClass() == -1) {
			return true;
		}
		else if (b->getTarget()->getClass() == -1) {
			return false;
		}
		else {
			return a->getTarget()->getClass() < b->getTarget()->getClass();
		}
	});
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

float Neuron::getValue() {
	return this->value;
}

void Neuron::setValue(float value) {
	this->value = value;
}

void Neuron::addValue(float value) {
	this->value += value; 
}





