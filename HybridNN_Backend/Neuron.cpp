#include "Neuron.h"

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


