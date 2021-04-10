#include "Neuron.h"

Neuron::Neuron() : Neuron(-1) {}

Neuron::Neuron(int selectedFeature) :
	selectedFeature(selectedFeature) {}

int Neuron::getSelectedFeature() {
	return this->selectedFeature;
}

void Neuron::setSelectedFeature(int selectedFeature) {
	this->selectedFeature = selectedFeature;
}
