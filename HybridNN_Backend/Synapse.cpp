#include "Neuron.h"
#include "Synapse.h"

Synapse::Synapse() : Synapse(0.0f) {}

Synapse::Synapse(float weight) : Synapse(weight, nullptr) {}

Synapse::Synapse(float weight, Neuron* target) :
	weight(weight),
	target(target),
	count(1){}

float Synapse::getWeight() {
	return this->weight;
}

void Synapse::setWeight(float weight) {
	this->weight = weight;
}

Neuron* Synapse::getTarget() {
	return this->target;
}

void Synapse::setTarget(Neuron* target) {
	this->target = target;
}

int Synapse::getCount() {
	return this->count;
}

void Synapse::setCount(int count) {
	this->count = count;
}