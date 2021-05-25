#pragma once
#ifndef NEURON_H
#define NEURON_H
#include <list>

using namespace std;

class Synapse;

class Neuron {
private:
	list<Synapse*> synapses;
	int selectedFeature;
	float clazz;
	float value;

public:
	Neuron();
	Neuron(float clazz);
	Neuron(int selectedFeature);
	int getSelectedFeature();
	void setSelectedFeature(int selectedFeature);
	float getClass();
	void setClass(float clazz);
	list<Synapse*> getSynapses();
	void addSynapse(float weight, Neuron* target);
	float getValue();
	void setValue(float value);
	void addValue(float value);
	void sortSynapses();
};

#endif