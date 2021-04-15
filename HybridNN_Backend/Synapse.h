#pragma once
#ifndef SYNAPSE_H
#define SYNAPSE_H

class Neuron;

class Synapse {
private:
	float weight;
	Neuron* target;
	int count;

public:
	Synapse();
	Synapse(float weight);
	Synapse(float weight, Neuron* target);
	float getWeight();
	void setWeight(float weight);
	Neuron* getTarget();
	void setTarget(Neuron* target);
	int getCount();
	void setCount(int count);
};

#endif