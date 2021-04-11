#pragma once
#ifndef NEURON_H
#define NEURON_H

class Neuron {
private:
	int selectedFeature;
	float clazz;

public:
	Neuron();
	Neuron(float clazz);
	Neuron(int selectedFeature);
	int getSelectedFeature();
	void setSelectedFeature(int selectedFeature);
	float getClass();
	void setClass(float clazz);
};

#endif