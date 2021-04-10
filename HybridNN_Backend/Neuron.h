#pragma once
#ifndef NEURON_H
#define NEURON_H

class Neuron {
private:
	int selectedFeature;

public:
	Neuron();
	Neuron(int selectedFeature);
	int getSelectedFeature();
	void setSelectedFeature(int selectedFeature);
};

#endif