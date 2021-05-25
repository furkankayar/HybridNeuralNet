#pragma once
#ifndef NNET_H
#define NNET_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

using namespace std;
namespace py = pybind11;

class Neuron;
class Layer;

class NNet {
private:
	vector<Layer*> layers;

public:
	NNet(int layerNum);
	vector<Layer*> getLayers();
	Layer* getInputLayer();
	Layer* getOutputLayer();
	Layer* findOrCreateLayerWithIndex(int index);
	void mapTree(DecisionTree* dtree, int maxTreeDepth);
	void print();
	void complete(list<float> labels);
	bool hasConnection(Neuron* neuron, Neuron* nextNeuron);
	void map(Node* node, int maxTreeDepth);
	py::tuple nnetToNumpy();
};

#endif
