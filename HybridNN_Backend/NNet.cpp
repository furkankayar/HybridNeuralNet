#include "Synapse.h"
#include "Neuron.h"
#include "Layer.h"
#include "DecisionTree.h"
#include "Edge.h"
#include "Node.h"
#include "NNet.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <math.h>

using namespace std;
namespace py = pybind11;


NNet::NNet(int layerNum):
	layers(vector<Layer*>(layerNum)){
}

vector<Layer*> NNet::getLayers() {
	return this->layers;
}

Layer* NNet::getInputLayer() {
	if (this->layers.size() == 0) {
		return NULL;
	}
	Layer* firstLayer = this->layers[0];

	return firstLayer;
}

Layer* NNet::getOutputLayer() {
	if (this->layers.size() == 0) {
		return NULL;
	}
	Layer* lastLayer = this->layers[this->layers.size() - 1];

	return lastLayer;
}

Layer* NNet::findOrCreateLayerWithIndex(int index) {
	Layer* layer = this->layers[index];
	if (layer == nullptr) {
		this->layers[index] = new Layer(index);
	}
	return this->layers[index];
}


void NNet::map(Node* node, int maxTreeDepth) {

	if (node->getLevel() < maxTreeDepth) {
		Neuron* neuron = this->findOrCreateLayerWithIndex(node->getLevel())->insertNeuronWithFeature(node->getSelectiveFeatureOrder());
		for (Edge* edge : node->getEdges()) {
			if (node->getLevel() + 1 == edge->getTarget()->getLevel()) {
				Layer* nextLayer = this->findOrCreateLayerWithIndex(edge->getTarget()->getLevel());
				Neuron* target = nullptr;
				if (edge->getTarget()->getLevel() == maxTreeDepth) {
					target = nextLayer->insertNeuronWithClass(edge->getTarget()->getClass());
				}
				else {
					target = nextLayer->insertNeuronWithFeature(edge->getTarget()->getSelectiveFeatureOrder());
				}
				neuron->addSynapse(edge->getInfoGain(), target);
			}
			else {
				// dummy section
				Layer* nextLayer = this->findOrCreateLayerWithIndex(node->getLevel() + 1);
				Neuron* target = nextLayer->insertDummyNeuron(edge->getTarget()->getClass());
				neuron->addSynapse(edge->getInfoGain(), target);
			}
		}
	}
	else {
		this->findOrCreateLayerWithIndex(maxTreeDepth)->insertNeuronWithClass(node->getClass());
	}

	for (Edge* edge : node->getEdges()) {
		this->map(edge->getTarget(), maxTreeDepth);
	}
}


void NNet::mapTree(DecisionTree* dtree, int maxTreeDepth) {
	this->map(dtree->getRoot(), maxTreeDepth);
}

/*
void NNet::mapTree(DecisionTree* dtree, int maxTreeDepth) {
	//OUTPUT

	list<Node*> outputNodes;
	dtree->getNodesWithLevel(dtree->getRoot(), maxTreeDepth, outputNodes);
	Layer* outputLayer = this->findOrCreateLayerWithIndex(maxTreeDepth);
	
	for (Node* node : outputNodes) {
		outputLayer->insertNeuronWithClass(node->getClass());
	}
		
	//INTERNAL
	for (int i = maxTreeDepth - 1; i > 0; i--) {
		Layer* hiddenLayer = this->findOrCreateLayerWithIndex(i);
		
		list<Node*> internalNodes;
		dtree->getNodesWithLevel(dtree->getRoot(), i, internalNodes);
		for (Node* node : internalNodes) {
			Neuron* neuron = hiddenLayer->insertNeuronWithFeature(node->getSelectiveFeatureOrder());
			for (Edge* edge : node->getEdges()) {
				if (node->getLevel() + 1 == edge->getTarget()->getLevel()) {
					Neuron* target = nullptr;
					Layer* nextLayer = this->findOrCreateLayerWithIndex(edge->getTarget()->getLevel());
					if (edge->getTarget()->getLevel() == maxTreeDepth) {
						target = nextLayer->getNeuronWithClass(edge->getTarget()->getClass());
					}
					else {
						target = nextLayer->getNeuronWithFeatureOrder(edge->getTarget()->getSelectiveFeatureOrder());
					}
					neuron->addSynapse(edge->getInfoGain(), target);
				}
			}
		}
	}
	
	//INPUT
	Layer* inputLayer = this->findOrCreateLayerWithIndex(0);
	Neuron* neuron = inputLayer->insertNeuronWithFeature(dtree->getRoot()->getSelectiveFeatureOrder()); // BURADA ASLINDA AYNI FEATURE ICIN OLUSTURULMUS NEURON VAR MI BAKILABILIR ANCAK DOGRU CALISTIGI TAKDIRDE BUNA GEREK YOK
	for (Edge* edge : dtree->getRoot()->getEdges()) {
		Neuron* target = nullptr;
		Layer* nextLayer = this->findOrCreateLayerWithIndex(edge->getTarget()->getLevel());
		if (edge->getTarget()->getLevel() == maxTreeDepth) {
			target = nextLayer->getNeuronWithClass(edge->getTarget()->getClass());
		}
		else {
			target = nextLayer->getNeuronWithFeatureOrder(edge->getTarget()->getSelectiveFeatureOrder());
		}
		if (edge->getTarget()->getLevel() == 1) {
			neuron->addSynapse(edge->getInfoGain(), target);
		}
	}
}*/

void NNet::print() {
	for (Layer* layer : this->layers) {
		cout << "Layer: " << layer->getLayerIndex() << endl;
		for (Neuron* neuron : layer->getNeurons()) {
			cout << "\tNeuron Feature: " << neuron->getSelectedFeature() << " Class: " << neuron->getClass() << endl;
			for (Synapse* synapse : neuron->getSynapses()) {
				cout << "\t\tSynapse Weight: " << synapse->getWeight() << " TO Class: " << synapse->getTarget()->getClass() << " Feature: " << synapse->getTarget()->getSelectedFeature() << endl;
			}
		}
		for (Neuron* neuron : layer->getDummyNeurons()) {
			cout << "\tDummy Neuron Class: " << neuron->getClass() << endl;
			for (Synapse* synapse : neuron->getSynapses()) {
				cout << "\t\tSynapse Weight: " << synapse->getWeight() << " TO Class: " << synapse->getTarget()->getClass() << " Feature: " << synapse->getTarget()->getSelectedFeature() << endl;
			}
		}
	}
}

bool NNet::hasConnection(Neuron* neuron, Neuron* nextNeuron) {
	for (Synapse* synapse : neuron->getSynapses()) {
		if (synapse->getTarget() == nextNeuron) {
			return true;
		}
	}
	return false;
}

void NNet::complete(list<float> labels) {

	for (int i = 1; i < this->layers.size() - 2; i++) {
		Layer* layer = this->layers[i];
		Layer* nextLayer = this->layers[i + 1];
		for (Neuron* neuron : layer->getDummyNeurons()) {
			if (nextLayer->getDummyNeuron(neuron->getClass()) == nullptr) {
				nextLayer->insertDummyNeuron(neuron->getClass());
			}
		}
	}
	
	for (int i = 0; i < this->layers.size() - 1; i++) {
		Layer* layer = this->layers[i];
		Layer* nextLayer = this->layers[i + 1];
		for (Neuron* neuron : layer->getNeurons()) {
			for (Neuron* nextNeuron : nextLayer->getNeurons()) {
				if (!hasConnection(neuron, nextNeuron)) {
					neuron->addSynapse(0.0f, nextNeuron);
				}
			}
			for (Neuron* nextNeuron : nextLayer->getDummyNeurons()) {
				if (!hasConnection(neuron, nextNeuron)) {
					neuron->addSynapse(0.0f, nextNeuron);
				}
			}
		}

		for (Neuron* neuron : layer->getDummyNeurons()) {
			for (Neuron* nextNeuron : nextLayer->getNeurons()) {
				if (!hasConnection(neuron, nextNeuron)) {
					if (neuron->getClass() == nextNeuron->getClass()) {
						neuron->addSynapse(1.0f, nextNeuron);
					}
					else {
						neuron->addSynapse(0.0f, nextNeuron);
					}
				}
			}
			for (Neuron* nextNeuron : nextLayer->getDummyNeurons()) {
				if (!hasConnection(neuron, nextNeuron)) {
					if (neuron->getClass() == nextNeuron->getClass()) {
						neuron->addSynapse(1.0f, nextNeuron);
					}
					else {
						neuron->addSynapse(0.0f, nextNeuron);
					}
				}
			}
		}

		layer->sortNeurons();
	}

	Layer* outputLayer = this->getOutputLayer();
	Layer* layerBeforeOutput = this->layers[this->layers.size() - 2];
	for (float label : labels) {
		if (outputLayer->getNeuronWithClass(label) == nullptr) {
			outputLayer->insertNeuronWithClass(label);
			Neuron* neuron = outputLayer->getNeuronWithClass(label);
			for (Neuron* n : layerBeforeOutput->getNeurons()) {
				n->addSynapse(0.0f, neuron);
			}
		}
	}

	for (Layer* layer : this->layers) {
		layer->sortNeurons();
	}
}



py::tuple NNet::nnetToNumpy() {

	vector<vector<vector<float>>> weights(this->layers.size() - 1);
	int layerCount = 0;
	for (int i = 0; i < this->layers.size() - 1; i++) {
		Layer* layer = this->layers[i];
		Layer* nextLayer = this->layers[i + 1];
		vector<vector<float>> layerWeights(layer->getNeurons().size() + layer->getDummyNeurons().size());
		int layerWeightCount = 0; 
		for (Neuron* neuron : layer->getNeurons()) {
			vector<float> neuronWeights(neuron->getSynapses().size());
			int neuronWeightCount = 0;
			for (Synapse* synapse : neuron->getSynapses()) {
				neuronWeights[neuronWeightCount++] = synapse->getWeight();
			}
			layerWeights[layerWeightCount++] = neuronWeights;
		}

		for (Neuron* neuron : layer->getDummyNeurons()) {
			vector<float> neuronWeights(neuron->getSynapses().size());
			int neuronWeightCount = 0;
			for (Synapse* synapse : neuron->getSynapses()) {
				neuronWeights[neuronWeightCount++] = synapse->getWeight();
			}
			layerWeights[layerWeightCount++] = neuronWeights;
		}

		weights[layerCount++] = layerWeights;
	}

	vector<int> shape(this->layers.size());
	for (int i = 0; i < this->layers.size(); i++) {
		shape[i] = this->layers[i]->getNeurons().size() + this->layers[i]->getDummyNeurons().size();
	}

	auto tuple = py::tuple(2);
	tuple[0] = py::cast(weights);
	tuple[1] = py::cast(shape);

	return tuple;
}

