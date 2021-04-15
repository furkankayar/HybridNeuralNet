#include "Synapse.h"
#include "Neuron.h"
#include "Layer.h"
#include "DecisionTree.h"
#include "Edge.h"
#include "Node.h"
#include "NNet.h"

#include <iostream>
using namespace std;

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
				Neuron* target = nullptr;
				Layer* nextLayer = this->findOrCreateLayerWithIndex(edge->getTarget()->getLevel());
				if (edge->getTarget()->getLevel() == maxTreeDepth) {
					target = nextLayer->getNeuronWithClass(edge->getTarget()->getClass());
				}
				else {
					target = nextLayer->getNeuronWithFeatureOrder(edge->getTarget()->getSelectiveFeatureOrder());
				}
				if (target == nullptr) {
					cout << "SIKINTI VAR" << endl;
					cout << edge->getTarget()->getSelectiveFeatureOrder() << endl;
					cout << nextLayer->getNeurons().size() << endl;
					cout << i << endl;

				}
				else {
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
		neuron->addSynapse(edge->getInfoGain(), target);
	}
}

void NNet::print() {
	for (Layer* layer : this->layers) {
		cout << "Layer: " << layer->getLayerIndex() << endl;
		for (Neuron* neuron : layer->getNeurons()) {
			cout << "\tNeuron Feature: " << neuron->getSelectedFeature() << " Class: " << neuron->getClass() << endl;
			for (Synapse* synapse : neuron->getSynapses()) {
				cout << "\t\tSynapse Weight: " << synapse->getWeight() << " TO Class: " << synapse->getTarget()->getClass() << " Feature: " << synapse->getTarget()->getSelectedFeature() << endl;
			}
		}
	}
}

