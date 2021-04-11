#include "Layer.h"
#include "DecisionTree.h"
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
	Layer* outputLayer = this->findOrCreateLayerWithIndex(maxTreeDepth - 1);
	
	for (Node* node : outputNodes) {
		outputLayer->insertNeuronWithClass(node->getClass());
	}
	
	
	//INTERNAL
	for (int i = 1; i < maxTreeDepth - 1; i++) {
		Layer* hiddenLayer = this->findOrCreateLayerWithIndex(i);
		list<Node*> internalNodes;
		dtree->getNodesWithLevel(dtree->getRoot(), i, internalNodes);
		for (Node* node : internalNodes) {
			hiddenLayer->insertNeuronWithFeature(node->getSelectiveFeatureOrder());
		}
	}
	
	//INPUT
	Layer* inputLayer = this->findOrCreateLayerWithIndex(0);
	inputLayer->insertNeuronWithFeature(dtree->getRoot()->getSelectiveFeatureOrder()); // BURADA ASLINDA AYNI FEATURE ICIN OLUSTURULMUS NEURON VAR MI BAKILABILIR ANCAK DOGRU CALISTIGI TAKDIRDE BUNA GEREK YOK
	
}
