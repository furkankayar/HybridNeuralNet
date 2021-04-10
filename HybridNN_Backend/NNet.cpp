#include "Layer.h"
#include "NNet.h"

NNet::NNet(int layerNum) : NNet(vector<Layer*>(layerNum)) {}

NNet::NNet(vector<Layer*> layers) :
	layers(layers) {
}

vector<Layer*> NNet::getLayers() {
	return this->layers;
}

Layer* NNet::getInputLayer() {
	if (this->layers.size() == 0) {
		return NULL;
	}
	Layer* firstLayer = this->layers[0];
	if (firstLayer->getType() != LayerType::INPUT) {
		throw exception("FIRST LAYER IS NOT INPUT LAYER!");
	}
	return firstLayer;
}

Layer* NNet::getOutputLayer() {
	if (this->layers.size() == 0) {
		return NULL;
	}
	Layer* lastLayer = this->layers[this->layers.size() - 1];
	if (lastLayer->getType() != LayerType::OUTPUT) {
		throw exception("LAST LAYER IS NOT OUTPUT LAYER!");
	}
	return lastLayer;
}

Layer* NNet::findOrCreateLayerWithIndex(LayerType type, int index) {
	Layer* layer = this->layers[index];
	if (layer == NULL) {
		layer = new Layer(type, index);
	}
	return layer;
}