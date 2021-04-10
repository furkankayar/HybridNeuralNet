#include "Neuron.h"
#include "Layer.h"

Layer::Layer() : Layer(LayerType::NOT_IN_USE) {}

Layer::Layer(LayerType type) : Layer(type, -1) {}

Layer::Layer(LayerType type, int layerIndex) :
	type(type),
	layerIndex(layerIndex) {}

LayerType Layer::getType() {
	return this->type;
}

void Layer::setType(LayerType type) {
	this->type = type;
}

int Layer::getLayerIndex() {
	return this->layerIndex;
}

void Layer::setLayerIndex(int layerIndex) {
	this->layerIndex = layerIndex;
}

list<Neuron*> Layer::getNeurons() {
	return this->neurons;
}
