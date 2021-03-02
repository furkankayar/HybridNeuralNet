#include "Node.h"
#include "Edge.h"

Edge::Edge() : Edge(NULL, 0.0) {}

Edge::Edge(Node* target) : Edge(target, 0.0) {}

Edge::Edge(Node* target, float infoGain):
	target(target),
	infoGain(infoGain){}

float Edge::getInfoGain() {
	return this->infoGain;
}

void Edge::setInfoGain(float infoGain) {
	this->infoGain = infoGain;
}

Node* Edge::getTarget() {
	return this->target;
}

void Edge::setTarget(Node* target){
	this->target = target;
}