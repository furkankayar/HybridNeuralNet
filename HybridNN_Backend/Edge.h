#pragma once
#ifndef EDGE_H
#define EDGE_H

class Node;

class Edge {
private:
	Node* target;
	float infoGain;
public:
	Edge();
	Edge(Node* target);
	Edge(Node* target, float infoGain);
	Node* getTarget();
	void setTarget(Node* target);
	float getInfoGain();
	void setInfoGain(float infoGain);
};

#endif