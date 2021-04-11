#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <thread>
#include "DatasetInfo.h"
#include "DecisionTree.h"
#include "Node.h"
#include "Edge.h"
#include "NNet.h"
#include "Neuron.h"
#include "Layer.h"

namespace py = pybind11;
using namespace std;

void initialization(py::array_t<float> dataset, py::array_t<float> types) {
    py::buffer_info datasetBufferInfo = dataset.request();
    py::buffer_info typesBufferInfo = types.request();
    
    /* Read dataset */
    float* datasetPtr = (float*)datasetBufferInfo.ptr;

    unsigned int datasetY = datasetBufferInfo.shape[0];
    unsigned int datasetX = datasetBufferInfo.shape[1];
    
    vector<vector <float>> vect_arr(datasetY, vector<float>(datasetX));

    for (unsigned int i = 0; i < datasetY; i++) {
        for (unsigned int j = 0; j < datasetX; j++) {
            vect_arr[i][j] = datasetPtr[j * datasetY + i ];
        }
    }

    /* Read types of descriptive features */
    float* typesPtr = (float*)typesBufferInfo.ptr;
    unsigned int typesY = typesBufferInfo.shape[0];
    vector<Type> typesVect(typesY);

    for (unsigned int i = 0; i < typesY; i++) {
        typesVect[i] = (typesPtr[i] == 0.0 ? Type::CATEGORICAL : Type::CONTINUOUS);
    }

 

    //vector<thread> threadList = vector<thread>(typesVect.size() - 1);

    vector<DecisionTree*> trees = vector<DecisionTree*>(typesVect.size() - 1);

    for (size_t i = 0; i < typesVect.size() - 1; i++) {
        DatasetInfo* datasetInfo = new DatasetInfo(vect_arr, typesVect);
        Node* root = new Node(datasetInfo, i);
        DecisionTree* dtree = new DecisionTree(root);
        dtree->splitRootNode();
        
        //threadList[i] = thread([dtree]() {
            int count = 0;
            for (Edge* edge : dtree->getRoot()->getEdges()) {
                Node* child = edge->getTarget();
                child->setName(dtree->getRoot()->getName() + "-" + to_string(count));
                dtree->buildTree(child, 1);
                count++;
            }
            trees[i] = dtree;
            cout << dtree->getMaxTreeDepth() << endl;
        //    });
        //dtree->printTree(dtree->getRoot());
        cout << "=====================" << endl;
    }


    /*for (int i = 0; i < typesVect.size() - 1; i++) {
        cout << "Waiting thread " << i << endl;
        threadList[i].join();
    }*/



    int maxTreeDepth = 0;
    for (int i = 0; i < typesVect.size() - 1; i++) {
        if (trees[i]->getMaxTreeDepth() > maxTreeDepth) {
            maxTreeDepth = trees[i]->getMaxTreeDepth();
        }
    }

    for (int i = 0; i < typesVect.size() - 1; i++) {
        trees[i]->moveLeafNodes(trees[i]->getRoot(), maxTreeDepth);
    }

    NNet* nnet = new NNet(maxTreeDepth);
    
    for (int i = 0; i < typesVect.size() - 1; i++) {
        nnet->mapTree(trees[i], maxTreeDepth);
    }

    for (int i = 0; i < maxTreeDepth; i++) {
        cout << "Layer " << i  << " size: " << nnet->findOrCreateLayerWithIndex(i)->getNeurons().size() << endl;
    }

    cout << "Successful finish" << endl;
}

/*
void mapTreeToNNet(DecisionTree* dtree, NNet* nnet, int maxTreeDepth) {
   
    //OUTPUT
    list<Node*> outputNodes;
    dtree->getNodesWithLevel(dtree->getRoot(), maxTreeDepth, outputNodes);
    Layer* outputLayer = nnet->findOrCreateLayerWithIndex(LayerType::OUTPUT, maxTreeDepth);
    for (Node* node : outputNodes) {
        outputLayer->insertNeuronWithClass(node->getClass());
    }


    //INTERNAL
    for (int i = 1; i < maxTreeDepth - 1; i++) {
        Layer* hiddenLayer = nnet->findOrCreateLayerWithIndex(LayerType::HIDDEN, i);
        list<Node*> internalNodes;
        dtree->getNodesWithLevel(dtree->getRoot(), i, internalNodes);
        for (Node* node : internalNodes) {
            hiddenLayer->insertNeuronWithFeature(node->getSelectiveFeatureOrder());
        }
    }

    //INPUT
    Layer* inputLayer = nnet->findOrCreateLayerWithIndex(LayerType::INPUT, 0);
    inputLayer->insertNeuronWithClass(dtree->getRoot()->getSelectiveFeatureOrder()); // BURADA ASLINDA AYNI FEATURE ICIN OLUSTURULMUS NEURON VAR MI BAKILABILIR ANCAK DOGRU CALISTIGI TAKDIRDE BUNA GEREK YOK

}
*/



PYBIND11_MODULE(HybridNN_Backend, m) {
    m.def("initialization", &initialization, R"pbdoc(
        Initialize structures.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}