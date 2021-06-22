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

#define VERSION_INFO "1.0.0"

namespace py = pybind11;
using namespace std;

py::tuple nnet_initialization(py::array_t<float> dataset, py::array_t<float> types) {
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

    vector<thread> threadList = vector<thread>(typesVect.size() - 1);

    vector<DecisionTree*> trees = vector<DecisionTree*>(typesVect.size() - 1);

    DatasetInfo* datasetInfo = nullptr;
    cout << "Started to build Neural Network Structure" << endl;
    cout << "Building trees" << endl;
    for (size_t i = 0; i < typesVect.size() - 1; i++) {
        datasetInfo = new DatasetInfo(vect_arr, typesVect);
        Node* root = new Node(datasetInfo, i);
        int acceptableMaxDepth = typesVect.size() > 5 ? 5 : typesVect.size() - 1;
        DecisionTree* dtree = new DecisionTree(root, acceptableMaxDepth);
        dtree->splitRootNode();
        //cout << "Building Tree " << i << endl;
        trees[i] = dtree;
        threadList[i] = thread([dtree]() {
            int count = 0;
            for (Edge* edge : dtree->getRoot()->getEdges()) {
                Node* child = edge->getTarget();
                child->setName(dtree->getRoot()->getName() + "-" + to_string(count));
                dtree->buildTree(child, 1);
                count++;
            }
          
            });
        //cout << "=====================" << endl;
    }


    for (int i = 0; i < typesVect.size() - 1; i++) {
        //cout << "Waiting thread " << i << endl;
        threadList[i].join();
    }



    int maxTreeDepth = 0;
    for (int i = 0; i < typesVect.size() - 1; i++) {
        if (trees[i]->getMaxTreeDepth() > maxTreeDepth) {
            maxTreeDepth = trees[i]->getMaxTreeDepth();
        }
    }

    for (int i = 0; i < typesVect.size() - 1; i++) {
        trees[i]->initializeNonAssignedWeights();
        trees[i]->moveLeafNodes(trees[i]->getRoot(), maxTreeDepth);
    }

    /*for (int i = 0; i < 4; i++) {
        trees[i]->printTree(trees[i]->getRoot());
        cout << "--------------------------- END -------------------------" << endl;
    }*/


    NNet* nnet = new NNet(maxTreeDepth + 1);
    
    for (int i = 0; i < typesVect.size() - 1; i++) {
        //cout << "Mapping Tree " << i << endl;
        nnet->mapTree(trees[i], maxTreeDepth);
    }


    
    nnet->complete(datasetInfo->getTokens());
    nnet->print();
    for (int i = 0; i < maxTreeDepth + 1; i++) {
        cout << "Layer " << i  << " size: " << nnet->findOrCreateLayerWithIndex(i)->getNeurons().size() + nnet->findOrCreateLayerWithIndex(i)->getDummyNeurons().size() << endl;
    }


    return nnet->nnetToNumpy();
}


PYBIND11_MODULE(HybridNN_Backend, m) {
    m.def("nnet_initialization", &nnet_initialization, R"pbdoc(
        Initialize structures.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}