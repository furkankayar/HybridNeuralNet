#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "DatasetInfo.h"
#include "DecisionTree.h"
#include "Node.h"

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
        typesVect[i] = (typesPtr[i] == 0.0 ? CATEGORICAL : CONTINUOUS);
    }

    DatasetInfo* datasetInfo = new DatasetInfo(vect_arr, typesVect);

    for (size_t i = 0; i < datasetInfo->getData()[0].size(); i++) {
        Node* root = new Node(datasetInfo, i);
        DecisionTree* dtree = new DecisionTree(root);
        dtree->splitRootNode();
        break;
    }
}



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