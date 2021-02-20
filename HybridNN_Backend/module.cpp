#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "DatasetInfo.h"
#include "DecisionTree.h"
#include "Node.h"


namespace py = pybind11;
using namespace std;

void initialization(py::array_t<float> item) {
    py::buffer_info info = item.request();

    float* vals = (float*)info.ptr;

    unsigned int Y = info.shape[0];
    unsigned int X = info.shape[1];
    
    vector<vector <float>> vect_arr(Y, vector<float>(X));

    for (unsigned int i = 0; i < Y; i++) {
        for (unsigned int j = 0; j < X; j++) {
            vect_arr[i][j] = vals[j * Y + i ];
        }
    }

    DatasetInfo datasetInfo(vect_arr);

    for (size_t i = 0; i < datasetInfo.getData()[0].size(); i++) {
        Node* root = new Node(&datasetInfo, i);
        DecisionTree* dtree = new DecisionTree(root);
        dtree->splitNode();
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