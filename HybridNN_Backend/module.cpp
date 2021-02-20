#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "structures.cpp"

namespace py = pybind11;
using namespace structures;

void initialization(py::array_t<float> item) {
    py::buffer_info info = item.request();
    DatasetInfo datasetInfo((float*)info.ptr, info.shape[1], info.shape[0]);
    DecisionTree decisionTree(datasetInfo);

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