#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void hello_world(py::array_t<float> item) {
    py::buffer_info info = item.request();


}



PYBIND11_MODULE(HybridNN_Backend, m) {
    m.def("hello_world", &hello_world, R"pbdoc(
        Hello.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}