#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(threaded_image_ops, m) {             // NOLINT
  m.def("noop", []() { return "hello from C++"; });  // NOLINT
}
