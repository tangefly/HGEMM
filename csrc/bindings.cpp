#include <torch/extension.h>
#include "ops/hgemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemm", &hgemm_cuda, "HGEMM (cuBLAS)");
}
