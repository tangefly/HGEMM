#include <torch/extension.h>

#include "ops/hgemm.h"
#include "ops/naive_hgemm.h"
#include "ops/hierarchical_tiling_hgemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "hgemm",
        &hgemm_cuda,
        "HGEMM (cuBLAS)"
    );

    m.def(
        "naive_hgemm",
        &naive_hgemm,
        "naive HGEMM (manual)"
    );

    m.def(
        "hierarchical_tiling_hgemm",
        &hierarchical_tiling_hgemm,
        "Hierarchical tiling HGEMM (manual)"
    );
}
