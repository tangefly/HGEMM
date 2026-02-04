#include <torch/extension.h>

#include "ops/hgemm.h"
#include "ops/naive_hgemm.h"
#include "ops/hierarchical_tiling_hgemm.h"
#include "ops/thread_tile_hgemm.h"
#include "ops/warp_hgemm.h"

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

    m.def(
        "thread_tile_hgemm",
        &thread_tile_hgemm,
        "Thread Tile HGEMM (manual)"
    );

    m.def(
        "warp_hgemm",
        &warp_hgemm,
        "Warp HGEMM (manual)"
    );
}
