#pragma once

#include <torch/extension.h>

// C++ 接口声明
torch::Tensor thread_tile_hgemm(
    torch::Tensor A,
    torch::Tensor B
);
