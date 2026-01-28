#pragma once

#include <torch/extension.h>

// C++ 接口声明
torch::Tensor hgemm_cuda(
    torch::Tensor A,
    torch::Tensor B
);
