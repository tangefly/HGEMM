#pragma once

#include <torch/extension.h>

// C++ 接口声明
torch::Tensor naive_hgemm(
    torch::Tensor A,
    torch::Tensor B
);
