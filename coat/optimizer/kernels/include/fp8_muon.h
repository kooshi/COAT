#pragma once
#include <torch/extension.h>

void FP8_Muon(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size
);

void FP8_Muon_expand(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor expand_momentum,
    torch::Tensor sqrtminmax_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size, int expand_min
);
