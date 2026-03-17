#include <torch/extension.h>
#include <torch/torch.h>

void FP8_Muon_expand_cuda(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor expand_momentum,
    torch::Tensor sqrtminmax_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size, int expand_min
);

void FP8_Muon_expand(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor expand_momentum,
    torch::Tensor sqrtminmax_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size, int expand_min
) {
    FP8_Muon_expand_cuda(
        nesterov_update,
        q_momentum, scale_momentum, expand_momentum, sqrtminmax_momentum,
        grads,
        mu, nesterov, qgroup_size, expand_min
    );
}
