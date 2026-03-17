#include <torch/extension.h>
#include <torch/torch.h>

void FP8_Muon_cuda(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size
);

void FP8_Muon(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size
) {
    FP8_Muon_cuda(
        nesterov_update,
        q_momentum, scale_momentum,
        grads,
        mu, nesterov, qgroup_size
    );
}
