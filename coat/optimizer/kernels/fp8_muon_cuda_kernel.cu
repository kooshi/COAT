#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdio.h>

#define QUANT_MIN_VAL 1e-30

namespace cg = cooperative_groups;
#define WARPSIZE 32

// ---------------------------------------------------------------------------
// fp8_muon_cuda_kernel  (simple per-group max scaling, no range expansion)
//
// Dequantize → momentum lerp → Nesterov blend → write update → max reduction
// → requantize with simple absmax scale.
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void fp8_muon_cuda_kernel(
    scalar_t        * __restrict__ nesterov_update,  // OUTPUT
    __nv_fp8_e4m3   * __restrict__ q_momentum,       // IN/OUT
    float           * __restrict__ scale_momentum,   // IN/OUT
    scalar_t        * __restrict__ grads,             // INPUT
    float mu, int nesterov, int qgroup_size,
    int total_elements
) {
    const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_idx = blockIdx.x;

    float old_buf, new_buf, float_grad;

    if (idx < total_elements) {
        // Dequantize (simple: q * scale, no expansion)
        old_buf    = float(q_momentum[idx]) * scale_momentum[scale_idx];
        float_grad = float(grads[idx]);

        // Momentum update
        new_buf = mu * old_buf + (1.0f - mu) * float_grad;

        // Nesterov blend
        float float_update = nesterov ? ((1.0f - mu) * float_grad + mu * new_buf)
                                      : new_buf;
        nesterov_update[idx] = scalar_t(float_update);
    } else {
        new_buf = 0.0f;
    }

    // ── Block-level absmax reduction ─────────────────────────────────────────
    int wid  = threadIdx.x / WARPSIZE;
    int lane = threadIdx.x % WARPSIZE;

    __shared__ float sharedMaxVal[32];

    cg::thread_block_tile<32> warpTile = cg::tiled_partition<32>(cg::this_thread_block());

    float maxVal = fabsf(new_buf);
    for (int i = warpTile.size() / 2; i > 0; i /= 2) {
        maxVal = fmax(maxVal, fabsf(warpTile.shfl_down(maxVal, i)));
    }
    if (lane == 0) sharedMaxVal[wid] = maxVal;
    __syncthreads();

    __shared__ float shared_absmax;
    maxVal = (threadIdx.x < blockDim.x / warpSize) ? sharedMaxVal[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
            maxVal = fmax(maxVal, fabsf(__shfl_down_sync(0xFFFFFFFF, maxVal, offset)));
        }
        if (lane == 0) shared_absmax = maxVal;
    }
    __syncthreads();

    if (idx < total_elements) {
        const float fp8MaxVal = 448.0f;
        float new_scale = (shared_absmax + QUANT_MIN_VAL) / fp8MaxVal;
        q_momentum[idx]           = static_cast<__nv_fp8_e4m3>(new_buf / new_scale);
        scale_momentum[scale_idx] = new_scale;
    }
}

void FP8_Muon_cuda(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size
) {
    int total_elements = grads.numel();

    AT_ASSERTM(qgroup_size == 128, "Only 128-element per-group quantization is supported");
    const int block_dim = 128;
    int grid_dim = (total_elements + qgroup_size - 1) / block_dim;
    AT_ASSERTM(grid_dim == scale_momentum.numel(),
               "scale_momentum shape mismatch with ceil(numel/group_size)");

    const dim3 blocks(grid_dim);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, grads.scalar_type(), "fp8_muon", ([&] {
            fp8_muon_cuda_kernel<scalar_t><<<blocks, block_dim>>>(
                nesterov_update.data_ptr<scalar_t>(),
                (__nv_fp8_e4m3*)q_momentum.data_ptr<at::Float8_e4m3fn>(),
                scale_momentum.data_ptr<float>(),
                grads.data_ptr<scalar_t>(),
                mu, nesterov, qgroup_size, total_elements
            );
        })
    );
}
