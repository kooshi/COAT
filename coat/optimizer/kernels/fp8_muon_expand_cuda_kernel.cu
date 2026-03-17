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
// fp8_muon_cuda_expand_kernel
//
// One block per quantization group (block_dim == qgroup_size == 128).
// One thread per element.
//
// Per-step work per element:
//   1. Dequantize FP8 momentum with dynamic-range expansion (COAT eq. 3 inverse)
//   2. Momentum lerp:   new_buf = mu * old_buf + (1 - mu) * grad
//   3. Nesterov blend:  update  = (1 - mu) * grad + mu * new_buf  (or new_buf)
//   4. Write BF16/FP16 update for Newton-Schulz (called from Python)
//   5. Block-level min/max reduction of new_buf
//   6. Compute new expansion exponent and re-quantize new_buf → FP8
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void fp8_muon_cuda_expand_kernel(
    scalar_t        * __restrict__ nesterov_update,   // OUTPUT: update for Newton-Schulz
    __nv_fp8_e4m3   * __restrict__ q_momentum,        // IN/OUT: FP8 quantized momentum
    float           * __restrict__ scale_momentum,    // IN/OUT: per-group scale
    float           * __restrict__ expand_momentum,   // IN/OUT: per-group expansion exponent k
    float           * __restrict__ sqrtminmax_momentum,// IN/OUT: per-group sqrt(min*max)
    scalar_t        * __restrict__ grads,              // INPUT:  gradient
    float mu,                                          // momentum coefficient (e.g. 0.95)
    int nesterov,                                      // 1 = Nesterov, 0 = standard
    int qgroup_size, int expand_min,
    int total_elements, int total_scale_elements
) {
    const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_idx = blockIdx.x;

    float old_buf, new_buf, float_grad;

    if (idx < total_elements) {
        // ── Step 1: Dequantize FP8 momentum ─────────────────────────────────
        // Forward quantization stored: sign(x) * (|x|/sqrtMM)^k  scaled by s
        // Inverse: x = sign(q*s) * |q*s|^(1/k) * sqrtMM
        float raw = float(q_momentum[idx]) * scale_momentum[scale_idx];
        int sign_m = 1 - 2 * signbit(raw);
        old_buf = sign_m * powf(fabsf(raw), 1.0f / expand_momentum[scale_idx])
                          * sqrtminmax_momentum[scale_idx];

        // ── Step 2: Gradient ─────────────────────────────────────────────────
        float_grad = float(grads[idx]);

        // ── Step 3: Momentum update: new_buf = mu*old + (1-mu)*grad ─────────
        new_buf = mu * old_buf + (1.0f - mu) * float_grad;

        // ── Step 4: Nesterov blend and write output ──────────────────────────
        //   Matches torch Muon: update = grad.lerp(buf, mu) = (1-mu)*grad + mu*buf
        float float_update = nesterov ? ((1.0f - mu) * float_grad + mu * new_buf)
                                      : new_buf;
        nesterov_update[idx] = scalar_t(float_update);
    } else {
        // Out-of-bounds thread: contribute 0 to max, sentinel to min.
        new_buf    = 0.0f;
        old_buf    = 0.0f;
        float_grad = 0.0f;
    }

    // ── Step 5: Block-level min/max reduction of new_buf ────────────────────
    int wid  = threadIdx.x / WARPSIZE;
    int lane = threadIdx.x % WARPSIZE;

    __shared__ float sharedMaxVal[32];
    __shared__ float sharedMinVal[32];

    cg::thread_block_tile<32> warpTile = cg::tiled_partition<32>(cg::this_thread_block());

    float maxVal = fabsf(new_buf);
    float minVal = fabsf(new_buf);
    // Out-of-bounds threads must not contribute to min.
    if (idx >= total_elements) {
        minVal = __int_as_float(0x7f7fffff);  // FLT_MAX as sentinel
    }

    // Warp-level reduction (skip zero values for min, like AdamW kernel)
    for (int i = warpTile.size() / 2; i > 0; i /= 2) {
        float rMax = warpTile.shfl_down(maxVal, i);
        float rMin = warpTile.shfl_down(minVal, i);
        maxVal = fmax(maxVal, fabsf(rMax));
        float absRMin = fabsf(rMin);
        minVal = (absRMin > 0.0f) ? fmin(minVal, absRMin) : minVal;
    }

    if (lane == 0) {
        sharedMaxVal[wid] = maxVal;
        sharedMinVal[wid] = minVal;
    }
    __syncthreads();

    __shared__ float shared_absmax;
    __shared__ float shared_absmin;

    // Second level: warp 0 reduces the per-warp results.
    maxVal = (threadIdx.x < blockDim.x / warpSize) ? sharedMaxVal[lane] : 0.0f;
    minVal = (threadIdx.x < blockDim.x / warpSize) ? sharedMinVal[lane] : 1e9f;

    if (wid == 0) {
        for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
            float rMax = __shfl_down_sync(0xFFFFFFFF, maxVal, offset);
            float rMin = __shfl_down_sync(0xFFFFFFFF, minVal, offset);
            maxVal = fmax(maxVal, fabsf(rMax));
            float absRMin = fabsf(rMin);
            minVal = (absRMin > 0.0f) ? fmin(minVal, absRMin) : minVal;
        }
        if (lane == 0) {
            shared_absmax = maxVal;
            shared_absmin = minVal;
        }
    }
    __syncthreads();

    // ── Step 6: Compute expansion exponent, requantize new_buf to FP8 ───────
    if (idx < total_elements) {
        const float fp8MaxVal = 448.0f;

        float finalMaxVal = shared_absmax + QUANT_MIN_VAL;
        float finalMinVal = shared_absmin + QUANT_MIN_VAL;

        // COAT dynamic-range expansion (eq. 3):
        //   ratio        = maxVal / minVal
        //   ratioUpper   = fp8Max^2 / 2
        //   k            = floor(log2(ratioUpper) / log2(ratio) * expand_min) / expand_min
        //   sqrtMM       = sqrt(maxVal * minVal)
        //   x_exp        = sign(x) * (|x| / sqrtMM)^k
        //   scale        = (maxVal / sqrtMM)^k / fp8Max
        float ratio        = finalMaxVal / finalMinVal;
        float sqrtMinMax   = sqrtf(finalMaxVal) * sqrtf(finalMinVal);
        float ratioUpper   = fp8MaxVal * fp8MaxVal / 2.0f;
        float exp_k        = floorf((log2f(ratioUpper) / log2f(ratio)) * expand_min)
                             / expand_min;

        int   sign_new     = 1 - 2 * signbit(new_buf);
        float new_buf_exp  = sign_new * powf(fabsf(new_buf) / sqrtMinMax, exp_k);
        float new_scale    = powf(finalMaxVal / sqrtMinMax, exp_k) / fp8MaxVal;

        // Quantize and store (same value written by all threads in block — safe)
        q_momentum[idx]              = static_cast<__nv_fp8_e4m3>(new_buf_exp / new_scale);
        scale_momentum[scale_idx]    = new_scale;
        expand_momentum[scale_idx]   = exp_k;
        sqrtminmax_momentum[scale_idx] = sqrtMinMax;
    }
}

void FP8_Muon_expand_cuda(
    torch::Tensor nesterov_update,
    torch::Tensor q_momentum,
    torch::Tensor scale_momentum,
    torch::Tensor expand_momentum,
    torch::Tensor sqrtminmax_momentum,
    torch::Tensor grads,
    float mu, int nesterov, int qgroup_size, int expand_min
) {
    int total_elements       = grads.numel();
    int total_scale_elements = scale_momentum.numel();

    AT_ASSERTM(qgroup_size == 128, "Only 128-element per-group quantization is supported");
    const int block_dim = 128;
    int grid_dim = (total_elements + qgroup_size - 1) / block_dim;
    AT_ASSERTM(grid_dim == scale_momentum.numel(),
               "scale_momentum shape mismatch with ceil(numel/group_size)");

    const dim3 blocks(grid_dim);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, grads.scalar_type(), "fp8_muon_expand", ([&] {
            fp8_muon_cuda_expand_kernel<scalar_t><<<blocks, block_dim>>>(
                nesterov_update.data_ptr<scalar_t>(),
                (__nv_fp8_e4m3*)q_momentum.data_ptr<at::Float8_e4m3fn>(),
                scale_momentum.data_ptr<float>(),
                expand_momentum.data_ptr<float>(),
                sqrtminmax_momentum.data_ptr<float>(),
                grads.data_ptr<scalar_t>(),
                mu, nesterov, qgroup_size, expand_min,
                total_elements, total_scale_elements
            );
        })
    );
}
