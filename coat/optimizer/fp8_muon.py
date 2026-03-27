# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""CoatMuon: COAT FP8-compressed momentum buffer for the Muon optimizer.

The Muon optimizer (Keller Jordan, 2024) maintains a single momentum buffer
per 2-D parameter.  This implementation stores that buffer in FP8
(float8_e4m3fn) using COAT's dynamic-range expansion to maximise quantisation
fidelity, yielding ~48 % memory savings on optimizer state.

Kernel-backed step (requires compiled qoptim_cuda):
  fp8_muon_expand_step  — with COAT dynamic-range expansion  (recommended)
  fp8_muon_step         — simple per-group absmax scaling     (ablation)

Newton-Schulz orthogonalisation is performed in PyTorch (it is a sequence
of matrix multiplications and cannot be element-wise fused into the kernel).
Polar Express orthogonalisation replaces Newton-Schulz with per-step
coefficients for faster convergence (enabled via ``polar_express=True``).
NorMuon variance reduction and cautious weight decay are optionally applied
after orthogonalisation.

The rest of the step — dequantize, lerp, Nesterov blend, requantize — is
fully fused in a single CUDA kernel launch per parameter.

Install:
    cd coat/optimizer/kernels && pip install -e .
"""

from __future__ import annotations

import math

import qoptim_cuda  # hard dependency — must compile from coat/optimizer/kernels/
import torch
from torch.optim import Optimizer

try:
    from torch.optim._muon import _zeropower_via_newtonschulz, _adjust_lr
except ImportError:  # pragma: no cover — keep in sync with torch/optim/_muon.py
    def _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps):
        a, b, c = ns_coefficients
        G = grad.bfloat16()
        if G.size(0) > G.size(1):
            G = G.T
        G.div_(G.norm().clamp(min=eps))
        for _ in range(ns_steps):
            gram = G @ G.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            G = torch.addmm(G, gram_update, G, beta=a)
        if grad.size(0) > grad.size(1):
            G = G.T
        return G

    def _adjust_lr(lr, adjust_lr_fn, param_shape):
        A, B = param_shape[:2]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            return lr * math.sqrt(max(1, A / B))
        elif adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(A, B))
        return lr


# ── Polar Express coefficients (one per Newton-Schulz iteration) ─────────────
_polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _polar_express_orthogonalize(G: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Polar Express orthogonalisation with per-step coefficients."""
    X = G.bfloat16()
    X = X / (X.norm() * 1.02 + 1e-6)
    if G.size(0) > G.size(1):
        for a, b, c in _polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in _polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


class CoatMuon(Optimizer):
    """Muon optimizer with COAT FP8-compressed momentum buffer.

    Includes optional Polar Express orthogonalisation (per-step coefficients),
    NorMuon variance reduction (``beta2``), and cautious weight decay.

    Optimizer state per 2-D parameter:
        q_momentum       (numel,)             float8_e4m3fn   — FP8 momentum
        scale            (ceil(numel/G),)     float32         — per-group scale
        k_expand         (ceil(numel/G),)     float32         — per-group expansion k
        sqrt_minmax      (ceil(numel/G),)     float32         — per-group sqrt(min*max)
        second_momentum  (A,1) or (1,B)       float32         — NorMuon (if beta2 > 0)

    Memory: ~1.05 bytes/element  vs  2 bytes/element for BF16 Muon  (~48 % saving).

    All hyperparameters are identical to ``torch.optim.Muon``; the COAT-specific
    ones (``use_expansion``, ``group_size``, ``expand_min``) default to the
    values used in the paper and can usually be left unchanged.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
        polar_express: bool = False,
        beta2: float | None = None,
        cautious: bool = False,
        # COAT-specific
        use_expansion: bool = True,
        group_size: int = 128,
        expand_min: int = 16,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum must be >= 0, got {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if beta2 is not None and not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"beta2 must be in [0, 1] or None, got {beta2}")
        if polar_express and ns_steps > len(_polar_express_coeffs):
            raise ValueError(
                f"ns_steps must be <= {len(_polar_express_coeffs)} with "
                f"polar_express=True, got {ns_steps}"
            )
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(
                f"adjust_lr_fn must be 'original', 'match_rms_adamw', or None; "
                f"got {adjust_lr_fn!r}"
            )
        if group_size != 128:
            raise ValueError(
                f"group_size must be 128 (CUDA kernel constraint); got {group_size}"
            )
        if expand_min < 1:
            raise ValueError(f"expand_min must be >= 1, got {expand_min}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            polar_express=polar_express,
            beta2=beta2,
            cautious=cautious,
            use_expansion=use_expansion,
            group_size=group_size,
            expand_min=expand_min,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"CoatMuon only supports 2D parameters; "
                        f"got parameter with shape {p.size()}"
                    )

    def _init_state(self, p: torch.Tensor, state: dict, group: dict) -> None:
        """Lazily initialise FP8 momentum state for parameter ``p``."""
        numel    = p.numel()
        gs       = group["group_size"]
        n_groups = math.ceil(numel / gs)

        # FP8 quantized momentum — all zeros dequantize to zero regardless of scale.
        state["q_momentum"] = torch.zeros(
            numel, device=p.device, dtype=torch.float8_e4m3fn
        )
        # float32 required by the CUDA kernel (data_ptr<float>()).
        # Initialise scale = 1/fp8Max so that the all-zero FP8 tensor yields
        # all-zero momentum without division-by-zero.
        state["scale"] = torch.full(
            (n_groups,), 1.0 / 448.0, device=p.device, dtype=torch.float32
        )
        state["k_expand"] = torch.ones(
            n_groups, device=p.device, dtype=torch.float32
        )
        state["sqrt_minmax"] = torch.ones(
            n_groups, device=p.device, dtype=torch.float32
        )
        # NorMuon second momentum — only allocated when beta2 is set.
        if group.get("beta2"):
            A, B = p.shape
            sm_shape = (A, 1) if A >= B else (1, B)
            state["second_momentum"] = torch.zeros(
                sm_shape, device=p.device, dtype=torch.float32
            )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr             = group["lr"]
            mu             = group["momentum"]
            wd             = group["weight_decay"]
            nesterov       = group["nesterov"]
            ns_coefs       = group["ns_coefficients"]
            ns_steps       = group["ns_steps"]
            eps            = group["eps"]
            adjust_lr_fn   = group["adjust_lr_fn"]
            polar_express  = group["polar_express"]
            beta2          = group["beta2"]
            cautious       = group["cautious"]
            use_expansion  = group["use_expansion"]
            gs             = group["group_size"]
            expand_min     = group["expand_min"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CoatMuon does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state, group)

                grad = p.grad.bfloat16()

                # Allocate output buffer for the fused kernel.
                # Shape matches p (2D) so _zeropower_via_newtonschulz accepts it directly.
                update = torch.empty_like(p, dtype=torch.bfloat16)

                # ── Fused kernel: dequantize → lerp → Nesterov → requantize ──────
                # Handles all FP8 bookkeeping; writes the Nesterov-blended update
                # to `update` in BF16.  Orthogonalisation (NS or PE) is a
                # sequence of matrix multiplications so it stays in PyTorch below.
                if use_expansion:
                    qoptim_cuda.fp8_muon_expand_step(
                        update,
                        state["q_momentum"],
                        state["scale"],
                        state["k_expand"],
                        state["sqrt_minmax"],
                        grad,
                        float(mu),
                        int(nesterov),
                        gs,
                        expand_min,
                    )
                else:
                    qoptim_cuda.fp8_muon_step(
                        update,
                        grad,
                        state["q_momentum"],
                        state["scale"],
                        float(mu),
                        int(nesterov),
                        gs,
                    )

                # ── Orthogonalisation ────────────────────────────────────────────
                if polar_express:
                    update = _polar_express_orthogonalize(update, ns_steps)
                else:
                    update = _zeropower_via_newtonschulz(update, ns_coefs, ns_steps, eps)

                # ── NorMuon variance reduction ───────────────────────────────────
                if beta2:
                    A, B = p.shape
                    red_dim = -1 if A >= B else -2
                    v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
                    red_dim_size = update.size(red_dim)
                    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
                    sm = state["second_momentum"]
                    sm.lerp_(v_mean, 1.0 - beta2)
                    step_size = sm.clamp_min(1e-10).rsqrt()
                    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
                    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
                    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
                    update = update * final_scale.to(update.dtype)

                # ── Parameter update ──────────────────────────────────────────────
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape)
                if cautious:
                    mask = (update * p) >= 0
                    p.sub_(adjusted_lr * update + lr * wd * p * mask)
                else:
                    p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-adjusted_lr)

        return loss
