# SPDX-License-Identifier: Apache-2.0
"""E-033 (P5): KDA fused norm kernels — l2norm(q,k) pair + gated RMSNorm (o_norm).

Two mx.fast.metal_kernel JIT kernels replacing the eager fp32 norm chains in
Glm5NextLinearAttention (vendored language.py:317-318 l2norm pair; :341 o_norm).

Proven equivalent (e033_kda_kernels.py): l2norm-pair max|d| 7.5e-09/6.0e-08;
o_norm-gated max|d| 9.5e-07. Per-call: l2norm 3.0x, o_norm 5.5x faster.

#3939-safe: row reductions over head_dim=128, 32-lane simd groups, one
threadgroup per row, full row parallelism, no GEMM geometry touched.

Env-gated: OMLX_KDA_FUSED_NORMS=1 enables at the vendored call sites.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_L2_SRC = """
    using namespace metal;
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_grid.x;
    constexpr uint D = 128;
    constexpr uint LANES = 32;
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    float qv[4];
    float kv[4];
    for (uint j = 0; j < 4; ++j) {
        uint col = lane + LANES * j;
        qv[j] = q_in[row * D + col];
        kv[j] = k_in[row * D + col];
        q_sq += qv[j] * qv[j];
        k_sq += kv[j] * kv[j];
    }
    float q_sum = simd_sum(q_sq);
    float k_sum = simd_sum(k_sq);
    const float eps = 1e-6f;
    float rq = rsqrt(q_sum + eps);
    float rk = rsqrt(k_sum + eps);
    for (uint j = 0; j < 4; ++j) {
        uint col = lane + LANES * j;
        q_out[row * D + col] = qv[j] * rq * rsqrt_scale;
        k_out[row * D + col] = kv[j] * rk;
    }
"""

_ONORM_SRC = """
    using namespace metal;
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_grid.x;
    constexpr uint D = 128;
    constexpr uint LANES = 32;
    float x_sq = 0.0f;
    float xv[4];
    float gv[4];
    for (uint j = 0; j < 4; ++j) {
        uint col = lane + LANES * j;
        xv[j] = x_in[row * D + col];
        gv[j] = gate_in[row * D + col];
        x_sq += xv[j] * xv[j];
    }
    float x_sum = simd_sum(x_sq);
    float mean_sq = x_sum / (float)D;
    const float eps = 1e-6f;
    float inv = rsqrt(mean_sq + eps);
    for (uint j = 0; j < 4; ++j) {
        uint col = lane + LANES * j;
        float normed = xv[j] * inv * weight[col];
        out[row * D + col] = normed / (1.0f + exp(-gv[j]));
    }
"""

_l2_pair = mx.fast.metal_kernel(
    name="kda_l2norm_pair",
    input_names=["q_in", "k_in", "rsqrt_scale"],
    output_names=["q_out", "k_out"],
    source=_L2_SRC,
)

_onorm = mx.fast.metal_kernel(
    name="kda_onorm_gated",
    input_names=["x_in", "gate_in", "weight"],
    output_names=["out"],
    source=_ONORM_SRC,
)

enabled = os.environ.get("OMLX_KDA_FUSED_NORMS") == "1"


def l2norm_pair(q, k, scale):
    """q,k: [..., 128] fp32. Returns (q*scale-normalised, k-normalised) fp32."""
    orig_shape = q.shape
    qf = q.reshape(-1, 128)
    kf = k.reshape(-1, 128)
    N = qf.shape[0]
    qn_list, kn_list = _l2_pair(
        inputs=[qf, kf, mx.array(scale, dtype=mx.float32)],
        output_shapes=[(N, 128), (N, 128)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
    )
    return qn_list.reshape(orig_shape), kn_list.reshape(k.shape)


def onorm_gated(x, gate, weight):
    """x,gate: [..., 128]; weight: [128]. RMSNorm-gated-sigmoid in one kernel (fp32 out)."""
    orig_shape = x.shape
    xf = x.reshape(-1, 128)
    gf = gate.reshape(-1, 128)
    N = xf.shape[0]
    out_list = _onorm(
        inputs=[xf.astype(mx.float32), gf.astype(mx.float32), weight.astype(mx.float32)],
        output_shapes=[(N, 128)],
        output_dtypes=[mx.float32],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
    )
    return out_list[0].reshape(orig_shape)


if enabled:
    logger.info("kda_fused_norms: OMLX_KDA_FUSED_NORMS=1 — kernels armed")