# SPDX-License-Identifier: Apache-2.0
"""HyperConnection `_mix` single-GEMM patch (E-031, P2).

Replaces the tiled_linear prefill branch of
mlx_vlm.models.deepseek_v4.hyper_connection.HyperConnection._mix with a single
hoisted GEMM `z @ fn.T`. Proven equivalent (block-matrix algebra; equiv_test.py
5/5 PASS, max diff 1.3e-06 small / 2.9e-05 full-scale fp32 accumulation order).

Measured (hc_bench.py, M5 Ultra, T=8192): 2.46 ms -> 0.62 ms/call (3.96x),
traffic 2.74 GB -> 539 MB/call (5.08x). Per 2048-chunk: 34 -> 1 dispatches,
~37-62 GB -> 12 GB.

Design rules honoured (mlx#3939):
- Row-parallel GEMM geometry preserved (M=T rows, K=16384 unchanged) —
  this ENLARGES the safe grid rather than restructuring it.
- Decode branches (L<=8 tokenwise/direct) untouched — bit-identical.
- fn fp32 pin untouched: fn_t is a transpose VIEW (+0 bytes); no astype.
- apply_branch (L<=8 fused path) untouched.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_hc_single_gemm_patch() -> bool:
    """Patch HyperConnection._mix's prefill branch to a single hoisted GEMM.

    Returns True if the patch was applied (or was already applied).
    """
    global _applied
    if _applied:
        return True

    try:
        from mlx_vlm.models.deepseek_v4.hyper_connection import (
            HyperConnection,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("hc_single_gemm: mlx_vlm hyper_connection unavailable: %s", e)
        return False

    original_mix = HyperConnection._mix

    def _mix_single_gemm(self, z):
        if z.shape[1] <= 8:  # DECODE_BLOCK_SIZE (linear.py:5)
            # Decode: keep stock paths bit-identical.
            return original_mix(self, z)
        # Prefill: one GEMM against the load-time transpose view of fn.
        # fn stays fp32 (cast_predicate/sanitize pin); fn_t is a view (+0 bytes).
        fn_t = self.fn.T
        return z @ fn_t

    HyperConnection._mix = _mix_single_gemm
    _applied = True
    logger.info(
        "hc_single_gemm: HyperConnection._mix prefill branch -> hoisted GEMM "
        "(decode branches untouched)"
    )
    return True