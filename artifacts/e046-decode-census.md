# E-046 — GLM decode: native kernels vs silent fallbacks + decode dispatch census

Served model: **GLM-m48** (`glm5_next`, vendored compat patch, oMLX 0.7.0.dev4, mlx 0.32.0,
NAX GPU). Config: 45 layers = **34 KDA** (`linear_attention`) + **11 DSA**
(`deepseek_sparse_attention`); MLP: 3 dense + **42 sparse MoE** (288 experts, top-8,
**4-bit affine, group 64**); indexer top-k = 2048, kpool 4; HC (`mhc=True`, hc_mult 4) on
every layer; quant 4/64 throughout. Server untouched (read-only HTTP + log reads).

**Headline: decode is NOT silently falling back where it matters — but it is also NOT
running most of the transplanted GLM kernels.** All 21 native symbols are present and
ABI-alive, yet the decode graph uses only 3 of them (22 native calls/token at ctx>2048;
0 at ctx≤2048). The decode fallbacks that DO run (MoE `mx.gather_qmm`, plain SDPA,
pure-MLX KDA glue) are taken by *design* of threshold/geometry gates, not by missing
symbols.

---

## 1. Symbol inventory (transplanted binaries)

`_ext.cpython-311-darwin.so`: 23 Python-visible names (22 kernels + `abi_probe`).
`libomlx_glm_kernel_ops.dylib`: 22 C++ exports, all in `omlx::glm_kernels` (mangled
`__ZN4omlx11glm_kernels...`). Both agree; `omlx_glm_kernels.metallib` contains the Air
functions incl. the decode-specialized families:

| Python symbol (all PRESENT) | Metal functions (metallib) |
|---|---|
| `dsa_decode_scores` | `dsa_decode_scores_{bfloat16,float16}_{of32,osame}_h32_d128_t256` |
| `dsa_indexer_scores` | `steel_dsa_indexer_score_{bf16,fp16}_bm64_bn64_bk16_wm2_wn2` |
| `dsa_topk_indices` | `steel_dsa_topk_indices_*_topk{512,2048}_t1024` |
| `dspark_fp32_topk_indices` | `dspark_fp32_topk_indices_topk512_t256` |
| `glm_dsa_sparse_mla_attention` | `steel_sparse_mla_{bf16,fp16}_bk{128,256}_dc32_h{32,64}_d512_pe64_wm{4,8}` |
| `glm_dsa_exact_block_attention` | `omlx_glm_exact_attention_*_bq{16,32}_bk{8,16}_bd256_*` |
| `glm_dsa_q8_vup_flat` | affine block loaders (DeepseekAffineBlockLoaderIDF16b…) |
| `glm_moe_weighted_sum` | `moe_weighted_sum_tiled_*_topk_{6,8}_t_256` |
| `deepseek_mxfp4_gather_qmm_blocks/_pair_blocks/_pair_concat_blocks/_expert` | `deepseek_mxfp4_gather_{blocks,pair_blocks,pair_concat_blocks,expert}_rhs_*` |
| `deepseek_affine_gather_qmm_blocks/_pair_concat_blocks` | `deepseek_affine_gather_{blocks,pair_concat_blocks}_rhs_*` + `affine_qmm_t_head_flat_*` |
| `deepseek_v4_sparse_attention` | `deepseek_v4_sparse_attention_*_bk256_dc32_h64_d512_wm8` |
| `qwen4_qsa_*` (3), `dspark_ring_gemm/_rowwise_gemm/_exact_mxfp8_qmv_pair` | `qwen4_qsa_*`, `omlx_dspark_*` families |

**Absent from the metallib (and expected to be): no KDA/gated-delta kernels, no
hyper-connection kernels, no indexer `dsa_decode` blocks beyond the above** — KDA and
HC kernels are built at runtime via `mx.fast.metal_kernel` (see §2b/e).

## 2. Call-site map — what decode actually requires per component

Served implementation = **vendored glm5_next** (`omlx/patches/mlx_vlm_glm5_next_compat/
vendor/mlx_vlm/models/glm5_next/language.py`), NOT the `glm_moe_dsa` v32 model classes
(v32 contributes only `Model.sanitize` + `group_expert_select` to this path).

### (a) Sparse-MLA decode (L==1 branch, language.py ~641-663)
- **Kernel required: NONE.** The L==1 branch gathers top-k latent rows
  (`mx.take_along_axis`) + builds `sel_mask`, then falls through to the **L≤8
  gathered-SDPA path** (`mx.fast.scaled_dot_product_attention`, 11 calls/token).
- `sparse_mla_attention()` → gate `hasattr(glm_fast,"glm_dsa_sparse_mla_attention")`,
  but its geometry gate **rejects `L <= 1` unconditionally** (sparse_mla.py:434) → it is
  PREFILL-ONLY by construction. Symbol present; unreachable at decode.
- `exact_block_token_attention` / `q8_vup_flat`: prefill-only call sites (L>8 branch);
  symbols present, decode-irrelevant.
- **Verdict: no silent fallback — decode MLA is a gather+SDPA design, native kernels
  not involved.**

### (b) KDA / gated-delta decode (T=1 recurrent step)
- `Glm5NextLinearAttention.__call__` → `gated_delta_update(...)` → at T=1
  `gated_delta_kernel` = `mx.fast.metal_kernel("gated_delta_step_vec")` — a **runtime-
  compiled custom Metal kernel, not a transplanted symbol** (gated_delta.py:20-131).
  Present & engaged: 34 calls/token (= 34 KDA layers).
- Gates: `mx.default_device()==gpu`, `metal.is_available()`, `k.shape[-1]%32==0` — all
  pass. Fallback would be the sequential `_gated_delta_step_ops` loop (not taken).
- **No transplanted KDA kernel exists; the "native" KDA step is the metal_kernel
  custom. All surrounding glue (l2norm q/k, o_norm gated RMSNorm, conv, reshapes) is
  pure eager MLX.** `OMLX_KDA_FUSED_NORMS` (fuses l2norm-pair + o_norm) is **unset** on
  the server → fused-norm patch OFF.

### (c) MoE decode (batch=1 → 8 routes)
- Gates in `omlx/patches/deepseek_v4/switch_layers.py`:
  - native block kernels need `_native_block_kind(...)`:
    - mxfp4: `mode=="mxfp4"` — model is **affine 4-bit** → NO.
    - affine: `bits in (2,3)` AND `routes >= 1024` (`_AFFINE_NATIVE_MIN_ROUTES`) →
      model is **bits=4** AND decode has **8 routes** → NO on both counts.
  - pair/concat variants: same gates → NO.
  - `glm_moe_weighted_sum`: needs `do_sort` (`routes >= _SORT_MIN_ROUTES=64`) → 8 < 64
    → **NO** (proxy + IR agree: 0 decode-only `GlmMoeWeightedSum` nodes).
  - `_nax_prefers_stock(8)` = False (routes < 1024) — irrelevant, block kernels already
    declined.
- **Decode MoE runs stock `mx.gather_qmm` ×3 per layer = 123 IR GatherQMM nodes/token
  (+ `Glm5NextClampedSwiGLU` compiled silu).** All required symbols are PRESENT; the
  fallback is threshold-gated by design. Note the transplanted `affine_qmm_t_head_flat`
  Metal functions exist in the metallib but no code path reaches them at decode.

### (d) Indexer decode scores
- **The symbol `dsa_decode_scores` is NOT used by the served model** — it gates the
  v32 `Indexer` s==1 branch (deepseek_v32.py:308-324), and the vendored glm5_next tree
  contains zero references to it (grep-verified). It is present in the binaries but
  dead for this server.
- Actual decode indexer = `Glm5NextIndexer`:
  - ctx ≤ index_topk (2048): `bypass_short` → returns None early → **0 native calls**;
    attention stays dense-gathered.
  - ctx > 2048: `_native_scores` → gate `fast.has_symbol("dsa_indexer_scores")` →
    **native, PRESENT** (`steel_dsa_indexer_score_*`); `_native_topk` →
    `fast.has_symbol("dsa_topk_indices")` → **native, PRESENT**
    (`steel_dsa_topk_indices_*_topk2048`); gathered attention then runs
    `mx.fast.sdpa` — IR also shows `OMLXGlmDsaExactBlockAttention` nodes only from the
    inherited prefill subgraph (pair-diff delta = 0).
- Fallbacks if symbols were missing: `q@kᵀ` head-sum + `mx.argpartition` — not taken.

### (e) Hyper-connection decode
- `_hc_kernel` (fused Sinkhorn-collapse `mx.fast.metal_kernel`) + compiled
  `_hc_expand_op`; gates: gpu + metal available → engaged (45 + 45 per token).
  `decode_consistency.matmul` is armed-guarded (`set_armed`), default OFF → plain `@`.
- **No transplanted HC symbol exists; runtime metal_kernel + eager glue.**

## 3. Live verification

Gate probe (same venv/interpreter as server, same modules): **all 21 kernel symbols
`has_symbol=True` via `_ext`; abi_probe OK; `_EXT_MASK_FOLD=True`; `_EXT_MMA_SCORE=True`;
`import_error=None`.** `_FastDispatch` (v32 shim) `has()` = True for all probed names.
Env: `GLM_SPLIT_COMPILE` unset, `OMLX_KDA_FUSED_NORMS` unset, all
`OMLX_DEEPSEEK_*` thresholds at defaults. **No ATTN-PATH-style logging or GLM_* trace
env exists anywhere in the tree** — the only env-gated instrumentation is the threshold
knobs above plus `GLM_SPLIT_COMPILE`/`OMLX_KDA_FUSED_NORMS` (feature toggles, not logs).

Instrumented probe = `/tmp/e046/probe_decode.py` (loads nothing; reports gates).
Live server decode (64-token requests, short prompt):
- 12:36:37 — `Chat completion: model=GLM-m48, 64 tokens in 3.38s (18.9 tok/s), prompt: 26`
- 12:36:48 — `Chat completion: model=GLM-m48, 64 tokens in 3.41s (18.8 tok/s), prompt: 26`
- Both runs identical → warm-cache stable. (18.9 tok/s includes prefill+overhead inside
  the 3.4 s; prior 8-token runs on the same server at 26.8k prompt context show 31-32
  tok/s steady decode.) Log: `/tmp/omlx-e043.log` (newest by mtime).

## 4. Decode dispatch census (T=1, B=1; lazy build, zero GPU alloc, no eval)

Harness `/tmp/e046/census_t1.py` (clone of `/tmp/census_a.py`, decode schedule); full
data `/tmp/e046/census_t1_results.json`; IR ground truth via prefill/decode dot PAIR
DIFF (`/tmp/e046/census_t1b.py` → `pair_*.dot`) — the raw decode dot inherits the whole
prefill subgraph through cache deps, so native-primitive attribution must use the pair
delta.

| regime | Python dispatches/token | IR delta (decode-only) | native calls | compiled-inner virtual prims |
|---|---|---|---|---|
| ctx 257 (topk bypassed) | **2,600** | 6,675 | **0** | 2,601 |
| ctx 2561 (topk active) | **3,392** | 7,482 | **22** (11 dsa_indexer_scores + 11 dsa_topk_indices) | 2,601 |

Composition (topk-active): eager 2,998 + compiled 158 (silu 34, compute_g_safe 34,
_hc_expand_op 45, _ffn_block 45) + custom metal_kernel 79 (hc_sinkhorn_collapse 45,
gated_delta_step_vec 34) + native 22 + fast 135 (rms_norm 113, layer_norm 11, sdpa 11).
Per-layer: KDA 59 events, DSA 125 (bypassed) → 305 (active). Eval hits: none.

Top consumers of ONE decode step (topk-active): slice 432, astype 339, multiply 339,
reshape 338, quantized_matmul 236, add 190, fast:rms_norm 113, rsqrt 102, concatenate
101, transpose 78, zeros 77, split 68, sum 68, sigmoid 68, matmul 56.
(343→359 of the quantized_matmuls and ~2.5M of ~93-163M elems are projections; the
transpose/concat/slice mass is KDA+cache glue.)

**Comparison with the prefill census (~1,600 kernels/token claim, 7,520 dispatches per
2048-token chunk):** the decode graph is the SAME layer structure at T=1, but per-token
dispatch cost is inverted — prefill ≈ 3.7 dispatches per generated token vs decode
2,600-3,392 dispatches per token (~700×). Decode does NOT reuse the prefill sparse-MLA
kernel at all; its per-token boundary count is dominated by small-op glue, not GEMMs.
At the measured 31-32 tok/s steady decode (~31 ms/token), 2,600-3,392 boundaries imply
~9-12 µs/boundary — consistent with E-022's ~16 µs/kernel if only ~2/3 of the Python
dispatches are distinct GPU kernel launches (rest are view/reshape-class ops fused at
submission).

## 5. Top-5 decode kernel-boundary reduction candidates

*Caveat honored: prefill fusion nulls (F1-F5 in census_a) do NOT transfer — prefill
amortizes per-token overhead over 2048-token rows; at T=1 the per-boundary fixed cost
(~7-16 µs) is the whole story, so the ranking below is by decode boundaries removed.*

| # | candidate | now | after | boundaries/token saved | note |
|---|---|---|---|---|---|
| D1 | **KDA glue fusion — astype/reshape/multiply chain** (l2norm q/k + scale, o_norm gated RMSNorm, conv-out qkv reshapes) | ~2,000 of 2,600 dispatches (34 KDA × ~59) | ≤ 8/layer fused kernels | **~1,700 (65%)** | `OMLX_KDA_FUSED_NORMS=1` already implements part of this and is UNSET on the server — zero-code test at next restart; remaining glue needs a dedicated T=1 fused kernel |
| D2 | **Raise decode top-k bypass threshold** (dense gather-SDPA is exact below index_topk) or single fused decode-indexer kernel (scores+topk+gather+attn) | +792 dispatches/token once ctx>2048 (59→305/DSA layer) | dense: 0; fused: ~4/layer | 792 (at ctx>2k) | the 4 transplanted decode kernels (`dsa_decode_scores` etc.) are exactly this fusion but unreachable from the vendored path — wiring glm5_next's indexer to v32's s==1 branch is the highest-leverage reuse |
| D3 | **Fuse L==1 sparse-MLA**: extend `sparse_mla_attention` geometry gate (`L<=1 → None`) to accept T=1 (kernels `steel_sparse_mla_*` already handle qL=1) | gather(2048×512) + sel_mask + SDPA ≈ 60 ops × 11 layers | 1 kernel/layer | ~600 | deletes the 1M-elem-per-layer gather intermediate; same kernels already in metallib |
| D4 | **HC mix fn.T hoist**: `self.fn.T` re-transposed on every call (90/token, [16384,24] fp32 ≈ 36M elem traffic) | 90 transposes + 90 matmuls | 90 matmuls | 90 + ~72 MB/token traffic | trivial: transpose once at load |
| D5 | **Decode MoE router + 8-route gather**: fp32 gate dance + compiled `group_expert_select` + 3× `mx.gather_qmm` per layer | ~10 router ops + 123 gather_qmm | fused router (1 kernel) + batch-1 affine qmv path | ~80-150 | the transplanted `affine_qmm_t_head_flat_*` / block kernels target this shape but are gated off (bits=4 + routes<1024); a routes==8 fast path would engage them |

D1+D2+D3 ≈ 3,100 of ~3,400 boundaries (~90%) → upper-bound saving ~30-34 ms/token at
9-12 µs/boundary if fully serialized (realistically 15-40% wall win after overlap).

## Files created (all NEW; nothing existing modified)
- <studio-artifacts>, gate_probe_out.txt — gate/ABI/env probe
- <studio-artifacts>, census_t1.log, census_t1_results.json — decode census
- <studio-artifacts>, pair_pre256.dot, pair_dec257.dot, pair_pre2560.dot, pair_dec2561.dot — IR pair-diff ground truth
- <studio-artifacts>, census_dot_d2560.dot — raw decode dots (prefill-polluted; superseded by pair diff)
- <studio-artifacts> — this report
