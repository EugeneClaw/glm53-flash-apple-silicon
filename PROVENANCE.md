# PROVENANCE & ROLLBACK

## What is ours vs upstream

| Component | Status | Origin |
|---|---|---|
| `scripts/spec_ladder.py`, `spec_recall2.py`, `spec_reps.py`, `spec_ab2.py`, `spec_order.py`, `hc_ab.py`, `e040_quality.py` | **ours** | mission-built benchmark harnesses |
| `scripts/e033_kda_kernels.py` | **ours** | Metal kernel ladder + equivalence harness (JIT `mx.fast.metal_kernel`, no toolchain) |
| `patches/glm_draft_adapter.py` | **ours**, adapted | standalone 1-layer GLM drafter assembled from the target's own classes + an external MTP-drafter checkpoint; **valid for importance scoring only — NOT generation** (its HyperConnection matrices are absent from MTP checkpoints; see LEDGER E-053) |
| `patches/kda_fused_norms_patch.py` | **ours** | two fused RMSNorm Metal kernels (equivalence ≤7.5e-9); decode-neutral as shipped (E-048) |
| `patches/hc_single_gemm.py` | **ours** | HyperConnection single-GEMM prefill rewrite (equivalent to 2.9e-5, neutral end-to-end at production chunk size; opt-in `OMLX_HC_SINGLE_GEMM=1`) |
| `patches/e038_fixes.py` | **ours** | draft-scorer stability fixes (keep-pool, head-skip, vectorized reduce) — the changes behind SpecPrefill's flat scoring cost |
| Native Metal kernels (`custom_kernels/glm_moe_dsa/*`) | **vendored, precompiled** | transplanted from the official oMLX 0.6.4 DMG — cpython-311 + mlx-0.32 ABI; NOT source-built; require `has_symbol()` engagement checks after any runtime change (ROOT-CAUSES RC-1) |
| Serving tree (`omlx/`, vendored `mlx_vlm` glm5_next) | **upstream + local patches** | oMLX 0.7.0.dev4; our patches are env-gated new files under `omlx/patches/` + SpecPrefill (`omlx/specprefill/`) — none modify upstream files in place except documented vendored-tree branches |

## Environment

- macOS 27.0, Mac Studio M5 Ultra (80-core GPU, 256 GB unified)
- Python 3.11 (venv), MLX 0.32.0, oMLX 0.7.0.dev4
- Checkpoint: PipeNetwork GLM-5.3-Flash MLX mixed-4/8-bit (~170 GB), `num_nextn_predict_layers: 1` declared but head shipped externally
- Drafter: avlp12/GLM-5.3-Flash-Alis-MTP-Drafter (29 tensors, `mtp.*`)
- Launch (env only survives via the python -c wrapper — the console-script
  binary silently drops environment variables, LEDGER E-048):

```
cd <omlx-tree> && SPEC_PREFILL_KEEP_POOL=1 OMLX_DRAFT_CACHE_SIDECAR=1 \
  nohup <venv>/bin/python3 -c "import sys; sys.path.insert(0,'<omlx-tree>'); \
  from omlx.cli import main; sys.exit(main())" \
  serve --model-dir <model-dir> --host 127.0.0.1 --port 8008 \
  > /tmp/omlx-canonical.log 2>&1 < /dev/null &
```

## Rollback instructions

Every experimental feature is env-gated and defaults OFF. Rolling back to
stock behavior is environment-only — no code changes, no rebuilds:

| To disable | Do this |
|---|---|
| SpecPrefill (approximate prefill) | `"specprefill": false` per request, or unload the drafter (full-fidelity is the default path when the drafter is absent) |
| Draft-cache sidecar | unset `OMLX_DRAFT_CACHE_SIDECAR` and restart |
| Scorer buffer keep-pool | unset `SPEC_PREFILL_KEEP_POOL` (restores per-chunk clears — slower under memory pressure, the E-031b pathology) |
| KDA fused norms | unset `OMLX_KDA_FUSED_NORMS` (they are inert unless armed) |
| HC single-GEMM | unset `OMLX_HC_SINGLE_GEMM` (default off) |
| Full code rollback | `git checkout internal-v1-rc -- patches/ scripts/` in this repo; serving-tree patches are separate files — delete the specific `omlx/patches/<name>/` dir to remove one |

**Nuclear option:** re-clone oMLX, re-transplant the DMG kernels
(`custom_kernels/glm_moe_dsa/`), re-apply the two env flags you want. The
benchmark scripts regenerate every headline number from scratch
(BENCHMARKS.md → invocations).

**Verification after ANY rollback or runtime change:** native kernels
silently fall back when absent (RC-1). Confirm engagement with
`kernels.fast.native_available()` → True and a prefill spot-check against
BENCHMARKS.md numbers before trusting any comparison.
