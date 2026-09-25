# BENCHMARKS — GLM-5.3-Flash on Apple Silicon (M5 Ultra 256 GB, oMLX)

# BENCHMARKS — GLM-5.3-Flash on Apple Silicon (M5 Ultra 256 GB, oMLX)

> PROPOSED CONTENT for `public/BENCHMARKS.md` — drafted by the E-055 Phase-1
> audit (2026-09-25). Numbers are verbatim from the private BENCHMARKS.md and
> the ledger; all values below already appear in public README/LEDGER or are
> pure performance facts. Scrub-checked: no IPs, hostnames, usernames,
> absolute local paths, or internal-only experiment IDs beyond those the
> public LEDGER already carries. Parent review required before commit.

All numbers: streaming TTFT, usage-verified tokens, unique prompt per request
(prefix-cache-defeating), temp 0, fresh or explicitly-labelled server state.
The corpus tokenizes at **6.78 chars/token** on this model's tokenizer
(calibrated; folk "~4 chars/token" overstates prompt size by ~70%). Full trap
list — prompt-cache reuse, arm-ordering warmup, powermetrics windows,
sync instrumentation, microbench timing floors — with the fixes, is in
[docs/METHOD.md](docs/METHOD.md). Every methodological rule there exists
because it once produced a false result we published to ourselves.

## Headline (V1 candidate)

| Metric | Full prefill (control) | SpecPrefill k=0.4 | Reference (2× NVIDIA DGX Spark, EXL3 TP) |
|---|---|---|---|
| Prefill eff. @16K | ~880 tok/s | **1,630 tok/s** | — |
| Prefill eff. @24K | ~1,030 tok/s | **2,089 tok/s** | 1,567 tok/s |
| Prefill eff. @32K | ~990 tok/s | **2,018 tok/s** | — |
| Prefill eff. @56K | ~1,030 tok/s | **2,040 tok/s** | — |
| TTFT @24K (24,030 tok) | 22.95 s | **11.49 s** | ~13 s |
| TTFT @56K (56,089 tok) | 54.24 s | **27.50 s** | — |
| Decode @4K ctx | 28.3 tok/s | unchanged | 31.4 tok/s |
| Decode @20K ctx | 34.6 tok/s | unchanged | — |
| Peak memory | ~194 GB | +14.5 GB (drafter) | n/a |

"Prefill effective" = prompt_tokens / TTFT, with TTFT including the draft
scoring cost — the honest end-to-end number, not the sparse-prefill kernel
rate. SpecPrefill is the *approximate* mode: the draft model scores token
importance and the target prefills only the ~40% most important tokens; the
full-fidelity path remains available at all times as a one-flag fallback.

## Quality gate (matched, six task families)

Matched-context quality: **0.942 (SpecPrefill k=0.4) vs 1.000 (full prefill)**
mean normalized task quality across six realistic families — tool-calling,
code editing, arithmetic, summarization at exact parity; exhaustive
long-context fact recall slightly reduced (that is where a 60%-token
approximation pays, and it is task-shaped, not uniform). Gate your own
workload: regenerate with `scripts/e040_quality.py` (deterministic scorers,
identical-content arms, 2048-token budget, enforced output formats).

## Decode — not one number

- **Short-context:** ~39 tok/s in the fast serving mode (oMLX ordinary decode 28–35 tok/s across 4–20K ctx historically; ~39.5 short-ctx best observed).
- **Mid-context:** **bimodal — 33.4 vs ~26 tok/s** (a ~21% spread). The fast/slow gate flips stochastically *per request* (autocorrelated runs, not coin flips), is localized to the ctx>2K decode path, and is invisible at INFO-level logging. **Caveat for reproduction: any mid-context decode A/B needs n≥10 paired arms or the bimodality will eat your conclusion.**
- **Production footprint (~26.8K ctx, sidecar warm):** ~31–32 tok/s.
- **Cross-engine ceiling statement:** ~31–35 tok/s across all measured engines (oMLX MLX 30–35; ds4 Q4 31 flat 2K→12K; llama.cpp GGUF+MTP historical 44.9 short / 42.2 @9K at −1.0 quality from quantization). The embedded-MTP lane measured decode-*negative* (verify pass costs more than an ordinary token); see [docs/DECODE.md](docs/DECODE.md) for the bandwidth physics and the open paths.

## Timeline of canonical milestones

| Date | Change | 24K prefill | Regenerate with |
|---|---|---|---|
| 09-23 | oMLX dev4 baseline (no native kernels — silent dense fallback) | 693–940 tok/s, degrading | — |
| 09-24 | Native kernel transplant (precompiled, from official 0.6.4 distribution) | ~1,025 tok/s (+46% long-ctx), flat to 56K | `scripts/spec_ladder.py` (control arm) |
| 09-25 | SpecPrefill first wiring (E-025/026/027) | 2,455 tok/s warm (bimodal scorer blocker) | `scripts/spec_ab2.py` |
| 09-25 | Draft-scoring fixes (lm-head skip, pool-keep, vectorized selection) | **2,018–2,089 tok/s stable, all sizes** | `scripts/spec_ladder.py` |

## Reproducibility

All harnesses are client scripts against the OpenAI-compatible endpoint
(`http://localhost:8008` — constants at the top of each file). Server launch
and environment for the SpecPrefill-capable build are specified in
[LEDGER.md](LEDGER.md) §"known-good configuration".

| Script | Measures | Invocation |
|---|---|---|
| `scripts/spec_ladder.py` | k=0.4 vs control across 8K/16K/24K/32K/56K, unique prompt per request, + 256-token decode regression at 24K | `python3 scripts/spec_ladder.py` |
| `scripts/spec_ab2.py` | keep-rate sweep 0.4/0.3/0.2 vs control at ~24K, per-arm receipts | `python3 scripts/spec_ab2.py` |
| `scripts/spec_reps.py` | rep'd, interleaved A/B with server-receipt decomposition (scoring / sparse-prefill / TTFT from server log) | `python3 scripts/spec_reps.py` |
| `scripts/spec_order.py` | arm-order / first-request-warmup confound probe (~60 s one-time draft materialization) | `python3 scripts/spec_order.py` |
| `scripts/spec_recall2.py` | fact-recall gate, fixed-format answers, 2 reps per arm (control + k=0.4) | `python3 scripts/spec_recall2.py` |
| `scripts/e040_quality.py` | the six-family matched quality gate behind the 0.942/1.000 number | `python3 scripts/e040_quality.py` |
| `scripts/hc_ab.py` | example patch-evaluation ladder (single-GEMM HyperConnection A/B; measured end-to-end neutral) | `python3 scripts/hc_ab.py` |
| `scripts/e033_kda_kernels.py` | offline kernel-equivalence + slope-fit microbench (needs `mlx`; no server) | `python3 scripts/e033_kda_kernels.py` |

Numbers above were produced by exactly these scripts (then-current versions)
against the configuration below. The public LEDGER records which experiment
used which harness.

## Environment (V1 candidate)

- Hardware: Apple Mac Studio **M5 Ultra** (80-core GPU), **256 GB** unified memory, macOS 27.0
- Runtime: **oMLX 0.7.0.dev4**, **MLX 0.32.0**, **Python 3.11**
- Target checkpoint: **PipeNetwork GLM-5.3-Flash MLX mixed-4/8-bit (~170 GB)**
- Drafter (SpecPrefill scoring): **avlp12/GLM-5.3-Flash-Alis-MTP-Drafter** (MTP block reused as a standalone scorer)
- Native Metal kernels: **precompiled binaries transplanted from the official oMLX 0.6.4 DMG** (cpython-311 / mlx-0.32.0-built) — vendored binaries, *not* source-built; no Xcode Metal toolchain is needed or present
- Serving flags for the headline config: SpecPrefill keep=0.4 (threshold 8192 tokens, default-armed), draft-scoring pool retention, draft-cache sidecar — per LEDGER §"known-good configuration"

## Known caveats

- SpecPrefill scoring adds **0.6–4.8 s per uncached request** (draft-model cost) and the first SpecPrefill request after server load pays a one-time ~60 s draft materialization (warm up before timing).
- The draft prefix-cache does not persist across different prompts (exact-match keyed; snapshot layout unsupported) — multi-turn conversations re-score every turn; the sidecar benefits identical re-requests only.
- Fresh-server vs long-lived-server control was A/B'd and is identical (early "fresh-server 2×" reports were an arm-ordering confound).
- Decode figures carry the mid-context bimodality caveat above.
