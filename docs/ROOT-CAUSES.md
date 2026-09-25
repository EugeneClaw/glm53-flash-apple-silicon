# ROOT CAUSES — four non-obvious failures found and fixed

Each of these was invisible in logs, plausible-sounding when wrong, and
produced a large measured delta once fixed. Written up because the next
person will hit variants of the same four.

## RC-1: Silent native-kernel fallback (46% prefill)

**Symptom:** prefill degraded with context length (1,139 → 693 tok/s by 35K);
upstream's "30× with kernels" claim nowhere in sight.

**Root cause:** oMLX source installs ship the Metal kernel *sources*
(`csrc/`) but the serving path silently falls back to generic MLX ops unless
the precompiled kernel extension is present. No error, no log line —
`sparse_mla_attention()` returned `None` on 297/297 calls and the model ran
dense-equivalent attention. The kernels are gated by `has_symbol()` checks
that fail quietly.

**Fix / verification:** transplant the precompiled binaries from the
official oMLX release DMG into `custom_kernels/glm_moe_dsa/` (cpython-311 +
mlx 0.32.0 ABI — a source venv on 3.13 will not load them; *another* silent
failure mode: `has_symbol()` returns False and the fallback resumes). Verify
with a one-liner: `kernels.fast.native_available()` must be True, and
instrumented call-site logs must show kernel hits on every call.

**Result:** flat prefill to 56K+ (was knee+30%); +46% at long context.

**Lesson:** after ANY runtime or checkpoint change, re-verify kernel
engagement explicitly. Silent fallback is this stack's default behavior, not
its error path.

## RC-2: Profiling-window artifact (a diagnosis inverted)

**Symptom:** system profiling reported the GPU ~94% idle during prefill;
power draw ~20 mW. Conclusion at the time: "the wall is CPU-side dispatch;
the GPU is starved."

**Root cause:** the `powermetrics` sampling window ended before the measured
prefill started (window derived from when sampling began, not when the
workload ran). Window-matched re-measurement: **GPU 88–98% busy, 104–124 W**
throughout prefill.

**Corroborating evidence (three independent methods):**
- build-vs-eval isolation: building the step's graph ~6 ms; executing ~1,870 ms
- `sample(1)` call graph: inference thread in `cvwait` *waiting on GPU completion*
- decode cross-check: the same ~1,600-op chain at decode rates is impossible if launch gaps dominated

**Consequence of the correction:** the entire optimisation program flipped
from "reduce CPU-side dispatch" (host fusion ceiling ~0.4%) to "reduce GPU-side
serial execution" — which is what produced the eventual wins.

**Lesson:** a system-profiler window that doesn't overlap the workload
doesn't produce a *smaller* estimate — it produces an *inverted* one. Anchor
traces to workload timestamps.

## RC-3: Per-chunk buffer-pool re-creation under memory pressure (130×)

**Symptom:** the speculative-prefill draft scorer was bimodal on *identical*
work: 0.4 s or 44–73 s for a 24K-token scoring pass, same weights, same
shapes, same code path — sometimes within minutes of each other.

**Root cause chain:**
1. The draft loop called `mx.clear_cache()` after every chunk (a habit valid
   at low memory pressure: it returns scratch to the pool).
2. Under near-cap footprint (~193 GB of a ~222 GB cap), every clear forced
   full Metal buffer-pool re-creation for the next chunk.
3. Three red herrings eliminated first (each with receipts): attention-path
   selection (deterministic per chunk index), quadratic fallback cost
   (analytic bound 300× too small), and `_compute_importance` cost (<2 ms).

**Fix:** keep the pool across chunks (`SPEC_PREFILL_KEEP_POOL`), clear once
per scoring pass; also skip the draft's full-vocabulary head projection on
chunk calls (its 2.6 TFLOP + 0.63 GB allocation was discarded — ~75% of the
fast-mode compute) and vectorize a per-chunk mean that issued ~750 scalar
`.item()` syncs.

**Result:** bimodality eliminated; scoring flat at ~0.6–1.7 s.

**Lesson:** the target model's scheduler had *already* disabled its per-chunk
clear for exactly this reason (with a comment). Read the comments on the code
you're about to reuse; the same pathology had been found and fixed one layer
up.

## RC-4: Dispatch count ≠ GPU time (the fusion nulls)

**Symptom:** a dispatch census attributed 41% of all Python dispatches to one
component (a hyper-connection tiled matmul chain, 3,060 dispatches per
2,048-token chunk, ~123 GB/chunk of fp32 tile traffic). Microbenches showed
the replacement (a single pre-packed GEMM) 3.96× faster per call with 5×
less traffic and numerical equivalence (max diff 2.9e-5).

**Result of shipping it: end-to-end neutral.** Measured again with fused
norm kernels (equivalent to 7.5e-9, 3–5.5×/call): neutral again.

**Why:** the component was ~3% of chunk GPU time — 41% of *dispatches* is
not 41% of *time* — and the wall is the per-op serial execution premium
(~1.16 ms/kernel-equivalent across ~1,600 ops/token), which point-fusions
don't remove: they remove *ops*, and the premium attaches to the boundaries
that remain (and to cache-object/quant-prep machinery *between* ops).

**What WOULD move it:** whole-graph capture/replay, or speculation that
amortizes boundaries across tokens (see DECODE.md).

**Lesson:** before building any fusion, reconcile the op-count share against
a measured GPU-time share. If they disagree, believe the GPU-time number.

---

## Bonus: RC-5 (methodology, not code) — the prefix-cache

The serving stack's radix prompt cache made every "A/B" with reused prompts a
cache-hit measurement. See METHOD.md §1. This one produced the fake 3×.
