# glm53-flash-apple-silicon

**Making GLM-5.3-Flash fast on Apple Silicon (M5 Ultra, 256 GB) — a receipted engineering log for prefill optimisation and decode advancement.**

Everything here is measured on real hardware with archived receipts. No hand-waving, no benchmarks-vibes. If a hypothesis died, the corpse is on display. If a number is claimed, a log line or JSON backs it.

## Starting point (measured, before this work)

| Metric | Value |
|---|---|
| Prefill @ ~24K tokens | **~700–940 tok/s** (degrading with context length: 693 @ 35K, knee at ~19K) |
| Decode | **~40 tok/s** short-context |
| Serving stack | oMLX 0.7.0.dev4 + vendored mlx-vlm (glm5_next) |
| Hardware | Mac Studio M5 Ultra, 256 GB unified memory |

Reference point for context: a 2× NVIDIA DGX Spark cluster running the same
model (EXL3, tensor-parallel) was measured at ~1,567 tok/s prefill.

## Result (measured, after this work)

| Metric | Before | After | |
|---|---|---|---|
| Prefill @ 16K | ~880 tok/s | **1,630 tok/s** | 1.9× |
| Prefill @ 24K | ~1,030 tok/s | **2,089 tok/s** | 2.0× |
| Prefill @ 32K | ~990 tok/s | **2,018 tok/s** | 2.0× |
| Prefill @ 56K | ~1,030 tok/s | **2,040 tok/s** | 2.0× |
| Long-context degradation | knee + 30% drop by 35K | **flat to 56K+** | — |

The headline mechanism is **SpecPrefill**: a draft-model scores token
importance and the target prefills only the ~40% most important tokens.
It is an *approximate* mode — matched-context quality was measured at
**0.94 vs 1.00 across six task families** (tool-calling, code editing,
arithmetic, summarization at parity; exhaustive fact-recall slightly reduced)
— with a one-flag full-fidelity fallback.

Decode was investigated across three serving engines; findings and the
honest bandwidth/latency analysis are in [docs/DECODE.md](docs/DECODE.md).

## Minimum performance targets (competitive bars)

These are the bars this project set for "worth shipping" — stated so others
can judge the result against the same yardstick:

- **Prefill: ≥ ~1,584 tok/s** at ~24K context (the 2× DGX Spark reference pace)
- **Decode: ≥ ~60–62 tok/s** (parity with competitive serving on this model class)

Prefill clears its bar with ~30% margin in SpecPrefill mode (and meets it
neither-mode at 16K). **Decode does not clear its bar** — see Known
limitations and BENCHMARKS.md for the honest numbers.

## What's in here

- **[LEDGER.md](LEDGER.md)** — the full experiment log (E-001…E-045): every hypothesis, measurement, verdict, and artifact reference. The interesting part is the *failed* experiments: two fusion programs (single-GEMM rewrite, fused norm kernels) were proven numerically perfect and 3–6× faster in isolation, yet **measurably neutral end-to-end** — the receipts explain why, and that explanation is the most valuable thing in this repo.
- **[docs/METHOD.md](docs/METHOD.md)** — benchmark methodology (what invalidates a benchmark on this stack: prompt-cache reuse, powermetrics sampling windows, sync-instrumented timing, arm-ordering warmup… every one of these produced a false result we caught).
- **[docs/DECODE.md](docs/DECODE.md)** — the decode investigation: kernel-chain latency analysis across three engines, MTP/speculative economics measured on-hardware, and the open paths.
- **[docs/ROOT-CAUSES.md](docs/ROOT-CAUSES.md)** — the four non-obvious root causes found (native-kernel silent fallback, a powermetrics sampling-window artifact that inverted a diagnosis, a buffer-pool re-creation pathology, and dispatch-count ≠ GPU-time).
- **scripts/** — the benchmark harnesses (prefill ladders, matched quality gates, A/B drivers, kernel equivalence tests).
- **patches/** — our serving-stack modifications, as reference implementations (draft-model adapter for speculative prefill; sidecar cache; gated experimental patches).

## Headline findings (for the impatient)

1. **oMLX source installs silently run fallback kernels** unless precompiled Metal kernels are present — a 46% long-context prefill difference, invisible in logs. (E-kernels)
2. **"GPU idle %"-style system profiling can invert your diagnosis**: a powermetrics sampling window that closes before the workload starts reads as "GPU 94% idle" when the GPU is actually 88–98% busy. We published the wrong conclusion first; the window-matched re-measurement is the one that held. (E-016→E-022)
3. **Dispatch-count ≠ GPU-time.** One component was 41% of all Python dispatches and ~3% of GPU time. Both a single-GEMM rewrite and fused norm kernels were proven equivalent (max diff ≤3e-5) and 3–6× faster per call, then measured **neutral end-to-end** — because the wall is per-op *serial execution premium* (~1.16 ms/kernel-equivalent), not per-op count. (E-030/E-031/E-033/E-037)
4. **A per-chunk `mx.clear_cache()` in a draft-model loop cost 130×** under memory pressure — the same lesson the target model's scheduler had already learned, unapplied to the draft path. Fixing it plus skipping a discarded full-vocab head projection turned a bimodal 0.4 s ↔ 73 s scorer into a flat ~0.6–1.7 s one. (E-031b/E-038)
5. **Speculative-prefill quality is task-shaped**: agent-relevant tasks (tool calls, code edits, arithmetic) at exact parity; exhaustive long-context fact recall is where a 60%-token approximation pays. Gate your own workload before adopting. (E-040)

## Known limitations

- **Decode is below the 60–62 tok/s target**: ~39 tok/s short-context
  (clean-state baseline), ~33.7 tok/s at ~9K context, ~31–32 at ~27K.
  Measured across three serving engines; the bandwidth-physics ceiling and
  the open paths are analyzed in [docs/DECODE.md](docs/DECODE.md).
- **Decode is bimodal at ctx > 2K**: requests land in a fast tier
  (33.5–33.8 tok/s) or a slow tier (23–28 tok/s) — a per-request gate on
  the sparse-attention decode path, characterized but mechanism unresolved.
- **SpecPrefill is approximate**: 0.94 relative quality vs full fidelity;
  exhaustive long-context fact recall is where the 60%-token budget shows.
  First use of the draft scorer after a cold start pays ~60 s one-time.
- **Mixed-4/8-bit checkpoint**: results are for this quantization; other
  quants will differ (a Q4_K GGUF lineage measured −1.0 quality on a
  scoring rubric in side testing).

## Hardware/software context for reproduction

- Mac Studio M5 Ultra (80-core GPU, 256 GB), macOS 27.0
- oMLX 0.7.0.dev4, MLX 0.32.0, Python 3.11
- Checkpoint: PipeNetwork GLM-5.3-Flash MLX mixed-4/8-bit (~170 GB)
- Native Metal kernels: precompiled binaries transplanted from the official oMLX 0.6.4 DMG (no Xcode Metal toolchain needed)
- Draft checkpoint for SpecPrefill: avlp12/GLM-5.3-Flash-Alis-MTP-Drafter (MTP block reused as a standalone scorer)

Individual environment values (hostnames, LAN IPs, ports, paths) are local
configuration and intentionally absent — see METHOD.md for what you need to
substitute.

## Status

Prefill: **V1 shipped** (the 2× above, quality-gated). Decode: investigated,
measured, honestly bounded — with the open paths documented for the next
person. See [docs/DECODE.md](docs/DECODE.md) §"Open paths" before assuming
it's done.

## License / use

Results and methodology free to use and build on. The serving-stack patches
reference upstream oMLX/mlx-vlm (Apache-2.0 lineage) — treat them as
reference implementations, not drop-in products.
