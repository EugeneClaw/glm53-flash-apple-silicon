# DECODE — investigation, economics, and open paths

Decode on this model/hardware starts at ~40 tok/s (short context), ~28–35
tok/s at 4–20K context. This document is the honest account of the
investigation: what the physics says, what three engines measure, why the
obvious lever (MTP) doesn't pay on this hardware today, and which paths
remain open.

## 1. The physics: bandwidth says ~100+ tok/s is possible

Rough per-token read volume for GLM-5.3-Flash at mixed-4/8-bit:
~12B active parameters/token → ~6–7 GB/token. Against the M5 Ultra's
measured ~800–1,000 GB/s effective bandwidth, the *bandwidth* ceiling is
**~110–150 tok/s**. We measure ~31. So decode runs at ~25% of achievable
bandwidth.

Where does the rest go? The model executes **~1,600 Metal kernel
dispatches per token** (measured via graph census). At the measured
~16 µs/kernel serial latency for this dependency chain: 1,600 × 16 µs ≈
26 ms/token ≈ **38 tok/s** — which is almost exactly what we measure. The
wall is not bandwidth; it is **kernel-chain latency**: thousands of tiny
serially-dependent ops per token, each paying launch/sync/scratch overhead.

Corroborating arithmetic: standalone replicas of the heavy components
(KDA scan 0.9 ms, MoE 2.0 ms, attention internals) sum to a small fraction
of the 29 ms/token wall — the premium lives *between* ops.

## 2. What was measured across engines

| Engine | Quant | Decode short | Decode ~9K | Notes |
|---|---|---|---|---|
| oMLX (MLX, this work) | mixed-4/8 | ~40 | 28–35 | the stack this repo optimizes |
| ds4 (antirez, native C/Metal) | Q4_K (own lineage) | 31 | 28–31 | built from source; own GGUF; embedded MTP |
| ds4 + MTP (draft=1) | Q4_K | — | 28–29 | **slower than ordinary** |
| ds4 + MTP (draft=3) | Q4_K | — | 28.7–29.1 | no change |
| llama.cpp fork (glm5next/upstream) + GGUF + MTP | UD-Q4_K_XL | 44.9 | 42.2 | historical receipt, at a measured −1.0 quality cost |

### Why MTP doesn't pay on ds4/Metal today (measured, with counters)

ds4 exposes the full speculative cycle. At draft-depth 1:
- verify pass: **47.1–48.0 ms**
- head+draft: **5.3 ms** (accepted) / **36.4 ms** (rejected)
- acceptance: **70%** (1120 accepted / 479 rejected)

Expected throughput = tokens/round ÷ round-time = 1.7 ÷ 62.4 ms ≈ **27 tok/s**
— matching the observed 28–29. The verify pass alone costs more than 1.5
ordinary tokens. At draft-depth 3 the verify gets cheaper (~41 ms) but
multi-token drafts do not land (acceptance counters show single-token
matches), so nothing compounds.

Compare the M5 Max published receipt (Q2 weights, 34.45 → 41.97 w/ MTP):
Q2's smaller weight read makes ordinary tokens cheaper (~29 ms) *and* the
verify proportionally smaller. **MTP economics are a function of
(verify cost ÷ ordinary-token cost) and acceptance rate** — on this
checkpoint at Q4 on Metal, that ratio is >1, so speculation loses.

The llama.cpp fork's historical +23% was at the same quant class — its
speculative loop is more efficient (fused verify, CUDA-graph-style
submission). That gap is engine maturity, not physics.

## 3. Open paths (ranked by expected-value-per-effort)

### Path A: Speculative *decode* with the SpecPrefill draft adapter (untested, infrastructure exists)

This project already built the missing piece llama.cpp had and oMLX lacks: a
**working GLM draft model** (`patches/` glm_draft_adapter — a standalone
1-layer scorer assembled from the MTP-drafter block + target embed/lm_head,
proven numerically). oMLX's decode-side speculative machinery
(`mlx_lm_mtp` patch layer, depth-k drafting, fused verify) exists and works
for other model families.

**Experiment:** wire `glm_draft_adapter` output as a decode drafter into
oMLX's MTP chain; measure acceptance rate × draft cost per verified token.
The economics differ from ds4's embedded MTP: the draft here is a *single
sparse-MLA layer with KV-latent reuse from the target's own cache objects*,
so head+draft may be much cheaper than ds4's 5.3–36 ms. If accept ≈ 70% and
draft+verify ≈ 1.3 ordinary tokens, that's ~1.5× ≈ 45–52 tok/s. **This is
the cheapest untested path to >60.**

### Path B: Decode-side kernel fusion (the prefill nulls do NOT transfer)

Our two fusion nulls were prefill experiments: at 2,048 tokens/chunk,
per-op overhead amortizes 2,048×, so killing ops buys nothing. **Decode has
no amortization** — every one of the ~1,600 boundaries per token is paid in
full. The fusion economics invert.

Highest-value candidates (decode census pending in the ledger):
- RMSNorm-gated + l2norm chains (already written and proven equivalent —
  `patches/` kda_fused_norms — measured neutral at prefill; **never measured
  at decode**, where they are a larger fraction of the step)
- The hyper-connection `_mix` single-GEMM (proven equivalent, +0 bytes) —
  decode runs the tokenwise branch, different geometry
- MoE gather/sort glue at batch=1 (sorting 16K routes is a prefill shape;
  decode routes ~top-8×1 — a completely different, much cheaper shape that
  may be fusion-friendly)

**Estimate if the chain shrinks 4×:** 26 ms/token → ~7–10 ms → 60–90+ tok/s.
This is where the bandwidth ceiling becomes reachable. Requires either
`mx.fast.metal_kernel` fusions validated at T=1, or upstream
capture/replay.

### Path C: Batched/continuous decode (throughput, not latency)

oMLX supports batched sessions; decode t/s *aggregate* scales sub-linearly
today. If your workload tolerates batching, aggregate throughput beats
single-stream optimizations. Out of scope for a latency-focused mission but
noted.

### Path D: Quantization of the *decode* hot path

The mixed-4/8 checkpoint keeps some projections at 8-bit. A decode-specific
quant sweep (4-bit the decode-heaviest projections only) trades a small
quality delta for weight-read time. Unmeasured. Note the historical
caution: full Q4_K_XL GGUF cost a measured −1.0 quality point; targeted
projection-only quant is a different, smaller trade.

## 4. What we rule out (with receipts)

- **Command-buffer env tuning** (`MLX_MAX_OPS_PER_BUFFER` etc.): +3.0–3.5%
  decode, inside rep spread. Free to try, not a lever. (Ledger E-034/E-035)
- **More CPU-side dispatch efficiency**: the GPU is 88–98% busy; host prep
  is 0.4% of the wall. (E-022)
- **Speculative prefill tricks applied to decode**: different mechanism,
  see Path A for the correct adaptation.

## 5. If you pick this up

1. Reproduce the baseline with `scripts/` (METHOD.md first — ten ways to fool yourself are listed there).
2. Verify native kernel engagement after any stack change (RC-1).
3. Run the decode census (ledger E-046 when landed) to get *your* per-token dispatch count.
4. Try Path A first (draft exists, infra exists, S effort); Path B second (kernels already written, need decode-shape measurement).
5. Publish your nulls. The two fusion nulls in this repo cost real compute and are as useful as the 2×.
