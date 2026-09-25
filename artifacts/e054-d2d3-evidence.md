# E-054 — D2 vs D3 evidence package (NO implementation, NO server changes)

**Scope:** read-only analysis of studio:<studio-path> sources + E-046 artifacts, to
decide whether D2 (wire dead-coded `dsa_decode_scores` into the served decode path) or
D3 (extend `sparse_mla_attention` geometry gate to T=1) is worth implementing — or neither.
Raw traces: `<studio-artifacts>` (source excerpts + IR label diff + metallib symbols).

---

## Part 1 — D2: fuse the decode indexer score chain into `dsa_decode_scores`

### 1.1 Current decode path for the indexer scores (file:line)

Served model = vendored glm5_next, NOT the v32 classes. Per DSA layer per decode token at
ctx > `index_topk` (2048) — `Glm5NextIndexer.__call__`
(`omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py`):

| step | location | what runs |
|---|---|---|
| gate: bypass | language.py:494 | `bypass_short and total_max <= 2048` → returns None (short ctx: 0 native calls) |
| pool compress | language.py:460-469 | `_compress_windows` softmax/sum glue + `cache.update_and_fetch` (cache_extras.py:109) |
| chunk loop | language.py:518-519 | `q_chunk = q[:, c0:c1]` (S=1 → one chunk), `weights = linear_forward(self.weights_proj, ...)`, `* weight_scale`, `.astype()` |
| **scores** | language.py:519 → `_native_scores` (language.py:403-442) | **`fast.dsa_indexer_scores(qt, keys, weights, causal=False)`** — the steel GEMM (`steel_dsa_indexer_score_{bf16,fp16}_bm64_bn64_bk16_wm2_wn2`), preceded by `qt = q.transpose(0,2,1,3)` and **pad-to-multiple-of-64** on both Q and K (language.py:429-436), followed by an un-pad **slice** (language.py:437) |
| fallback (not taken) | language.py:521-527 | `q@kᵀ` + relu + weighted head-sum eager chain |
| **top-k** | language.py:532 → `_native_topk` (language.py:444-452) | `fast.dsa_topk_indices(scores[:,None], 512*…)` — native radix; **already engaged** |
| candidate mask | language.py:522-531, 534-545 | `valid_candidates` where, `pool_end` arithmetic, `mx.where(-1e30)`, tail-window concat, −1 padding, `valid_cur` where |

**The only part of this chain D2 can replace is the scores step** (transpose + pad +
steel GEMM + slice). Top-k is already native. The pool/mask bookkeeping around it is
model glue, not something `dsa_decode_scores` touches.

### 1.2 Op/boundary count D2 would replace

E-046 headline said "792 boundaries/token at ctx>2K" — but that is the **entire IR delta**
of the ctx-2561 graph over the ctx-257 graph, i.e. the cost of the top-k path *as a whole*
activating (indexer + candidate masking + the L==1 gathered attention that D3 targets +
MoE affine glue), not the indexer score chain alone.

Fresh re-derivation from E-046's own IR pair dots (`labeldiff.py`, traces/labeldiff_out.txt):
decode-only node multiset diff pair_dec2561 − pair_dec257 = **11,781 nodes** (≈1,071/layer
over 11 DSA layers), of which only 66 are the two native indexer primitives themselves
(`OMLXDSAIndexerScores` ×66, `OMLXDSATopKIndices` ×66 = 6/layer) and 332 are
`Qwen35QAffineQmmTPrimitive` (MoE input-proj glue, not indexer).

The score-chain portion D2 replaces, per layer per token, is bounded by:
- 1 transpose + 2 pads + 1 slice + 2 astype/weight-scale glue ≈ **6-8 eager boundaries**
- the `dsa_indexer_scores` steel GEMM launch itself (replaced by a different kernel, not deleted)
- **total: ~8-10 boundaries/layer × 11 layers ≈ 90-110 boundaries/token**, not 792.

The residual ~680 of the 792 are: candidate-mask/tail-window construction (~300-400),
the gathered-attention path (D3's target, ~600 by E-046's own D3 estimate — overlapping
accounting), and MoE affine glue. **D2-as-directed (swap the scores kernel) cannot touch
those.** Wiring the v32 `s==1` branch wholesale is not available: the vendored tree's
indexer produces *pooled, masked, tail-augmented* top-k indices that the v32 branch
never computes (v32 gates on `k.shape[2] >= 4096` raw cache, no pooling — deepseek_v32.py:305-324).

### 1.3 GPU-time contribution of those boundaries

Baseline: 33.4 t/s mid-ctx = **29.9 ms/token**. At 9-12 µs/boundary (E-046 §4, E-022 ~16 µs
× ~2/3 launch-vs-view correction):

- 792 boundaries (full ctx>2K delta): 7.1-9.5 ms = **24-32%** of a token — this is the
  *ceiling for D2+D3+glue work combined*, not D2.
- D2's actual ~90-110 boundaries: 0.8-1.3 ms = **2.7-4.4%** upper bound.

Assumptions stated: (a) every eager boundary costs the full 9-12 µs (real average is lower —
many are view ops fused at submission, and consecutive small ops pipeline); (b) boundaries
are fully serialized (they overlap on 40+ GPU cores); (c) the replacement kernel is free.

### 1.4 What the replacement does

`dsa_decode_scores(q[B,32,1,128], k[B,1,S,128], w[B,32]) → [B,1,1,S]` (fast.py:354-369;
csrc/dsa_indexer.cpp:1172-1221; metallib `dsa_decode_scores_{bf16,f16}_{of32,osame}_h32_d128_t256`).
One thread per key position: `score = Σ_h w_h · relu(q_h·k)`, fp32 accumulate, K read by
raw strides (capacity-backed cache slices consumed **in place, no copy**; row stride must be
a multiple of 8 elements). One threadgroup of 256 threads per 256 keys; grid (⌈S/256⌉, 1, B).

Geometry constraints (csrc/dsa_indexer.cpp:746-786 `unsupported()`):
- q exactly [B,32,1,128]; k [B,1,S,128], S ≥ 1024; w [B,32]; bf16/fp16 only
- q and w row-contiguous; **k rows 16B-aligned** (stride%8==0) — satisfied by pooled cache rows
- mask is not an input: **no candidate masking** — scores come back for ALL keys

Input mismatch vs current path: the vendored indexer feeds `dsa_indexer_scores` a
**transposed+padded** qt/keys and relies on downstream `valid_candidates` masking to
enforce causality/pool-boundary validity. `dsa_decode_scores` needs unpadded [B,32,1,128]
q (trivial) but produces unmasked scores — the downstream mask/where/topk chain must stay
regardless (it is the bulk of the boundaries D2 was credited with).

### 1.5 Savings

- **Theoretical max** (replacement free): 0.8-1.3 ms/token ≈ **3-4%**.
- **Realistic**: the native kernel is not free — it streams S×128×2B (at S=4K: 1 MB ≈ 5 µs
  at ~200 GB/s effective, but it occupies ⌈4096/256⌉=16 threadgroups = under half a wave,
  so latency-bound ≈ 5-15 µs/layer). It *replaces* a steel GEMM launch plus ~6-8 glue
  launches plus 2 pad allocations. Net ≈ 40-80 µs/layer × 11 = 0.4-0.9 ms → **~1.5-3%**
  expected end-to-end decode delta. Under 26 t/s slow-mode conditions the same absolute
  saving is a slightly larger relative win (~2-3.5%) but nothing more.

### 1.6 Complexity / risk

- **Files to touch:** `language.py` `Glm5NextIndexer._native_scores` (add an s==1 branch
  calling `fast.dsa_decode_scores` with the existing post-hoc mask chain kept intact) —
  **modifies an existing vendored file**, breaking the env-gated-new-file house style
  (prior items shipped as new patch modules + env gates; this one has no clean seam —
  `_native_scores` is a method on the vendored class, so the choice is in-place edit vs a
  monkey-patch shim module, the latter being fragile per E-048's dual-instance lesson).
- Could be env-gated (`OMLX_GLM_DSA_DECODE_SCORES=1`) with fallback to the current
  `dsa_indexer_scores` call, giving a clean rollback: unset env = byte-identical path.
- **RC-1 silent-fallback risk: HIGH.** The natural failure is an exception inside the new
  branch (or a geometry mismatch — e.g. pooled-cache k rows not stride-8-aligned after a
  cache regrow) caught by the existing `except` in `_native_scores` (language.py:433-442),
  which **logs once and silently reverts to stock**. E-048 proved the "armed" log line can
  be invisible (import-order) and shims can miss dual module instances.
- **Engagement verification (mandatory):** (1) log the first native call at WARNING via a
  dedicated one-shot flag *after* logging config is up — and treat absence as failure;
  (2) IR census: rebuild the E-046 pair-dot census with the flag on and require
  `OMLXDSADecodeScores` nodes (66/token) to appear and `OMLXDSAIndexerScores` to vanish
  from the decode-only delta — the pair-diff method is the only attribution that survived
  E-046's prefill-pollution trap; (3) numeric A/B: greedy 64-token generation must be
  token-identical (fp32 accumulation makes the new scores *strictly tighter* than bf16
  chain — near-identical but check top-k sets at the selection boundary).

### 1.7 Verdict input

Expected end-to-end: **+1.5-3% decode at ctx>2K, 0% below 2K.** Against the spread already
observed *within* one configuration (26↔33.4 t/s = 21%, E-049/050/051), this is smaller
than measurement noise unless n≥10 A/B arms are used. The last three decode attempts closed
negative (ds4 MTP: acceptance 0.03 vs 0.52 needed; D1 fusion: NEUTRAL-as-shipped, µs-level
prize; Path A: confirmed dead). D2's honest expected value does not clear the bar that
killed them.

---

## Part 2 — D3: extend `sparse_mla_attention` geometry gate to T=1

### 2.1 Current decode path for L==1 sparse MLA (file:line)

Per DSA layer per decode token at ctx>2048 (`Glm5NextSparseAttention.__call__`,
language.py:641-677), after the indexer returns top-k:

| step | location | what runs |
|---|---|---|
| validity | language.py:649 | `valid_sel = topk_indices >= 0` |
| L==1 gather | language.py:652-659 | `clip(topk_indices, 0, Kv-1)` → broadcast → **`mx.take_along_axis`** of kv_latent [B,1,Kv,512] → **gathered [B,1,2048(+tail),512] ≈ 2 MB intermediate/layer** |
| sel_mask | language.py:660-674 | broadcast, optional left-pad mask gather (`mask` is None in single-stream decode — the 661 branch is skipped), `&` |
| attention | language.py:745 via fall-through | **`mx.fast.scaled_dot_product_attention`** over expanded q [B,64,1,256→512 latent] × k=v=latent (11 calls/token; E-046 confirms stock SDPA) |
| unembed | language.py:747-752 | `unembed_out` + transpose/reshape + `o_proj` |

Below ctx 2048 the indexer returns None and the same SDPA runs *dense* over the full cache
(no gather) — D3 does not apply there either (the `sparse_mla_attention` call site at
language.py:682-688 additionally requires `Kv >= 4096` and lives only in the L>8 branch).

### 2.2 Op/boundary count D3 would replace

E-046's D3 row: "gather(2048×512) + sel_mask + SDPA ≈ 60 ops × 11 layers → ~600 boundaries."
The IR diff bears this out: within the 11,781-node decode-only delta, Gather/Slice/
Broadcast/Contiguous/ExpandDims/AsType + mask arithmetic attributable to the L==1 branch
plus the SDPA replacement accounts for roughly 550-650 nodes/11-layer token. **D3's ~600
figure is credible and is the dominant share of the 792.**

### 2.3 GPU-time contribution

600 × 9-12 µs = 5.4-7.2 ms = **18-24% of the 29.9 ms token** (same assumptions as §1.3 —
fully serialized, every boundary at full cost; real fraction lower due to view-op fusion
and overlap, so treat 18-24% as the ceiling, not the estimate).

### 2.4 What the replacement does

`glm_dsa_sparse_mla_attention(q_latent[B,H,L,512], q_pe[B,H,L,64], kv_latent[B,1,K,512],
k_pe[B,1,K,64], topk_indices[B,1,L,TOPK]) → [B,H,L,512]` — one steel kernel doing the
sparse attention directly over **selected** rows (indices read inside the kernel; the 2 MB
gathered intermediate never materializes).

Two gates block it at decode:
1. **Python geometry gate** — `omlx/patches/glm_moe_dsa/sparse_mla.py:431`: `L <= 1` →
   return None (also `topk_rows == L` required, :437 — satisfiable at decode: the vendored
   indexer emits exactly one row). This is the D3 directive target.
2. **C++ primitive gate** — `csrc/sparse_mla.cpp:121`: `q_latent.shape(2) <= 1` →
   `unsupported()` → throws → fast.py:431-444 catches?? **No: fast.py's `_ext` call path
   does NOT catch — the throw propagates** (fast.py:415-443 has no try/except; only the
   `mx.fast` fallback path exists). **Removing only the Python gate produces a guaranteed
   crash, not a silent fallback — the C++ gate must be relaxed too, which means a C++
   rebuild, contradicting the "transplanted binaries already contain the symbols" premise.**

The metallib symbols (`steel_sparse_mla_{bf16,f16}_bk{128,256}_dc32_h32_d512_pe64_wm4`,
verified in traces/metallib_symbols.txt) take `qL` as a **runtime param**
(sparse_mla.cpp:263 — `/* int qL = */ q_latent.shape(2)` in params struct; grid is
`MTL::Size(qL, B, 1)`, :320) — the *machine code* plausibly handles qL=1 (the kernel was
built for arbitrary qL). But reaching it requires either editing
`GlmDsaSparseMlaAttentionPrimitive::unsupported` and rebuilding `_ext`/dylib (M effort,
new binaries), or bypassing the primitive wrapper entirely and invoking the metallib
function through a hand-rolled `mx.fast` custom path (the metal_kernel route, whose
launch floor E-048 measured at 10-15 µs — eroding the win).

Also blocking correctness: the kernel requires **q_pe/k_pe (64-wide RoPE keys)** and
`do_causal=True` (sparse_mla.cpp:83: `!do_causal` → unsupported). GLM-5.3-Flash is
**NoPE** — the vendored path passes `q_pe = zeros`, `k_pe = zeros` and non-causal masked
selection (language.py:679-681). Zeros pass dtype/shape checks, and causal-vs-masked
differs only on masked-out rows — but `topk_length`/`causal` plumbing must be exactly
right or selection silently changes.

### 2.5 Savings

- **Theoretical max** (kernel free): 5.4-7.2 ms = **18-24%**.
- **Realistic**: the kernel at qL=1 launches **one threadgroup per (qL,B) = 1 threadgroup
  per layer-call** (grid = (qL,B,1)). One threadgroup = ~1 GPU core ≈ 1/40 of an M3
  Ultra's bandwidth. It must stream 2048 selected rows × 512 × 2B ≈ 2 MB scattered:
  ~100-200 µs/layer, ×11 = **1.1-2.2 ms of NEW kernel time**, against removing ~5.4-7 ms
  of eager overhead → net ~3-6 ms/token ≈ **+10-20%** *if* the eager boundaries really
  cost full price. Haircutting for view-op fusion/overlap (the recurring reason every
  census-derived estimate in this mission has landed high — D1 projected 3-5.5× and
  measured neutral): a defensible central estimate is **+5-10% end-to-end decode at
  ctx>2K**, with genuine downside risk that the one-threadgroup kernel is *slower* than
  the eager path's parallel SDPA (which spreads 64 heads × keys across the whole GPU).
  The BK=128 h32 instantiation exists but the model uses H=64 → bk256/wm8, the variant
  E-046-era benchmarks showed most sensitive to occupancy.

### 2.6 Complexity / risk

- **Files to touch:** sparse_mla.py:431 (Python gate — trivial), sparse_mla.cpp:121
  (C++ gate — **rebuild of `_ext` + metallib required**), plus a new vendored call site at
  language.py ~652-677 (the L==1 branch must call `sparse_mla_attention` — today it never
  reaches it). Three files in two languages; violates env-gated-new-file house style on
  the C++ side no matter how it's sliced. Rollback: the Python call site can be env-gated,
  but once the C++ gate is relaxed, rollback = binary swap back (the tree has git; the
  binaries are build artifacts — acceptable but heavier than an env flag).
- **RC-1 silent-fallback risk: LOW-MEDIUM, unusually.** The dominant failure mode is loud
  (exception propagates through fast.py's un-guarded `_ext` path → request 500s), not
  silent. The *silent* risks are numeric: (a) do_causal semantics on NoPE zeros;
  (b) bf16 summation-order differences (accepted for prefill, but decode runs 100× the
  layer-invocations per output token — drift compounds across 45 layers × thousands of
  tokens); (c) the tail-window/−1 sentinel handling: the kernel's `topk_length` /
  valid-prefix contract must match the vendored indexer's −1 padding exactly, and a
  mismatch here **silently attends to wrong keys** (quality regression, no error).
- **Engagement verification:** IR pair-diff again (`GlmDsaSparseMlaAttention` primitive
  nodes appearing in the decode-only delta — currently 0), plus greedy-output A/B vs the
  stock path (any token divergence = selection-contract bug, stop), plus metallib-usage
  counter if available (`dsa_decode`-style instrumentation does not exist in-tree —
  E-046 §3 confirmed no ATTN-PATH-style trace env exists).

### 2.7 Verdict input

Expected end-to-end: **+5-10% central, wide error bars (+0% to +20%), zero benefit below
ctx 2K, real risk of regression from the 1-threadgroup kernel.** This is the largest
honest number on the D-list — and it requires a C++ rebuild (new binaries) to even test,
contradicting the transplant-only premise, plus a selection-contract port (−1 sentinels,
tail windows, left-padding) where the failure mode is silent wrong-attention.

---

## Part 3 — Comparative verdict

| | D2 (decode-scores fuse) | D3 (sparse-MLA T=1) |
|---|---|---|
| Boundaries actually replaceable | ~90-110 of the credited 792 | ~550-650 (its ~600 estimate holds) |
| Ceiling | 3-4% | 18-24% |
| Realistic | **+1.5-3%** | **+5-10% (range 0-20%)** |
| Work | 1 vendored file, env-gated shim possible | 3 files incl. C++ rebuild of `_ext`/metallib |
| Fallback risk (RC-1) | HIGH (except-guard swallows; E-048 proved invisibility) | LOW-MEDIUM loud crash, but silent wrong-key attention on contract bugs |
| Verification cost | moderate (IR census) | high (IR census + numeric A/B + contract audit) |

**NEITHER, at current evidence — with D3 the strictly better candidate if one must be
chosen.** The honest arithmetic:

1. **D2 is mislabeled in the ledger.** The 792-boundary credit belongs to the whole
   ctx>2K delta; the directed change (swap one score kernel) touches ~100 boundaries
   worth ≤1.3 ms ceiling, ~0.5-0.9 ms realistic — **+1.5-3%**, inside the 21% fast/slow
   bimodality noise band. It cannot justify implementation ahead of settling H-decode-variance,
   which offers ~21% with zero kernel work and is parked, not solved.
2. **D3 has a real prize (the only D-item that does) but its cost premise is stale:**
   it is not a Python-gate one-liner; the C++ `unsupported()` gate at sparse_mla.cpp:121
   forces a native rebuild, and the qL=1 launch shape (1 threadgroup/layer) may give back
   much of the win. Best case +20%, central +5-10%, worst case negative.
3. **Sequencing argument:** the fast/slow flip (E-049/050/051, ~21% on exactly the
   ctx>2K graph both D2/D3 target) bounds and confounds any D-item A/B — until it is
   resolved (or pinned by pinning server state), a +3% D2 result is unmeasurable and even
   D3's +5-10% needs n≥10 paired arms. The evidence package for the parent: **the next
   decode session should spend its budget on the flip (cheap instrumentation was scoped at
   E-051), and D3 only if the flip closes negative AND a C++ rebuild is acceptable. D2
   should be recorded as correctly-estimated-at-≤3% and shelved.**

If forced to pick one first candidate: **D3** — bigger honest prize, louder failure mode,
and its two blockers (C++ gate, selection contract) are enumerable and testable offline
without the server. Expected delta for D3: **+5-10% mid/long-ctx decode (central),
0-20% range.** Biggest single risk: **the C++ geometry gate means "kernel replacement"
actually means "native rebuild" — and if the selection contract (−1 sentinels / tail
windows / causal-on-NoPE-zeros) is subtly wrong, the model silently attends to the wrong
keys with no error and no crash** (RC-1's worst shape: wrong answers at full speed).

---

## Files

- `<studio-artifacts>` — this package
- `<studio-artifacts>` — raw traces: vendored source excerpts (native_scores/L1
  branch/bypass/SDPA tail), v32 s==1 gate, dsa_indexer.cpp unsupported()+wrapper,
  sparse_mla.cpp gates+evalgpu, dsa_indexer.metal header, sparse_mla.metal instantiations,
  metallib symbol strings, `labeldiff.py` + `labeldiff_out.txt` (fresh 11,781-node IR
  delta decomposition), `collect_traces.sh`
- local `/tmp/e054/labeldiff.py` (agent-side copy)
