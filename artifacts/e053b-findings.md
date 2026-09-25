# E-053b — Acceptance Probe Redo with TRUE h_prev

Date: 2026-09-25
Target gens: /tmp/e053/server_gens.json (GLM-m48 via omlx :8008, greedy, 64 tokens — reused, not re-collected)
Drafter: <studio-path>, canonical Glm5NextMTPBlock
Fix vs E-053: h_prev is now the **real 45-layer trunk pre-final-norm hidden state** (E-053 used h_prev=0 after the full-trunk load crashed Metal).

## VERDICT: CLOSE-PATH-A-CONFIRMED (pos-2 agreement = 0.031 < 0.40)

```
 id cat            pos1  pos1-2    1-8   1-16   1-32   1-64
  1 chit-chat      0.00    0.00   0.00   0.00   0.00   0.00
  2 chit-chat      0.00    0.00   0.00   0.00   0.00   0.00
  3 technical      0.00    0.00   0.00   0.00   0.00   0.00
  4 technical      0.00    0.00   0.00   0.00   0.00   0.00
  5 code           0.00    0.00   0.00   0.00   0.00   0.00
  6 code           0.00    0.00   0.00   0.00   0.00   0.00
  7 summarization  0.00    0.00   0.00   0.00   0.00   0.00
  8 summarization  0.00    0.00   0.00   0.00   0.00   0.00
  9 creative       0.00    0.00   0.00   0.00   0.00   0.00
 10 creative       0.00    0.00   0.00   0.00   0.00   0.00
 11 chit-chat      0.00    0.50   0.38   0.19   0.09   0.05
 12 technical      0.00    0.00   0.00   0.00   0.00   0.00
 13 code           0.00    0.00   0.00   0.00   0.00   0.00
 14 summarization  0.00    0.00   0.00   0.00   0.00   0.00
 15 creative       0.00    0.00   0.00   0.00   0.00   0.00
 16 technical      0.00    0.00   0.00   0.00   0.00   0.00
ALL                0.00    0.03   0.02   0.01   0.01   0.00
```

## Trunk load: SUCCEEDED (E-053's blocker resolved)

The E-053 crash was eager full-materialization. Working recipe (mirrors
VLMBatchedEngine exactly; /tmp/e053b/probe.py):

1. `apply_mlx_vlm_glm5_next_compat_patch()` (registers vendored glm5_next +
   installs omlx PoolingCache into mlx_lm.models.cache).
2. `maybe_apply_pre_load_patches(model_dir, model_settings=None, for_vlm=True)`
   (glm5_next MTP runtime patch: Glm5NextMTPBlock, patched Glm5NextModel with
   `return_raw_hidden`).
3. Strip the `format: mlx` safetensors shard metadata around the load
   (omlx `_force_qwen4_exp_sanitize_on_load` equivalent — forces the
   pre-quantize `Model.sanitize`; per-layer 8-bit overrides in
   quantization_config then bind correctly).
4. `mlx_vlm.utils.load_model(path, lazy=True, strict=True)`.
5. Materialize `language_model` ONLY, in 10-layer chunks (`mx.eval` per
   chunk); embed/norm/lm_head after; leave vision tower untouched.
6. Attach the drafter head AFTER the trunk load: the trunk checkpoint has no
   `mtp.*` keys, and a strict load with an attached head fails with
   "Missing 29 parameters" (first run died exactly this way; the server never
   attaches — its mtp_enabled=False for SpecPrefill drafters).

Memory (mx.metal, machine 256GB, wired limit 0):
- after lazy build: active 0.00GB
- drafter head load: 13.84GB
- layers 0-9 / 10-19 / 20-29 / 30-39 / 40-44: 42.4 / 82.1 / 121.7 / 161.3 / 181.1GB
- after embed/norm/lm_head: **182.38GB active** (final load)
- peak during generation: **186.72GB** — no crash, ~36GB headroom vs the
  222.7GB Metal cap. Server was stopped during measurement (HTTP 200 recorded
  before pkill; pgrep 0 after).

## What was measured

Per prompt: server-identical chat template + tokenizer (prompt token counts
match server 16/16) -> trunk prefill with [KVCache, PoolingCache] per layer ->
h_prev = raw pre-final-norm hidden at the last position (`lm.model(...,
return_raw_hidden=True)`, exactly what hidden_sink captures server-side) ->
canonical block autoregressive 64 greedy steps per the mtp_forward contract:

    e  = enorm(embed(next)); x = eh_proj([e, hnorm(h_prev)])
    h' = x + attn(ln(x));    h''= h' + MoE(pln(h'))
    logits = lm_head(norm(h''));  h_prev := h''

All 29 drafter tensors bound: enorm/hnorm/eh_proj direct, shared_head_norm ->
block norm, remaining 25 under `block.*`.

## Diagnosis: measurement is now clean; drafter is simply not aligned

- Unlike E-053 (pure garbage: repeated subwords, CJK fragments), the drafter
  now produces *plausible English continuations* — e.g. prompt 1 starts
  "The user is asking about AI applications in healthcare and wants...".
  The trunk hidden demonstrably reaches the head and improves it.
- But it is not the *target's* continuation: pos-1 match 0/16, overall
  1-64 agreement 0.7% (prompt 11 contributes all of it: 5% there).
- The offline MTP block is greedy-single-step with its own fresh KV/pooling
  caches; the server's drafter operates inside the target's verify loop
  (shared indexer pools, chain rollback). This offline probe measures the
  block in isolation — and in isolation its greedy path does not track the
  server's greedy path. CLOSE-PATH-A per the E-052 rule stands, now with a
  valid measurement.
- Residual caveat (smaller than E-053's, but real): the server's own
  inference of the drafter could still differ (shared caches, verify-mode
  quantization paths). An in-server instrumentation run would settle that
  residual; per the E-052/E-053b rule the offline measurement is decisive.

## Files

- /tmp/e053b/probe.py — load + generation (new files only; nothing in /tmp/e053 modified)
- /tmp/e053b/probe_run.log — memory trail + per-prompt logs
- /tmp/e053b/draft_gens_trueh.json — 16 x 64 greedy token ids + text
- /tmp/e053b/score.py — E-053-identical scoring
- /tmp/e053b/ACCEPTANCE.json — full results + verdict
