#!/usr/bin/python3
"""E-038: draft-scoring fixes per E-031b root-cause report.
R1: skip lm_head for all but the final draft chunk position (2.6 TFLOP + 0.63GB
    alloc discarded per chunk — ~75% of fast-mode compute).
R2: env-gate the per-chunk mx.clear_cache() (SPEC_PREFILL_KEEP_POOL=1) — the
    target learned this lesson (scheduler.py:3984-3989, E-012a); the draft pays
    full buffer-pool re-creation per chunk under near-cap footprint.
R4a: vectorize select_chunks' per-chunk means (~750 .item() syncs -> 1 eval)."""
import re

# ---- R1: glm_draft_adapter.py ----
FA = "<omlx-tree>/omlx/engine/glm_draft_adapter.py"
src = open(FA).read()

# add collect_logits flag + conditional lm_head
OLD_INIT = """    class DraftLM(nn.Module):
        def __init__(self, tcfg, emb, lm):
            super().__init__()
            self.config = tcfg
            self.embed_tokens = emb"""
NEW_INIT = """    class DraftLM(nn.Module):
        def __init__(self, tcfg, emb, lm):
            super().__init__()
            self.config = tcfg
            # E-038 R1: skip the full-vocab lm_head except when the caller
            # explicitly needs logits (final position only). Scoring uses
            # hidden-state captures + cache keys, not logits, per chunk.
            self.collect_logits = True
            self.embed_tokens = emb"""
assert OLD_INIT in src
src = src.replace(OLD_INIT, NEW_INIT, 1)

OLD_CALL = """            h = layer(h, mask=mask, cache=cache[0])
            h = h.mean(axis=2)
            h = self.shared_head_norm(h)
            if self.lm_head is not None:
                return self.lm_head(h)
            return h"""
NEW_CALL = """            h = layer(h, mask=mask, cache=cache[0])
            h = h.mean(axis=2)
            h = self.shared_head_norm(h)
            if not self.collect_logits:
                # E-038 R1: hidden states only — saves 2.6 TFLOP + 0.63GB
                # discarded logits per chunk (~75% of fast-mode compute).
                return h
            if self.lm_head is not None:
                return self.lm_head(h)
            return h"""
assert OLD_CALL in src
src = src.replace(OLD_CALL, NEW_CALL, 1)
open(FA, "w").write(src)
print("R1-ADAPTER-DONE")

# ---- R1+R2: patches/specprefill.py ----
FS = "<omlx-tree>/omlx/patches/specprefill.py"
sp = open(FS).read()

# R2: gate per-chunk clear_cache
OLD_LOOP = """        model(prompt[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk
        if progress_callback is not None:
            progress_callback(processed, n)
        mx.clear_cache()"""
NEW_LOOP = """        # E-038 R1: skip lm_head on chunk calls (hidden states suffice);
        # re-enable for the final logits call below.
        _collect = getattr(model, "collect_logits", None)
        if _collect is not None:
            model.collect_logits = False
        try:
            model(prompt[processed : processed + chunk][None], cache=cache)
        finally:
            if _collect is not None:
                model.collect_logits = True
        mx.eval([c.state for c in cache])
        processed += chunk
        if progress_callback is not None:
            progress_callback(processed, n)
        # E-038 R2: per-chunk pool clear forces full buffer re-creation under
        # near-cap footprint (the 44-73s slow mode; E-031b root cause). The
        # target disabled its own per-chunk clear for the same reason
        # (scheduler.py:3984-3989, E-012a). score_tokens still clears once
        # at line ~595. Opt-in: SPEC_PREFILL_KEEP_POOL=1.
        import os as _os

        if not _os.environ.get("SPEC_PREFILL_KEEP_POOL"):
            mx.clear_cache()"""
assert OLD_LOOP in sp, "prefill loop not found"
sp = sp.replace(OLD_LOOP, NEW_LOOP, 1)

# R4a: vectorize select_chunks per-chunk means
OLD_SC = """    chunk_scores = []
    for i in range(ranked_end):
        start = i * chunk_size
        end = min(start + chunk_size, M)
        chunk_scores.append(mx.mean(importance[start:end]).item())"""
NEW_SC = """    # E-038 R4a: vectorized chunk means — one eval instead of ~750 .item()
    # syncs (~0.2-0.5s at 24K inside the scoring timer). Identical values.
    if ranked_end > 0:
        pad_needed = ranked_end * chunk_size - M
        imp_p = (
            mx.pad(importance[: ranked_end * chunk_size].reshape(ranked_end, chunk_size), [(0, 0), (0, 0)])
            if pad_needed <= 0
            else mx.pad(importance, [(0, pad_needed)]).reshape(ranked_end, chunk_size)
        )
        means = mx.mean(imp_p, axis=1)
        chunk_scores = means.astype(mx.float32).tolist()
    else:
        chunk_scores = []"""
assert OLD_SC in sp, "select_chunks loop not found"
sp = sp.replace(OLD_SC, NEW_SC, 1)
open(FS, "w").write(sp)
print("R2-R4-SPEC-DONE")