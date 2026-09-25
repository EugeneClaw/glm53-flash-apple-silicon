# SPDX-License-Identifier: Apache-2.0
"""E-026: Standalone GLM-5.3-Flash MTP draft scorer for SpecPrefill.

Builds a one-layer draft LM from an ALREADY-LOADED target model plus the
avlp12 GLM-5.3-Flash-Alis-MTP-Drafter block checkpoint. Bypasses mlx_lm's
glm5_next_mtp rejection (no supported loader for that model_type).

Proven offline in E-025 (forward correct, score_tokens 1,024 tokens ->
512 selected in 135.7 ms, keep_pct 0.4).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _load_linear_forward():
    # Vendored glm5_next package cannot be imported standalone (its __init__
    # chain requires mlx_vlm.models.base); load the module directly from file.
    import importlib.util as ilu
    import os

    vendored = os.environ.get(
        "OMLX_GLM5_VENDORED_DIR",
        "<omlx-tree>/omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next",
    )
    path = os.path.join(vendored, "linear.py")
    spec = ilu.spec_from_file_location("glm5_next_linear", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.linear_forward


def glm_sparse_extract_queries(attn, x, cache=None, **kwargs):
    """GLM sparse-MLA query projection for SpecPrefill importance scoring.

    Replicates the attention's own query path: q_a_proj -> q_a_layernorm ->
    q_b_proj -> reshape to heads -> embed_q into the shared 512-d latent
    space (the same space as the KV latents, which is what the scoring
    QK^T needs).
    """
    linear_forward = _load_linear_forward()
    B, L, D = x.shape
    qr = attn.q_a_layernorm(linear_forward(attn.q_a_proj, x))
    q = linear_forward(attn.q_b_proj, qr)
    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    return attn.embed_q(q)  # MultiLinear: [B, H, L, 512]


def _target_text_config(target_model):
    """Extract TextConfig + language model handle from a loaded VLM."""
    lm = target_model.language_model
    cfg = lm.args if hasattr(lm, "args") else lm.config
    return cfg, lm


def build_glm_mtp_draft_model(target_model, drafter_path):
    """Build the standalone DraftLM (E-025 adapter, server-integrated form).

    Returns an nn.Module-compatible scorer with .layers (sparse-MLA layer),
    .embed_tokens, .lm_head, make_cache(), and __call__(inputs, cache=...)
    -> [B, T, vocab] logits.
    """
    import mlx.nn as nn
    import sys

    sys.path.insert(0, "<omlx-tree>")
    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()

    from mlx_vlm.models.glm5_next.language import (
        TextConfig,
        Glm5NextDecoderLayer,
    )

    cfg, lm = _target_text_config(target_model)
    if isinstance(cfg, TextConfig):
        tcfg = cfg
    else:
        raw = json.load(open(_model_dir_of(lm, target_model) + "/config.json"))
        tcfg = TextConfig.from_dict(raw.get("text_config", raw))

    # Lift embed + head modules (stay quantized; correct oQ format via loader)
    emb_module = lm.model.embed_tokens
    lm_module = getattr(lm, "lm_head", None)
    if lm_module is None:
        lm_module = getattr(lm.model, "lm_head", None)
    if lm_module is None and hasattr(emb_module, "as_linear"):
        lm_module = emb_module.as_linear

    dt = mx.load(f"{drafter_path}/model.safetensors")

    class DraftLM(nn.Module):
        def __init__(self, tcfg, emb, lm):
            super().__init__()
            self.config = tcfg
            self.embed_tokens = emb
            import copy as _copy

            _cfg = _copy.copy(tcfg)
            _lt = list(_cfg.layer_types)
            _lt[44] = "deepseek_sparse_attention"
            _cfg.layer_types = _lt
            self.layers = [Glm5NextDecoderLayer(_cfg, layer_idx=44)]
            self.shared_head_norm = nn.RMSNorm(tcfg.hidden_size, eps=tcfg.rms_norm_eps)
            self.shared_head_norm.weight = dt["mtp.shared_head_norm.weight"]
            self.lm_head = lm

        def __call__(self, inputs, cache=None):
            from mlx_vlm.models.glm5_next.language import (
                create_attention_mask,
                create_ssm_mask,
            )

            h = self.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(self.layers)
            layer = self.layers[0]
            if layer.is_linear:
                mask = create_ssm_mask(h, cache[0])
            else:
                mask = create_attention_mask(
                    h, cache[0][0] if cache[0] else None, return_array=True
                )
            hc_mult = self.config.hc_mult
            h = mx.broadcast_to(
                h[:, :, None, :], (h.shape[0], h.shape[1], hc_mult, h.shape[2])
            )
            h = mx.contiguous(h)
            h = layer(h, mask=mask, cache=cache[0])
            h = h.mean(axis=2)
            h = self.shared_head_norm(h)
            if self.lm_head is not None:
                return self.lm_head(h)
            return h

        def make_cache(self):
            layer = self.layers[0]
            from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, PoolingCache

            if layer.is_linear:
                return [ArraysCache(size=2)]
            return [
                CacheList(
                    KVCache(),
                    PoolingCache(layer.self_attn.indexer.index_kpool),
                )
            ]

    draft = DraftLM(tcfg, emb_module, lm_module)

    rename = {
        "mtp.self_attn.": "self_attn.",
        "mtp.mlp.": "mlp.",
        "mtp.input_layernorm.weight": "input_layernorm.weight",
        "mtp.post_attention_layernorm.weight": "post_attention_layernorm.weight",
    }
    flat = {}
    for k, v in dt.items():
        if k in (
            "mtp.eh_proj.weight",
            "mtp.enorm.weight",
            "mtp.hnorm.weight",
            "mtp.shared_head_norm.weight",
        ):
            continue
        short = None
        for old, new in rename.items():
            if k.startswith(old) or k == old:
                short = k.replace(old, new)
                break
        if short is not None:
            flat[short] = v
    draft.layers[0].load_weights(list(flat.items()), strict=False)
    logger.info(
        "GLM MTP draft model built from target (layers=1, sparse-MLA, E-025 adapter)"
    )
    return draft


def _model_dir_of(lm, target_model):
    """Best-effort recovery of the target model directory for config reload."""
    # oMLX engines keep the source path on the wrapper; try common attrs.
    for attr in ("_model_dir", "model_dir", "source_path", "path"):
        v = getattr(lm, attr, None) or getattr(target_model, attr, None)
        if isinstance(v, str):
            return v
    raise RuntimeError("cannot locate target model directory for TextConfig reload")


def install_glm_query_extractor():
    """Register the GLM sparse-MLA extractor in specprefill's auto-detect chain.

    _detect_query_extractor (patches/specprefill.py:303) picks
    _nemotron_h_extract_queries for NoPE attention (no .rope attr), which
    expects q_proj and fails on GLM's MLA module. This installs a GLM-aware
    detection so the scheduler's extractor-less score_tokens() calls work.
    """
    import omlx.patches.specprefill as sp

    orig = sp._detect_query_extractor

    def _detect(attn_obj):
        if hasattr(attn_obj, "q_a_proj") and hasattr(attn_obj, "embed_q"):
            return glm_sparse_extract_queries
        return orig(attn_obj)

    sp._detect_query_extractor = _detect
    # score_tokens resolves the extractor via module attribute at call time
    # (line 503: `if query_extractor is None: query_extractor =
    # _detect_query_extractor(attn_obj)`) — so monkeypatching the module
    # attribute is sufficient.
    logger.info("GLM query extractor installed in specprefill detect chain")


# Installed at import of the server (engine loads this module only on
# glm5_next_mtp draft paths, so the patch is inert for every other model).
install_glm_query_extractor()