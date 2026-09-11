#!/usr/bin/env python
"""Load the model and confirm the Gemma-3 text tower.  BLOCKING GATE 2.

`google/gemma-3-4b-pt` is a `Gemma3ForConditionalGeneration` (multimodal wrapper) and
TransformerLens does a **text-only extraction**.  That extraction is exactly where a silent
mismatch hides, and a silent mismatch produces a wrong published result rather than a crash.

Six shape assertions against `hf_cfg.text_config` — NOT `hf_cfg`, which on the multimodal
wrapper carries the vision tower's numbers — plus four Gemma-3-specific structural checks
that a generic loader gets wrong:

  S1  interleaved local-sliding / global attention, element-wise against `layer_types`
  S2  two RoPE bases: `rope_local_base_freq` (local) vs `rope_theta` (global), per layer
  S3  sqrt(d_model) embedding scale applied exactly ONCE, with a tied unembedding
  S4  QK-norm / query_pre_attn_scalar / softcapping honored as the config declares

S1-S3 are checked EMPIRICALLY (from the built modules and their buffers) wherever possible
rather than by attribute name, because the TL-side attribute names are unverified on 3.7.0
and a name-based check that silently finds nothing is worse than no check.

Usage:
    python scripts/interp/01_load_and_verify.py --model google/gemma-3-1b-it   # de-risk
    python scripts/interp/01_load_and_verify.py --model google/gemma-3-4b-pt   # target
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402


# --------------------------------------------------------------------------- reporting

class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def check(self, cid, desc, ok, got=None, want=None, note=None):
        if ok is not None:
            ok = bool(ok)          # numpy bools are not `True`
        status = "PASS" if ok is True else ("FAIL" if ok is False else "UNKNOWN")
        self.rows.append({"id": cid, "desc": desc, "status": status,
                          "got": _j(got), "want": _j(want), "note": note})
        mark = {"PASS": "  ok ", "FAIL": "FAIL", "UNKNOWN": " ?? ", "NOTE": "note"}[status]
        line = f"[{mark}] {cid:<5} {desc}"
        if status != "PASS":
            line += f"\n           got={_j(got)!r} want={_j(want)!r}"
        if note:
            line += f"\n           note: {note}"
        print(line)
        return ok

    def note(self, cid, desc, got=None, want=None, note=None):
        """A recorded observation that is not a pass/fail condition — used for design
        assumptions that turned out wrong without invalidating the model."""
        self.rows.append({"id": cid, "desc": desc, "status": "NOTE",
                          "got": _j(got), "want": _j(want), "note": note})
        print(f"[note] {cid:<5} {desc}")
        if note:
            print(f"           {note}")

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]

    @property
    def unknown(self):
        return [r for r in self.rows if r["status"] == "UNKNOWN"]


def _j(v):
    if isinstance(v, torch.Tensor):
        return v.tolist() if v.numel() <= 16 else f"<tensor {tuple(v.shape)}>"
    if isinstance(v, (list, tuple)) and len(v) > 40:
        return list(v[:20]) + ["…"] + list(v[-10:])
    return v


def first(obj, *names, default=None):
    """First attribute present among `names` — the TL 3.7.0 attribute names for Gemma-3
    are unverified, so every structural check tries the plausible spellings and reports
    UNKNOWN rather than silently passing when none is found."""
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return n, v
    if isinstance(obj, dict):
        for n in names:
            if obj.get(n) is not None:
                return n, obj[n]
    return None, default


# --------------------------------------------------------------------------- rope probe

def rope_base_from_buffers(cos: torch.Tensor, sin: torch.Tensor, d_head: int,
                           scale_factor: float = 1.0):
    """Recover theta from a rotary cos/sin table without knowing any attribute name.

    inv_freq[i] = theta^(-2i/d).  Row `pos` of the table is cos(pos_eff * inv_freq), where
    `pos_eff = pos / scale_factor` under linear RoPE scaling.  Take pos = 1 and the second
    frequency (i = 1, because inv_freq[0] == 1 for every theta):

        atan2(sin[1,1], cos[1,1]) = inv_freq[1] / scale_factor
        theta = (that * scale_factor) ** (-d/2)

    ⚠ `scale_factor` is not optional in practice: gemma-3-4b applies `rope_type: linear,
    factor: 8.0` on its GLOBAL layers only.  Ignoring it inflates the recovered base by
    8**(d/2) ~ 4e115 — which is what a first version of this check reported as a Gate-2
    failure before the scaling was accounted for.
    """
    try:
        c = cos.float()
        s = sin.float()
        if c.dim() == 3:
            c, s = c[0], s[0]
        if c.shape[0] < 2:
            return None
        ang = math.atan2(float(s[1, 1]), float(c[1, 1]))
        if ang <= 0:
            return None
        return (ang * scale_factor) ** (-d_head / 2.0)
    except Exception:
        return None


def rope_table_error(cos, sin, d_head: int, base: float, scale_factor: float,
                     n_pos: int = 32):
    """Max abs error between a layer's rotary table and the table the config implies.

    Stronger than inverting for the base: it checks the base AND the position scaling AND
    the frequency layout, over many (position, frequency) cells at once.
    """
    try:
        c = cos.float().cpu()
        s = sin.float().cpu()
        if c.dim() == 3:
            c, s = c[0], s[0]
        n_pos = min(n_pos, c.shape[0])
        half = d_head // 2
        inv = base ** (-torch.arange(0, half, dtype=torch.float64) * 2.0 / d_head)
        pos = torch.arange(n_pos, dtype=torch.float64) / float(scale_factor)
        ang = pos[:, None] * inv[None, :]
        return max(float((c[:n_pos, :half].double() - ang.cos()).abs().max()),
                   float((s[:n_pos, :half].double() - ang.sin()).abs().max()))
    except Exception:
        return None


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--revision", default=None, help="default: locally cached SHA")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/interp")
    ap.add_argument("--dump", action="store_true",
                    help="dump cfg/attn __dict__ keys")
    args = ap.parse_args()

    rev = args.revision or B.cached_revision(args.model)
    tag = args.model.split("/")[-1]
    rep = Report()
    t0 = time.time()

    print("=" * 78)
    print(f"Text-tower confirmation (GATE 2)   {args.model}")
    print(f"revision: {rev}   dtype: {args.dtype}   device: {args.device}")
    print("=" * 78)

    # ---------------------------------------------------------------- HF config
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(args.model, revision=rev)
    txt = getattr(hf_cfg, "text_config", None)
    nested = txt is not None
    if txt is None:
        txt = hf_cfg
    print(f"\nHF architecture: {hf_cfg.architectures}   "
          f"text_config nested: {nested}   text model_type: {txt.model_type}")

    # ---------------------------------------------------------------- load TL
    print("\nloading TransformerLens model (no weight processing) …")
    tl = B.get_backend("tl", args.model, revision=rev, dtype=args.dtype,
                       device=args.device,
                       tokenizer_id=args.model)
    tl.load()
    cfg = tl.model.cfg
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f}s")

    if args.dump:
        print("\n--- model.cfg.__dict__ ---")
        for k, v in sorted(vars(cfg).items()):
            print(f"  {k} = {_j(v)!r}")
        print("\n--- model.blocks[0].attn.__dict__ keys ---")
        a0 = tl.model.blocks[0].attn
        print(f"  class: {type(a0).__name__}")
        for k, v in sorted(vars(a0).items()):
            if k.startswith("_"):
                continue
            print(f"  {k} = {_j(v)!r}")
        print(f"  buffers: {[n for n, _ in a0.named_buffers()]}")
        print(f"  top-level model attrs: "
              f"{[n for n in ('embed','unembed','ln_final','blocks','pos_embed') if hasattr(tl.model,n)]}")

    # ================================================================ the six
    print("\n## six shape assertions against hf_cfg.text_config (NOT hf_cfg)")
    rep.check("A1", "n_layers == text_config.num_hidden_layers",
              cfg.n_layers == txt.num_hidden_layers, cfg.n_layers, txt.num_hidden_layers)
    rep.check("A2", "d_model == text_config.hidden_size",
              cfg.d_model == txt.hidden_size, cfg.d_model, txt.hidden_size)
    rep.check("A3", "n_heads == text_config.num_attention_heads",
              cfg.n_heads == txt.num_attention_heads, cfg.n_heads, txt.num_attention_heads)
    rep.check("A4", "d_head == text_config.head_dim",
              cfg.d_head == txt.head_dim, cfg.d_head, txt.head_dim)
    kvname, kv = first(cfg, "n_key_value_heads", "n_kv_heads")
    rep.check("A5", f"n_kv_heads == text_config.num_key_value_heads (TL attr: {kvname})",
              kv == txt.num_key_value_heads if kv is not None else None,
              kv, txt.num_key_value_heads,
              None if kv is not None else "no n_key_value_heads/n_kv_heads on cfg")

    # d_vocab: Gemma-3's text_config.vocab_size is the padded 262208, while the tokenizer
    # holds 262145 usable ids.  The assertion is against the CONFIG.
    rep.check("A6", "d_vocab == text_config.vocab_size",
              cfg.d_vocab == txt.vocab_size, cfg.d_vocab, txt.vocab_size,
              f"tokenizer len = {len(tl.tok)} (padded vocab is expected to exceed it)")

    # If hf_cfg is the multimodal wrapper, show that asserting against it instead would
    # have been wrong — this is the silent-mismatch the gate exists to catch.
    if nested:
        top_hidden = getattr(hf_cfg, "hidden_size", None)
        print(f"     (top-level hf_cfg.hidden_size = {top_hidden}; text_config.hidden_size "
              f"= {txt.hidden_size} — asserting against hf_cfg would be the silent bug)")

    # ================================================================ S1 interleave
    print("\n## S1 — interleaved local-sliding / global attention")
    layer_types = getattr(txt, "layer_types", None)
    if layer_types is None:
        # transformers 5.x derives layer_types from sliding_window_pattern
        pat = getattr(txt, "sliding_window_pattern", None)
        if pat:
            layer_types = ["full_attention" if (i + 1) % pat == 0 else "sliding_attention"
                           for i in range(txt.num_hidden_layers)]
    tl_name, tl_types = first(cfg, "attn_types", "layer_types", "attention_types")
    win_name, win = first(cfg, "window_size", "sliding_window", "attn_window_size")

    if layer_types is None:
        rep.check("S1", "layer_types available on text_config", None, None, None,
                  "text_config exposes neither layer_types nor sliding_window_pattern")
    elif tl_types is None:
        # No per-layer list on cfg — fall back to the built modules.
        per_layer = []
        for L in range(cfg.n_layers):
            a = tl.model.blocks[L].attn
            n, v = first(a, "attn_type", "attention_type", "is_local", "use_local_attn")
            per_layer.append(v)
        if any(v is not None for v in per_layer):
            derived = ["sliding_attention" if _is_local(v) else "full_attention"
                       for v in per_layer]
            rep.check("S1", f"per-layer attention type (blocks[L].attn.{n}) == layer_types "
                            "element-wise",
                      derived == list(layer_types), derived, list(layer_types))
        else:
            rep.check("S1", "per-layer attention type discoverable on TL model", None,
                      None, list(layer_types),
                      "neither cfg nor blocks[L].attn exposes a per-layer attention type; "
                      "checked empirically by S1b instead")
    else:
        derived = ["sliding_attention" if _is_local(v) else "full_attention"
                   for v in tl_types]
        rep.check("S1", f"cfg.{tl_name} == text_config.layer_types element-wise",
                  derived == list(layer_types), derived, list(layer_types))

    if layer_types is not None:
        n_local = sum(1 for t in layer_types if "slid" in str(t) or "local" in str(t))
        print(f"     layer_types: {n_local} sliding / {len(layer_types) - n_local} full "
              f"of {len(layer_types)}; pattern head = {list(layer_types)[:8]}")
    rep.check("S1w", f"sliding window size == text_config.sliding_window (TL attr: {win_name})",
              (win == txt.sliding_window) if win is not None else None,
              win, getattr(txt, "sliding_window", None),
              None if win is not None else "no window attr found on cfg")

    # S1b — EMPIRICAL.  Attribute agreement is not proof that the mask is built
    # differently; read each layer's own attention mask buffer and count, on its last row,
    # how many positions it may attend to.  A sliding layer must be capped at
    # `window_size`; a full layer must see everything.  Gate 3 alone would NOT catch a
    # window-size error, because the fidelity prompts are far shorter than the window.
    if layer_types is not None and win:
        T = int(win) + 64                                  # long enough to exceed the window
        ids = torch.randint(10, 100000, (1, T), device=args.device)
        seen: dict[int, int] = {}

        # Read PRE-softmax scores, not the pattern: masked keys are set to the attention
        # module's IGNORE sentinel, so `scores != IGNORE` is exact.  Counting non-zero
        # softmax weights instead gives false failures whenever an in-window weight
        # underflows to 0.0 (observed on gemma-3-1b-it layer 19: 511 vs 512).
        def mk_scores(L):
            ign = tl.model.blocks[L].attn.IGNORE

            def hook(scores, hook):  # TL calls hook(tensor, hook=self); kwarg name fixed
                row = scores[0, :, -1, :]                     # [head, key]
                seen[L] = int((row != ign).all(0).sum().item())
                return scores
            return hook

        from transformer_lens import utils as tlu
        with torch.no_grad():
            tl.model.run_with_hooks(
                ids, return_type=None,
                fwd_hooks=[(tlu.get_act_name("attn_scores", L), mk_scores(L))
                           for L in range(cfg.n_layers)])
        rows = [seen.get(L) for L in range(cfg.n_layers)]
        if all(r is None for r in rows):
            rep.check("S1b", "attention actually respects the sliding window", None,
                      None, None, "hook_pattern produced nothing")
        else:
            exp = [min(int(win), T) if _is_local(t) else T for t in layer_types]
            bad = [(L, rows[L], exp[L]) for L in range(len(exp)) if rows[L] != exp[L]]
            rep.check("S1b", f"EMPIRICAL: on a {T}-token sequence (window={win}), the "
                             "number of keys the last query attends to == window_size on "
                             "sliding layers and == T on full layers",
                      not bad, {"first8": rows[:8], "T": T, "window": int(win)},
                      {"first8": exp[:8]},
                      None if not bad else f"mismatched layers (L, got, want): {bad[:8]}")
    else:
        rep.check("S1b", "attention actually respects the sliding window", None,
                  None, None, "layer_types or window size unavailable")

    # ================================================================ S2 rope
    print("\n## S2 — two RoPE bases (local vs global)")
    want_local, want_global, rope_src = _rope_bases(txt)
    print(f"     text_config rope fields: rope_local_base_freq="
          f"{getattr(txt, 'rope_local_base_freq', None)}  "
          f"rope_theta={getattr(txt, 'rope_theta', None)}")
    print(f"     rope_scaling={getattr(txt, 'rope_scaling', None)}")
    print(f"     rope_parameters={getattr(txt, 'rope_parameters', None)}")
    print(f"     -> local={want_local} global={want_global}  (source: {rope_src})")
    if rope_src != "rope_local_base_freq/rope_theta":
        rep.note("S2src", "text_config does NOT expose rope_local_base_freq / rope_theta",
                 rope_src, "rope_local_base_freq/rope_theta",
                 "transformers 5.12.1 moved Gemma-3's two RoPE bases into a per-layer-type "
                 "dict; the flat attributes no longer exist. Values "
                 f"were read from {rope_src} instead. Not a defect.")

    cfg_rope = {n: getattr(cfg, n) for n in
                ("rotary_base", "rotary_dim", "rotary_adjacent_pairs", "use_NTK_by_parts_rope",
                 "rope_local_base_freq", "rotary_base_local", "NTK_by_parts_factor")
                if hasattr(cfg, n)}
    print(f"     TL cfg rope fields: {cfg_rope}")

    # Per-layer-type linear-scaling factor.  gemma-3-4b declares rope_type 'linear' with
    # factor 8.0 on its FULL-attention layers and plain 'default' on sliding layers; the
    # 1b declares no scaling at all.  Reading the factor per layer type is what makes this
    # check work on both.
    fac_local, fac_global = _rope_factors(txt)
    print(f"     linear-scaling factors: local={fac_local} global={fac_global}")

    derived_bases, tab_err = [], []
    for L in range(cfg.n_layers):
        bufs = dict(tl.model.blocks[L].attn.named_buffers())
        cos, sin = bufs.get("rotary_cos"), bufs.get("rotary_sin")
        is_loc = _is_local(layer_types[L]) if layer_types is not None else None
        fac = fac_local if is_loc else fac_global
        base = want_local if is_loc else want_global
        if cos is None or sin is None:
            derived_bases.append(None)
            tab_err.append(None)
            continue
        derived_bases.append(rope_base_from_buffers(cos, sin, cfg.d_head, fac))
        tab_err.append(rope_table_error(cos, sin, cfg.d_head, base, fac)
                       if base else None)

    if any(b is not None for b in derived_bases) and layer_types is not None:
        loc = [b for b, t in zip(derived_bases, layer_types) if _is_local(t) and b]
        glo = [b for b, t in zip(derived_bases, layer_types) if not _is_local(t) and b]
        got = {"local": _round(loc), "global": _round(glo)}
        if want_local is None or want_global is None:
            rep.check("S2", "per-layer RoPE base matches the config's local/global bases",
                      None, got, {"local": want_local, "global": want_global},
                      "config bases unavailable — nothing to compare the recovered bases to")
        else:
            ok = (bool(loc) and bool(glo)
                  and all(abs(b - want_local) / want_local < 0.02 for b in loc)
                  and all(abs(b - want_global) / want_global < 0.02 for b in glo))
            rep.check("S2", "per-layer RoPE base recovered from each layer's rotary table "
                            "== the local base on sliding layers and the global base on "
                            "full layers",
                      bool(ok), got, {"local": want_local, "global": want_global},
                      "bases recovered numerically from the cos/sin tables (with the "
                      "declared linear-scaling factor divided out), not by attribute name")

        errs = [e for e in tab_err if e is not None]
        worst = max(errs) if errs else None
        rep.check("S2t", "each layer's FULL rotary table reproduces the table its config "
                         "implies (base + position scaling + frequency layout), "
                         "32 positions x d_head/2 frequencies",
                  (worst is not None and worst < 1e-4), {"max_abs_err": worst},
                  {"max_abs_err": "< 1e-4"},
                  "stronger than the base recovery: catches a right base applied with the "
                  "wrong position scaling, which is exactly the gemma-3-4b hazard")

        rep.check("S2b", "the two RoPE bases are actually different per layer",
                  bool(loc and glo and abs(_round(loc)[0] - _round(glo)[0]) > 1.0),
                  got, "distinct")
    else:
        rep.check("S2", "per-layer RoPE base recoverable", None,
                  {"derived": _j(derived_bases)}, {"local": want_local,
                                                   "global": want_global},
                  "no rotary_cos/rotary_sin buffers on blocks[L].attn and no per-layer "
                  "rotary_base list on cfg — cannot verify by inspection")

    # ================================================================ S3 embed scale
    print("\n## S3 — sqrt(d_model) embedding scale applied exactly once, tied unembed")
    # Ground truth is the raw checkpoint tensor, read straight off disk — comparing TL to
    # itself cannot detect a doubled or missing scale.
    scale = math.sqrt(cfg.d_model)
    hf_emb, emb_key = hf_embed_matrix(args.model, rev)
    W_E = tl.model.W_E.detach().float().cpu()
    W_U = tl.model.W_U.detach().float().cpu()
    probe = torch.tensor([2, 1000, 2000, 5000, 200000])

    if hf_emb is None:
        rep.check("S3", "TL embedding == checkpoint embedding x sqrt(d_model)", None,
                  None, None, "could not read the embedding tensor off disk")
    else:
        hf_emb = hf_emb.float()
        V = min(W_E.shape[0], hf_emb.shape[0])
        rows_hf = hf_emb[probe]
        rows_tl = W_E[probe]
        ratio = rows_tl.norm(dim=-1) / rows_hf.norm(dim=-1).clamp_min(1e-12)
        r = float(ratio.mean())
        # 1.0 => scale missing from W_E (must then be applied in the forward pass);
        # sqrt(d_model) => folded in exactly once; d_model => doubled.
        folded = abs(r - scale) / scale < 1e-3
        unscaled = abs(r - 1.0) < 1e-3

        # Whichever it is, the *effective* embedding the residual stream receives must be
        # exactly hf_row * sqrt(d_model).  Check the actual forward path.
        with torch.no_grad():
            eff = tl.model.embed(probe.unsqueeze(0).to(args.device))[0].float().cpu()
        want_eff = rows_hf * scale
        err_eff = float((eff - want_eff).abs().max())
        rel = err_eff / float(want_eff.abs().max().clamp_min(1e-12))
        rep.check("S3", "EFFECTIVE embedding entering the residual stream == "
                        "checkpoint_row * sqrt(d_model), i.e. applied exactly once",
                  rel < 1e-4, {"max_abs_err": err_eff, "rel_err": rel,
                               "W_E/ckpt norm ratio": round(r, 4),
                               "sqrt(d_model)": round(scale, 4)},
                  {"rel_err": "< 1e-4"},
                  f"W_E holds the scale pre-folded ({folded}); W_E is unscaled "
                  f"({unscaled}); ckpt tensor: {emb_key}")

        tied_ok = (W_U.shape == hf_emb.T.shape
                   and float((W_U - hf_emb.T[:, :W_U.shape[1]]).abs().max()) < 1e-5)
        rep.check("S3b", "unembedding tied to the checkpoint embedding, UNSCALED "
                         "(W_U == ckpt_embed.T)", tied_ok,
                  {"W_U": tuple(W_U.shape), "ckpt": tuple(hf_emb.shape),
                   "max_abs_diff": (float((W_U - hf_emb.T[:, :W_U.shape[1]]).abs().max())
                                    if W_U.shape[0] == hf_emb.shape[1] else None)},
                  "W_U == ckpt_embed.T",
                  f"text_config.tie_word_embeddings = "
                  f"{getattr(txt, 'tie_word_embeddings', None)}; TL cfg.tie_word_embeddings "
                  f"= {getattr(cfg, 'tie_word_embeddings', None)}")

        # Exactly-once: if W_E carries the scale then W_U must not, and vice versa.
        ratio_EU = float(W_E.abs().max()) / max(float(W_U.abs().max()), 1e-12)
        expect = scale if folded else 1.0
        rep.check("S3c", "scale applied on ONE side only (|W_E|max / |W_U|max)",
                  abs(ratio_EU - expect) / expect < 1e-3, round(ratio_EU, 4),
                  round(expect, 4),
                  "a value of d_model here would mean the scale is applied twice")

    # ================================================================ S4 qk-norm etc
    print("\n## S4 — QK-norm / query_pre_attn_scalar / softcapping")
    qpas = getattr(txt, "query_pre_attn_scalar", None)
    want_attn_scale = math.sqrt(qpas) if qpas else math.sqrt(cfg.d_head)
    n_as, attn_scale = first(cfg, "attn_scale")
    rep.check("S4a", "cfg.attn_scale == sqrt(query_pre_attn_scalar)",
              (abs(attn_scale - want_attn_scale) < 1e-4) if attn_scale is not None else None,
              attn_scale, want_attn_scale,
              f"text_config.query_pre_attn_scalar = {qpas}; note sqrt(d_head) would be "
              f"{math.sqrt(cfg.d_head):.4f} — for gemma-3 these coincide only when "
              f"query_pre_attn_scalar == head_dim")

    n_qk, use_qk = first(cfg, "use_qk_norm", "qk_norm", "use_qk_layernorm")
    has_q_norm = any("q_norm" in n or "k_norm" in n
                     for n, _ in tl.model.blocks[0].attn.named_parameters())
    rep.check("S4b", "QK-norm present (Gemma-3 replaces Gemma-2's attn softcapping)",
              bool(use_qk) or has_q_norm,
              {"cfg_attr": n_qk, "cfg_value": use_qk,
               "attn params": [n for n, _ in tl.model.blocks[0].attn.named_parameters()]},
              True)

    a_cap = getattr(txt, "attn_logit_softcapping", None)
    f_cap = getattr(txt, "final_logit_softcapping", None)
    n_ac, tl_ac = first(cfg, "attn_scores_soft_cap", "attn_logit_softcapping")
    n_fc, tl_fc = first(cfg, "output_logits_soft_cap", "final_logit_softcapping")
    rep.check("S4c", "attention-logit softcapping matches config",
              _cap_eq(tl_ac, a_cap), {"tl": tl_ac, "attr": n_ac}, a_cap,
              "Gemma-3 declares null/None; TL encodes 'off' as -1.0 or 0.0 in some versions")
    rep.check("S4d", "final-logit softcapping matches config",
              _cap_eq(tl_fc, f_cap), {"tl": tl_fc, "attr": n_fc}, f_cap)

    n_ne, norm_ba = first(cfg, "use_normalization_before_and_after")
    rep.check("S4e", "post-attn/post-MLP norms present (Gemma sandwich norm)",
              bool(norm_ba) if norm_ba is not None else None, norm_ba, True,
              f"cfg attr: {n_ne}")

    # ================================================================ verdict
    print("\n" + "=" * 78)
    n_fail, n_unk = len(rep.failed), len(rep.unknown)
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"GATE 2 [{args.model}] : {verdict}   "
          f"({len(rep.rows) - n_fail - n_unk} pass / {n_fail} fail / {n_unk} unknown)")
    for r in rep.failed:
        print(f"   FAIL {r['id']}: {r['desc']}")
    for r in rep.unknown:
        print(f"   ??   {r['id']}: {r['desc']}")
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "model": args.model, "revision": rev, "dtype": args.dtype,
        "hf_architectures": hf_cfg.architectures,
        "text_config_nested": nested,
        "text_config_model_type": txt.model_type,
        "load_seconds": round(load_s, 1),
        "tl_cfg": {k: _j(v) for k, v in vars(cfg).items()
                   if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
        "verdict": verdict,
        "n_pass": len(rep.rows) - n_fail - n_unk, "n_fail": n_fail, "n_unknown": n_unk,
        "checks": rep.rows,
    }
    p = os.path.join(args.out_dir, f"gate2_text_tower.{tag}.json")
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[written] {p}")
    return 0 if n_fail == 0 else 1


def hf_embed_matrix(model_id: str, revision: str | None):
    """The token-embedding tensor read straight out of the local safetensors shards.

    Cheaper than instantiating the HF model and, more importantly, it is *independent* of
    both TL and transformers, so it is a real reference for the embedding-scale check.
    """
    import glob as _glob

    snap = B.snapshot_dir(model_id, revision)
    if snap is None:
        return None, None
    try:
        from safetensors.torch import safe_open
    except Exception:
        return None, None
    for path in sorted(_glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight") and "vision" not in k:
                    return f.get_tensor(k), k
    return None, None


def _rope_bases(txt):
    """Gemma-3's two rotary bases, wherever this `transformers` version keeps them.

    ⚠ Older code reads `txt.rope_local_base_freq` and `txt.rope_theta`.  In
    transformers 5.12.1 those flat attributes are gone from `Gemma3TextConfig`; the bases
    live in a per-layer-type dict (`rope_scaling` / `rope_parameters`) keyed by
    'sliding_attention' and 'full_attention'.  Returns (local, global, source).
    """
    loc = getattr(txt, "rope_local_base_freq", None)
    glo = getattr(txt, "rope_theta", None)
    if loc is not None and glo is not None:
        return float(loc), float(glo), "rope_local_base_freq/rope_theta"

    for attr in ("rope_parameters", "rope_scaling"):
        d = getattr(txt, attr, None)
        if isinstance(d, dict) and {"sliding_attention", "full_attention"} <= set(d):
            l = d["sliding_attention"].get("rope_theta")
            g = d["full_attention"].get("rope_theta")
            if l is not None and g is not None:
                return float(l), float(g), f"text_config.{attr}[<layer_type>]['rope_theta']"
    return (float(loc) if loc is not None else None,
            float(glo) if glo is not None else None, "NOT FOUND")


def _rope_factors(txt) -> tuple[float, float]:
    """(local, global) linear RoPE-scaling factors.  1.0 when the layer type declares
    `rope_type: default`.  gemma-3-4b uses factor 8.0 on full-attention layers only."""
    out = {}
    for attr in ("rope_parameters", "rope_scaling"):
        d = getattr(txt, attr, None)
        if isinstance(d, dict) and {"sliding_attention", "full_attention"} <= set(d):
            for k in ("sliding_attention", "full_attention"):
                e = d[k]
                out[k] = float(e.get("factor", 1.0)) \
                    if str(e.get("rope_type", "default")) == "linear" else 1.0
            return out["sliding_attention"], out["full_attention"]
    d = getattr(txt, "rope_scaling", None)
    if isinstance(d, dict) and str(d.get("rope_type")) == "linear":
        return 1.0, float(d.get("factor", 1.0))
    return 1.0, 1.0


def _is_local(v) -> bool:
    s = str(v).lower()
    if s in ("true", "false"):
        return s == "true"
    return "slid" in s or "local" in s


def _round(xs):
    return [round(float(x), 1) for x in xs[:4]]


def _cap_eq(tl_v, hf_v) -> bool:
    off_tl = tl_v is None or (isinstance(tl_v, (int, float)) and tl_v <= 0)
    off_hf = hf_v is None or (isinstance(hf_v, (int, float)) and hf_v <= 0)
    if off_hf:
        return off_tl
    return tl_v is not None and abs(float(tl_v) - float(hf_v)) < 1e-6


if __name__ == "__main__":
    raise SystemExit(main())
