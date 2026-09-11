#!/usr/bin/env python
"""Symmetric activation patching, metric = logit difference.

Per Zhang & Nanda (ICLR 2024), probability and logprob saturate; **logit difference is the
metric.**

For a pair sharing a template:
  clean      = the `+` prompt;  LD_clean = logit(tok_pos) - logit(tok_neg) at position p
  corrupted  = the `-` prompt — the MINIMAL-PAIR PARTNER, not noise.  This is a
               clean-to-clean interchange intervention, which avoids the off-distribution
               problems of Gaussian-noise patching.
  intervention = overwrite blocks.{L}.hook_resid_post at position p in the corrupted run
                 with the cached clean activation.

    r(L) = ( LD_patched(L) - LD_corrupted ) / ( LD_clean - LD_corrupted )

**Symmetric**: both directions (+ -> - and - -> +) are run and reported separately.  A large
asymmetry is itself a finding (one direction is the default and needs less evidence to
reach) and must be reported, not averaged away.

`onset_patch` = the normalized depth at which mean r(L) first crosses 0.5 and stays above
it, linearly interpolated.  95% CI by **cluster bootstrap over templates**.

    python scripts/interp/08_patch.py --slots number,rationality,gender --conditions C1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402


def onset(depths, values, thr=0.5):
    v = np.asarray(values, float)
    d = np.asarray(depths, float)
    ok = v >= thr
    if not ok.any():
        return 1.0
    bad = np.where(~ok)[0]
    start = 0 if len(bad) == 0 else bad[-1] + 1
    if start >= len(v):
        return 1.0
    if start == 0:
        return float(d[0])
    x0, x1, y0, y1 = d[start - 1], d[start], v[start - 1], v[start]
    return float(x1) if y1 == y0 else float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))


def cluster_bootstrap_onset(depths, R, templates, thr=0.5, n=2000, seed=0):
    """R: [n_items, n_layers] of per-item r(L).  Resample TEMPLATES, recompute the mean
    curve and its crossing on each draw."""
    rng = np.random.default_rng(seed)
    templates = np.asarray(templates)
    uniq = np.unique(templates)
    idx_by_t = {t: np.where(templates == t)[0] for t in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_t[t] for t in pick])
        out.append(onset(depths, np.nanmean(R[idx], axis=0), thr))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/interp/pairs.json")
    ap.add_argument("--cache", default="results/interp/cache/gemma-3-4b-pt")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--slots", default="number,rationality,gender")
    ap.add_argument("--conditions", default="C1")
    ap.add_argument("--max-pairs", type=int, default=60,
                    help="per (slot, condition); compute is n_pairs x n_layers x 2 forwards")
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/interp/patch_gemma-3-4b-pt.csv")
    ap.add_argument("--onsets", default="results/interp/onsets.csv")
    args = ap.parse_args()

    with open(args.pairs) as fh:
        pairs = {p["pair_id"]: p for p in json.load(fh)["pairs"]}
    with open(os.path.join(args.cache, "meta.json")) as fh:
        cm = json.load(fh)
    meta = cm["meta"]
    n_layers = cm["n_layers"]
    z = np.load(os.path.join(args.cache, "resid.npz"))
    R = {"pos": z["resid_pos"], "neg": z["resid_neg"]}

    slots = tuple(s.strip() for s in args.slots.split(","))
    conds = tuple(c.strip() for c in args.conditions.split(","))
    layers = list(range(0, n_layers, args.layer_stride))

    rev = args.revision or B.cached_revision(args.model)
    tl = B.get_backend("tl", args.model, revision=rev, dtype="float32",
                       device=args.device, tokenizer_id=args.model).load()
    from transformer_lens import utils as tlu

    rows, onset_rows = [], []
    t0 = time.time()
    n_fwd = 0

    for slot in slots:
        for cond in conds:
            sel = [i for i, m in enumerate(meta)
                   if m["slot"] == slot and m["condition"] == cond]
            # Prefer bucket 0 (first-token divergence) when subsampling.
            sel.sort(key=lambda i: (min(meta[i]["divergence_offset"], 2),
                                    meta[i]["pair_id"]))
            sel = sel[:args.max_pairs]
            if len(sel) < 20:
                print(f"skip {slot}/{cond}: {len(sel)} pairs")
                continue

            # ⚠ patching correctness: the two runs must have identical token length up to
            # and including the decision position, or index p is a different linguistic
            # site in the two runs.  Both sequences are therefore TRUNCATED at p+1: the
            # prompts differ (different English cue) but are equal-length by construction
            # (03_pairs.py drops unmatched prompts), and the Tamil prefix up to the
            # divergence token is identical by definition of the divergence index.  Tokens
            # after p cannot affect the logits at p in a causal model, so truncating loses
            # nothing.
            keep, dropped = [], 0
            for i in sel:
                pr = pairs[meta[i]["pair_id"]]
                a, b = pr["ids_pos"][:pr["p"] + 1], pr["ids_neg"][:pr["p"] + 1]
                if len(a) == len(b) and pr["p"] == meta[i]["p"] and len(a) == pr["p"] + 1:
                    keep.append(i)
                else:
                    dropped += 1
            sel = keep
            print(f"\n{slot}/{cond}: {len(sel)} pairs, {dropped} dropped for "
                  f"length/position mismatch, {len(layers)} layers, "
                  f"{len(sel) * len(layers) * 2} forwards")

            templates = [meta[i]["template_id"] for i in sel]
            curves = {"+-": np.full((len(sel), len(layers)), np.nan),
                      "-+": np.full((len(sel), len(layers)), np.nan)}

            for j, i in enumerate(sel):
                pr = pairs[meta[i]["pair_id"]]
                p = pr["p"]
                ids_pos = torch.tensor([pr["ids_pos"][:p + 1]], dtype=torch.long,
                                       device=args.device)
                ids_neg = torch.tensor([pr["ids_neg"][:p + 1]], dtype=torch.long,
                                       device=args.device)
                ld_clean = meta[i]["ld_clean_pos"]      # + run
                ld_corr = meta[i]["ld_clean_neg"]       # - run
                denom = ld_clean - ld_corr
                if abs(denom) < 1e-6:
                    continue

                for li, L in enumerate(layers):
                    for direction, (ids_run, src_arr, ld_base, ld_target) in {
                        # + -> -  : run the CORRUPTED (-) prompt, write the CLEAN (+) resid
                        "+-": (ids_neg, R["pos"], ld_corr, ld_clean),
                        # - -> +  : run the CLEAN (+) prompt, write the CORRUPTED (-) resid
                        "-+": (ids_pos, R["neg"], ld_clean, ld_corr),
                    }.items():
                        vec = torch.tensor(src_arr[i, L + 1], dtype=torch.float32,
                                           device=args.device)

                        def hook(act, hook, _v=vec, _p=p):
                            act[:, _p, :] = _v.to(act.dtype)
                            return act

                        with torch.no_grad():
                            out = tl.model.run_with_hooks(
                                ids_run,
                                fwd_hooks=[(tlu.get_act_name("resid_post", L), hook)])
                            row = out[0, -1].float()   # sequences truncated at p
                            ld_patched = float(row[pr["tok_pos"]] - row[pr["tok_neg"]])
                        n_fwd += 1
                        curves[direction][j, li] = \
                            (ld_patched - ld_base) / (ld_target - ld_base)

                if (j + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"   {j + 1}/{len(sel)}  {n_fwd} forwards  "
                          f"{el / max(n_fwd, 1) * 1000:.0f}ms/forward", flush=True)

            depths = [(L + 1) / n_layers for L in layers]
            mean_curves = {}
            for direction in ("+-", "-+"):
                C = curves[direction]
                mean_curves[direction] = np.nanmean(C, axis=0)
                on = onset(depths, mean_curves[direction])
                lo, hi, draws = cluster_bootstrap_onset(depths, C, templates,
                                                        n=args.boot)
                onset_rows.append({
                    "model": "gemma-3-4b-pt", "slot": slot, "condition": cond,
                    "direction": direction, "n_pairs": len(sel),
                    "n_templates": len(set(templates)),
                    "onset_patch": on, "ci_lo": lo, "ci_hi": hi,
                    "max_r": float(np.nanmax(mean_curves[direction])),
                })
                np.save(os.path.join(os.path.dirname(args.out),
                                     f"bootdraws_{slot}_{cond}_{direction}.npy"), draws)
                print(f"  onset_patch({slot},{cond},{direction}) = {on:.3f} "
                      f"[{lo:.3f},{hi:.3f}]  max r = "
                      f"{np.nanmax(mean_curves[direction]):.3f}")

            # mean of the two directions
            C_mean = np.nanmean(np.stack([curves["+-"], curves["-+"]]), axis=0)
            on = onset(depths, np.nanmean(C_mean, axis=0))
            lo, hi, draws = cluster_bootstrap_onset(depths, C_mean, templates,
                                                    n=args.boot)
            onset_rows.append({
                "model": "gemma-3-4b-pt", "slot": slot, "condition": cond,
                "direction": "mean", "n_pairs": len(sel),
                "n_templates": len(set(templates)),
                "onset_patch": on, "ci_lo": lo, "ci_hi": hi,
                "max_r": float(np.nanmax(np.nanmean(C_mean, axis=0)))})
            np.save(os.path.join(os.path.dirname(args.out),
                                 f"bootdraws_{slot}_{cond}_mean.npy"), draws)
            print(f"  onset_patch({slot},{cond},mean) = {on:.3f} [{lo:.3f},{hi:.3f}]")

            for li, L in enumerate(layers):
                for direction in ("+-", "-+"):
                    C = curves[direction][:, li]
                    rows.append({
                        "model": "gemma-3-4b-pt", "slot": slot, "condition": cond,
                        "direction": direction, "layer": L, "depth": depths[li],
                        "n": int(np.sum(~np.isnan(C))),
                        "r_mean": float(np.nanmean(C)),
                        "r_median": float(np.nanmedian(C)),
                        "r_sd": float(np.nanstd(C)),
                    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.onsets, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(onset_rows[0].keys()))
        w.writeheader()
        w.writerows(onset_rows)
    print(f"\n[written] {args.out}   {n_fwd} forwards in {time.time() - t0:.0f}s")
    print(f"[written] {args.onsets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
