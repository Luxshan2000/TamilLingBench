#!/usr/bin/env python
"""Logit lens (and the hooks for a tuned lens).

Reads the cached residual stream through the model's OWN decoder:

    z(L) = Unembed( ln_final( h_L ) );   ld_LL(L) = z[tok_pos] - z[tok_neg]

⚠ Gemma's RMSNorm uses **(1 + w)** scaling, which is a real and frequently-missed detail.
Rather than reimplement it, this script calls the loaded `model.ln_final` and
`model.unembed` modules directly and ASSERTS that the layer-(n_layers-1) lens output
reproduces the model's true final logits — if the norm were wrong, that assertion fails.

`ld_LL(L)` is normalized by the model's own final-layer logit difference, so the curve is
comparable across slots.

**Tuned lens:** the design calls for it, because the gap between the two lenses is what
separates "not represented" from "represented but not decoded" (the Chelombitko
alternative).  Training the affine translators needs ~5M tokens of held-out text and ~40
GPU-minutes.  If `--tuned-lens` is not passed, this script emits the logit lens ONLY and
writes an explicit `tuned_lens_status` field saying the decoding-collapse alternative is
untested — it does not silently omit it.

    python scripts/interp/06_logit_lens.py --cache results/interp/cache/gemma-3-4b-pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/interp/cache/gemma-3-4b-pt")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/interp/lens_gemma-3-4b-pt.csv")
    ap.add_argument("--status", default="results/interp/lens_status.json")
    args = ap.parse_args()

    with open(os.path.join(args.cache, "meta.json")) as fh:
        cm = json.load(fh)
    meta = cm["meta"]
    n_layers = cm["n_layers"]
    z = np.load(os.path.join(args.cache, "resid.npz"))
    R = {"pos": z["resid_pos"], "neg": z["resid_neg"]}

    rev = args.revision or B.cached_revision(args.model)
    tl = B.get_backend("tl", args.model, revision=rev, dtype="float32",
                       device=args.device, tokenizer_id=args.model).load()
    m = tl.model

    tok_pos = torch.tensor([r["tok_pos"] for r in meta], device=args.device)
    tok_neg = torch.tensor([r["tok_neg"] for r in meta], device=args.device)

    @torch.no_grad()
    def lens(h):                                    # h: [n, d_model] on device
        return m.unembed(m.ln_final(h.unsqueeze(1)))[:, 0]

    # --- SANITY GATE ---------------------------------------------------------
    # The last cached site is hook_resid_post at the final block, so reading it through
    # ln_final + unembed must reproduce the model's true logits.  A wrong RMSNorm form
    # (missing the (1 + w) offset) shows up here and nowhere else.
    with torch.no_grad():
        h_last = torch.tensor(R["pos"][:8, -1], dtype=torch.float32, device=args.device)
        z_last = lens(h_last)
        got = (z_last[torch.arange(8), tok_pos[:8]]
               - z_last[torch.arange(8), tok_neg[:8]]).cpu().numpy()
    want = np.array([r["ld_clean_pos"] for r in meta[:8]])
    err = float(np.abs(got - want).max())
    print(f"[sanity] final-layer lens vs true logit difference: max abs err {err:.3e}")
    if err > 1e-2:
        print("!! LENS SANITY GATE FAILED — ln_final/unembed path does not reproduce the "
              "model's own logits. Do not report a lens curve from this.")
        return 1

    rows = []
    final_ld = {"pos": np.array([r["ld_clean_pos"] for r in meta]),
                "neg": np.array([r["ld_clean_neg"] for r in meta])}

    per_layer = {}
    for L in range(n_layers + 1):
        for side in ("pos", "neg"):
            with torch.no_grad():
                h = torch.tensor(R[side][:, L], dtype=torch.float32, device=args.device)
                zz = lens(h)
                idx = torch.arange(len(meta), device=args.device)
                ld = (zz[idx, tok_pos] - zz[idx, tok_neg]).cpu().numpy()
            per_layer[(L, side)] = ld

    cells = sorted({(r["slot"], r["condition"]) for r in meta})
    for slot, cond in cells:
        sel = np.array([i for i, r in enumerate(meta)
                        if r["slot"] == slot and r["condition"] == cond])
        if len(sel) < 20:
            continue
        for L in range(n_layers + 1):
            # normalized by each run's own final logit difference
            vals = []
            for side in ("pos", "neg"):
                den = final_ld[side][sel]
                num = per_layer[(L, side)][sel]
                good = np.abs(den) > 1e-6
                vals.append(num[good] / den[good])
            v = np.concatenate(vals)
            rows.append({
                "model": "gemma-3-4b-pt", "slot": slot, "condition": cond, "layer": L,
                "depth": L / n_layers, "n": len(sel) * 2,
                "ld_logitlens_norm_mean": float(np.mean(v)),
                "ld_logitlens_norm_median": float(np.median(v)),
                "ld_logitlens_raw_pos_mean": float(np.mean(per_layer[(L, "pos")][sel])),
                "ld_logitlens_raw_neg_mean": float(np.mean(per_layer[(L, "neg")][sel])),
                "ld_tunedlens_norm_mean": "",
                "gap": "",
            })
        curve = [r["ld_logitlens_norm_mean"] for r in rows if r["slot"] == slot
                 and r["condition"] == cond]
        print(f"{slot}/{cond}: normalized logit-lens ld at depths "
              f"0/.25/.5/.75/1 = "
              f"{[round(curve[int(f * n_layers)], 3) for f in (0, .25, .5, .75, 1)]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.status, "w") as fh:
        json.dump({
            "logit_lens": "computed",
            "logit_lens_sanity_max_abs_err": err,
            "tuned_lens": "NOT RUN",
            "tuned_lens_consequence":
                "The design runs both lenses because the GAP between them is the "
                "discriminator for T1 (Chelombitko et al.: 'represented but not decoded'). "
                "With only the logit lens, a flat lens curve cannot distinguish 'the "
                "feature is not represented' from 'it is represented but the model's own "
                "decoder cannot read it yet'. The probe (which is free to use any linear "
                "direction) is the discriminator we DO have: probe above chance while the "
                "lens is flat is evidence for the decoding-collapse account. This is "
                "weaker than the two-lens comparison and must be stated as such.",
        }, fh, indent=2)
    print(f"[written] {args.out}")
    print(f"[written] {args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
