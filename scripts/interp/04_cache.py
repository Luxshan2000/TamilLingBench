#!/usr/bin/env python
"""The one caching pass.

One forward per pair member (batch size 1, fp32), caching the residual stream at the
**decision position** at every layer: `hook_resid_pre` at layer 0 plus `hook_resid_post` at
every block, i.e. n_layers+1 read sites.

Also records, per member, the final-layer logits of the two contrast tokens, so the
behavioural label (which form the model actually emits) is available without a second pass.

    python scripts/interp/04_cache.py --pairs results/interp/pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402


@torch.no_grad()
def run(model, ids, names, p):
    _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
    return torch.stack([cache[n][0, p] for n in names]).float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/interp/pairs.json")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/interp/cache/gemma-3-4b-pt")
    args = ap.parse_args()

    rev = args.revision or B.cached_revision(args.model)
    with open(args.pairs) as fh:
        blob = json.load(fh)
    pairs = blob["pairs"]

    tl = B.get_backend("tl", args.model, revision=rev, dtype=args.dtype,
                       device=args.device, tokenizer_id=args.model).load()
    from transformer_lens import utils as tlu
    names = [tlu.get_act_name("resid_pre", 0)] + \
            [tlu.get_act_name("resid_post", L) for L in range(tl.n_layers)]
    print(f"caching {len(names)} sites x {len(pairs)} pairs x 2 members "
          f"(d_model={tl.d_model})")

    os.makedirs(args.out, exist_ok=True)
    resid_pos, resid_neg = [], []
    meta = []
    t0 = time.time()
    for i, r in enumerate(pairs):
        out = {}
        for side in ("pos", "neg"):
            ids = torch.tensor([r[f"ids_{side}"]], dtype=torch.long, device=args.device)
            # ⚠ no_grad is load-bearing: run_with_cache does NOT disable autograd, so
            # without this every cached activation keeps a graph alive — memory grows and
            # the pass runs several times slower.
            with torch.no_grad():
                logits, cache = tl.model.run_with_cache(
                    ids, names_filter=lambda n: n in names)
                h = torch.stack([cache[n][0, r["p"]] for n in names]).float().cpu()
                row = logits[0, r["p"]].float().cpu()      # reuse; no second forward
            out[side] = h
            out[f"lp_{side}"] = float(row[r["tok_pos"]])
            out[f"ln_{side}"] = float(row[r["tok_neg"]])
            del cache, logits
        # ⚠ float32, NOT float16.  float16 storage was the first choice; on Gemma-3
        # that OVERFLOWS.  Gemma's residual stream grows monotonically with depth and its
        # max |element| passes 65504 at about layer 28 of 34, producing inf in ~550 of 729
        # items at the deepest layers — silently, and exactly in the layers the result is
        # about.  Measured: max|h| per layer runs 16.8 (L0) -> 65504+ (L28..34).
        resid_pos.append(out["pos"].numpy().astype(np.float32))
        resid_neg.append(out["neg"].numpy().astype(np.float32))
        meta.append({
            "pair_id": r["pair_id"], "slot": r["slot"], "condition": r["condition"],
            "template_id": r["template_id"], "gold": r["gold"],
            "value_pos": r["value_pos"], "value_neg": r["value_neg"],
            "tok_pos": r["tok_pos"], "tok_neg": r["tok_neg"], "p": r["p"],
            "divergence_offset": r["divergence_offset"],
            "n_prompt_tokens": r["n_prompt_tokens"],
            # logit difference (pos - neg) at the decision position, in each run
            "ld_clean_pos": out["lp_pos"] - out["ln_pos"],
            "ld_clean_neg": out["lp_neg"] - out["ln_neg"],
            # behavioural label: which form the model actually prefers in the + run
            "emit_pos_run": int(out["lp_pos"] > out["ln_pos"]),
            "emit_neg_run": int(out["lp_neg"] > out["ln_neg"]),
        })
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(pairs)}  {el / (i + 1):.2f}s/pair  "
                  f"eta {(len(pairs) - i - 1) * el / (i + 1) / 60:.1f}min", flush=True)

    np.savez_compressed(
        os.path.join(args.out, "resid.npz"),
        resid_pos=np.stack(resid_pos), resid_neg=np.stack(resid_neg))
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump({"model": args.model, "revision": rev, "dtype": args.dtype,
                   "n_layers": tl.n_layers, "d_model": tl.d_model,
                   "sites": names, "seconds": round(time.time() - t0, 1),
                   "meta": meta}, fh, ensure_ascii=False)
    print(f"[written] {args.out}/resid.npz  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
