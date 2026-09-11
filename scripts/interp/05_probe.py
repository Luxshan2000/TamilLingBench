#!/usr/bin/env python
"""Layerwise linear probe with the surface-token control.

Two probes, never conflated:

* **P_out (primary)** — label = which Tamil form the model actually emits (argmax over
  tok_pos vs tok_neg at the decision position).  Answers *at what depth has the model
  committed to a value for this slot?*  Well posed in C1 and C3 alike; in C3 it is the only
  well-posed probe, because C3 has no gold.
* **P_cue (secondary, C1 only)** — label = the source-side gold.

**`excess_AUC` is the reportable quantity, not raw AUC.**  In C1 the English cue word is
lexically present, so a probe on the raw residual is near 1.0 from layer 0 and means
nothing.  Two surface controls are trained on the same folds:

    excess_AUC(L) = AUC_resid(L) - max(AUC_bag_of_token_ids, AUC_layer0_embedding)

**GroupKFold on `template_id` is non-negotiable.**  With ~12 templates and a few hundred
examples in d=2560, a non-grouped split memorizes templates and returns AUC ~1.0 at every
layer.  Asserted in code, not by inspection.

Also emits the label-shuffled null band (labels permuted WITHIN template group) — with
d >> n the empirical chance ceiling does not sit at 0.5, and pretending it does is the error.

    python scripts/interp/05_probe.py --cache results/interp/cache/gemma-3-4b-pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def probe_auc(X, y, groups, Cs, seed=0, n_splits=5):
    """Out-of-fold AUC under GroupKFold, with C chosen by an inner grouped split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_groups = len(set(groups))
    assert n_groups >= n_splits, f"only {n_groups} template groups; need >= {n_splits}"
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best, best_auc = Cs[0], -1.0
        if len(Cs) > 1:
            gtr = np.asarray(groups)[tr]
            inner = GroupKFold(n_splits=min(3, len(set(gtr))))
            for C in Cs:
                s = []
                for itr, ite in inner.split(Xtr, y[tr], gtr):
                    if len(set(y[tr][itr])) < 2 or len(set(y[tr][ite])) < 2:
                        continue
                    m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                           max_iter=2000).fit(Xtr[itr], y[tr][itr])
                    s.append(roc_auc_score(y[tr][ite], m.decision_function(Xtr[ite])))
                if s and np.mean(s) > best_auc:
                    best_auc, best = float(np.mean(s)), C
        m = LogisticRegression(penalty="l2", C=best, solver="lbfgs",
                               max_iter=2000).fit(Xtr, y[tr])
        oof[te] = m.decision_function(Xte)
    ok = ~np.isnan(oof)
    if len(set(y[ok])) < 2:
        return float("nan"), oof
    return float(roc_auc_score(y[ok], oof[ok])), oof


def cluster_bootstrap_auc(y, scores, groups, n=2000, seed=0):
    """95% CI resampling CLUSTERS of items grouped by template."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy, ss = y[idx], scores[idx]
        m = ~np.isnan(ss)
        if len(set(yy[m])) < 2:
            continue
        out.append(roc_auc_score(yy[m], ss[m]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def onset(depths, values, thr):
    """Smallest normalized depth at which `values` first reaches `thr` AND stays >= thr at
    every deeper layer.  Linearly interpolated so the statistic is continuous and
    bootstrappable.  1.0 ('never') if the criterion is never met."""
    v = np.asarray(values, dtype=float)
    d = np.asarray(depths, dtype=float)
    ok = v >= thr
    if not ok.any():
        return 1.0
    # last index where the criterion is violated; onset is the crossing just after it
    bad = np.where(~ok)[0]
    start = 0 if len(bad) == 0 else bad[-1] + 1
    if start >= len(v):
        return 1.0
    if start == 0:
        return float(d[0])
    x0, x1, y0, y1 = d[start - 1], d[start], v[start - 1], v[start]
    if y1 == y0:
        return float(x1)
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/interp/cache/gemma-3-4b-pt")
    ap.add_argument("--pairs", default="results/interp/pairs.json")
    ap.add_argument("--out", default="results/interp/probe_gemma-3-4b-pt.csv")
    ap.add_argument("--onsets", default="results/interp/onsets_probe.csv")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    with open(os.path.join(args.cache, "meta.json")) as fh:
        cm = json.load(fh)
    meta = cm["meta"]
    n_layers = cm["n_layers"]
    z = np.load(os.path.join(args.cache, "resid.npz"))
    R = {"pos": z["resid_pos"].astype(np.float32), "neg": z["resid_neg"].astype(np.float32)}
    with open(args.pairs) as fh:
        pairs = {p["pair_id"]: p for p in json.load(fh)["pairs"]}

    print(f"cache: {R['pos'].shape} (pairs, sites, d_model)")
    Cs = [1e-3, 1e-2, 1e-1, 1.0]
    rows, onset_rows = [], []
    t0 = time.time()

    cells = sorted({(m["slot"], m["condition"]) for m in meta})
    for slot, cond in cells:
        sel = [i for i, m in enumerate(meta) if m["slot"] == slot and m["condition"] == cond]
        if len(sel) < 20:
            print(f"skip {slot}/{cond}: only {len(sel)} pairs")
            continue

        # Stack BOTH members of every pair as separate examples.  The + and - runs of one
        # pair are the two classes; grouping by template keeps a pair's two members and
        # every other item from the same template inside one fold.
        groups = np.array([meta[i]["template_id"] for i in sel] * 2)
        # P_out: label = which form the model actually emits at the decision position.
        y_out = np.array([meta[i]["emit_pos_run"] for i in sel]
                         + [meta[i]["emit_neg_run"] for i in sel])
        # P_cue: label = the source-side value (which member's prompt this is).
        y_cue = np.array([1] * len(sel) + [0] * len(sel))

        # --- surface controls, trained on the same folds -------------------
        # (i) bag of token ids, (ii) the layer-0 residual (embeddings only).
        #
        # ⚠ Both controls are TRUNCATED AT THE DECISION POSITION.  The cached sequences
        # continue past p with the full Tamil form, and those trailing tokens ARE the
        # label — a bag-of-tokens over the whole sequence separates the classes perfectly
        # while the residual probe, reading position p of a causal model, cannot see them.
        # That would make the control strictly stronger than the probe and drive
        # excess_AUC to 0 for reasons that have nothing to do with representation.
        def prompt_ids(i, side):
            pr = pairs[meta[i]["pair_id"]]
            return pr[f"ids_{side}"][:pr["p"] + 1]

        vocab = {}
        for i in sel:
            for side in ("pos", "neg"):
                for t in prompt_ids(i, side):
                    vocab.setdefault(t, len(vocab))
        BOT = np.zeros((2 * len(sel), len(vocab)), dtype=np.float32)
        for j, i in enumerate(sel):
            for s, side in enumerate(("pos", "neg")):
                for t in prompt_ids(i, side):
                    BOT[j + s * len(sel), vocab[t]] = 1.0
        EMB = np.concatenate([R["pos"][sel, 0], R["neg"][sel, 0]], 0)

        for label, y in (("P_out", y_out), ("P_cue", y_cue)):
            if label == "P_cue" and cond != "C1":
                continue                      # C3 has no gold; P_cue is undefined
            if len(set(y)) < 2:
                print(f"skip {slot}/{cond}/{label}: single class")
                continue

            auc_bot, _ = probe_auc(BOT, y, groups, Cs[:2])
            auc_emb, _ = probe_auc(EMB, y, groups, Cs)
            surface = max(auc_bot, auc_emb)
            print(f"\n{slot}/{cond}/{label}  n={len(y)}  "
                  f"templates={len(set(groups))}  base rate={y.mean():.2f}")
            print(f"  surface controls: bag-of-tokens {auc_bot:.3f}  "
                  f"layer0-embed {auc_emb:.3f}  -> max {surface:.3f}")

            depths, excess = [], []
            for L in range(0, n_layers + 1, args.layer_stride):
                X = np.concatenate([R["pos"][sel, L], R["neg"][sel, L]], 0)
                auc, oof = probe_auc(X, y, groups, Cs)
                lo, hi = cluster_bootstrap_auc(y, oof, groups, n=args.boot)
                exc = auc - surface

                null = np.nan
                if args.n_perm and L % max(1, (n_layers // 4)) == 0:
                    rng = np.random.default_rng(1234 + L)
                    perms = []
                    for _ in range(max(20, args.n_perm // 10)):
                        yp = y.copy()
                        for g in set(groups):
                            m = groups == g
                            yp[m] = rng.permutation(y[m])
                        a, _ = probe_auc(X, yp, groups, Cs[:1])
                        perms.append(a)
                    null = float(np.nanpercentile(perms, 95))

                depths.append(L / n_layers)
                excess.append(exc)
                rows.append({"model": "gemma-3-4b-pt", "slot": slot, "condition": cond,
                             "probe": label, "layer": L, "depth": L / n_layers,
                             "n": len(y), "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
                             "auc_bagoftokens": auc_bot, "auc_embed": auc_emb,
                             "surface_max": surface, "excess_auc": exc,
                             "shuffled_null_p95": null})
                if L % 4 == 0 or L == n_layers:
                    print(f"    L{L:>2} d={L / n_layers:.2f}  AUC {auc:.3f} "
                          f"[{lo:.3f},{hi:.3f}]  excess {exc:+.3f}"
                          + (f"  null_p95 {null:.3f}" if null == null else ""))

            on = onset(depths, excess, args.threshold)
            onset_rows.append({"model": "gemma-3-4b-pt", "slot": slot, "condition": cond,
                               "probe": label, "threshold": args.threshold,
                               "onset_probe": on,
                               "max_excess_auc": float(np.nanmax(excess)),
                               "n_pairs": len(sel)})
            print(f"  -> onset_probe({slot},{cond},{label}) = {on:.3f}  "
                  f"(max excess {np.nanmax(excess):+.3f})")

    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.onsets, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(onset_rows[0].keys()))
        w.writeheader()
        w.writerows(onset_rows)
    print(f"\n[written] {args.out}  ({time.time() - t0:.0f}s)")
    print(f"[written] {args.onsets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
