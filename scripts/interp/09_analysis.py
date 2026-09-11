#!/usr/bin/env python
"""Evaluate the pre-registered test against the frozen thresholds.

Reads `results/interp/preregistration.json` (frozen BEFORE any probe ran), the probe/lens/
patch outputs, and reports the primary test **with its result whatever the sign**, plus the
Holm-corrected secondary comparisons and an explicit kill-criteria verdict.

The primary statistic:

    Delta_onset = onset_patch(rationality, C1) - onset_patch(number, C1)

with a 95% percentile bootstrap CI on the DIFFERENCE, computed from the paired cluster-
bootstrap draws saved by 08_patch.py (same draw index = same resample, so the difference is
correctly paired).

    python scripts/interp/09_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def holm(pvals, labels):
    order = np.argsort(pvals)
    m = len(pvals)
    out, running = {}, 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[labels[i]] = running
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp")
    ap.add_argument("--out", default="results/interp/verdict.json")
    args = ap.parse_args()
    D = args.dir

    with open(os.path.join(D, "preregistration.json")) as fh:
        prereg = json.load(fh)
    thr = prereg["thresholds"]

    onsets = list(csv.DictReader(open(os.path.join(D, "onsets.csv")))) \
        if os.path.exists(os.path.join(D, "onsets.csv")) else []
    probe_onsets = list(csv.DictReader(open(os.path.join(D, "onsets_probe.csv")))) \
        if os.path.exists(os.path.join(D, "onsets_probe.csv")) else []

    def draws(slot, cond, direction):
        p = os.path.join(D, f"bootdraws_{slot}_{cond}_{direction}.npy")
        return np.load(p) if os.path.exists(p) else None

    def onset_of(slot, cond, direction="mean"):
        for r in onsets:
            if r["slot"] == slot and r["condition"] == cond \
                    and r["direction"] == direction:
                return float(r["onset_patch"]), float(r["ci_lo"]), float(r["ci_hi"])
        return None, None, None

    report = {"preregistration_frozen": prereg["frozen_utc"],
              "thresholds": thr, "primary": {}, "secondary": {}, "kill_criteria": {}}

    print("=" * 78)
    print("PRE-REGISTERED PRIMARY TEST")
    print(prereg["primary_test"]["statistic"])
    print("=" * 78)

    on_r, r_lo, r_hi = onset_of("rationality", "C1")
    on_n, n_lo, n_hi = onset_of("number", "C1")
    d_r, d_n = draws("rationality", "C1", "mean"), draws("number", "C1", "mean")

    if on_r is None or on_n is None:
        print("!! patching onsets unavailable — primary test NOT EVALUABLE")
        report["primary"] = {"status": "NOT EVALUABLE",
                             "reason": "onsets.csv missing rationality/C1 or number/C1"}
    else:
        delta = on_r - on_n
        ci = (None, None)
        if d_r is not None and d_n is not None:
            n = min(len(d_r), len(d_n))
            dd = d_r[:n] - d_n[:n]          # paired: same resample index
            ci = (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
            # one-sided p: fraction of draws where the predicted ordering fails
            p_one = float(np.mean(dd <= 0))
        else:
            p_one = float("nan")
        excludes0 = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
        clears_floor = delta >= thr["effect_floor_delta_onset"]
        print(f"  onset_patch(rationality, C1) = {on_r:.3f}  [{r_lo:.3f}, {r_hi:.3f}]")
        print(f"  onset_patch(number,      C1) = {on_n:.3f}  [{n_lo:.3f}, {n_hi:.3f}]")
        print(f"  Delta_onset                  = {delta:+.3f}  "
              f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None
              else f"  Delta_onset = {delta:+.3f}")
        print(f"  one-sided bootstrap p (rationality NOT later) = {p_one:.4f}")
        print(f"  effect floor >= {thr['effect_floor_delta_onset']}: "
              f"{'MET' if clears_floor else 'NOT MET'}")
        print(f"  CI excludes 0: {'YES' if excludes0 else 'NO'}")
        verdict = ("DISSOCIATION DETECTED" if (clears_floor and excludes0)
                   else "NO DETECTABLE DISSOCIATION")
        print(f"  --> {verdict}")
        report["primary"] = {
            "onset_rationality_C1": on_r, "ci_rationality": [r_lo, r_hi],
            "onset_number_C1": on_n, "ci_number": [n_lo, n_hi],
            "delta_onset": delta, "delta_ci95": list(ci),
            "one_sided_p": p_one,
            "clears_effect_floor": bool(clears_floor),
            "ci_excludes_zero": bool(excludes0),
            "verdict": verdict,
        }

    # ---------------------------------------------------------------- control
    print("\n" + "=" * 78)
    print("NUMBER CONTROL (kill criterion #5 — the Ferrando & Costa-jussa half)")
    print("=" * 78)
    ctrl = None
    for r in onsets:
        if r["slot"] == "number" and r["condition"] == "C1" and r["direction"] == "mean":
            ctrl = r
    if ctrl:
        max_r = float(ctrl["max_r"])
        reproduced = max_r >= thr["patch_restored_effect"]
        print(f"  max mean restored effect r = {max_r:.3f} "
              f"(needs >= {thr['patch_restored_effect']} for the control to reproduce)")
        print(f"  onset_patch(number, C1) = {float(ctrl['onset_patch']):.3f}")
        print(f"  --> control {'REPRODUCES' if reproduced else 'DOES NOT REPRODUCE'} "
              "the published positive result")
        report["number_control"] = {"max_r": max_r, "reproduced": bool(reproduced),
                                    "onset": float(ctrl["onset_patch"])}

    # ---------------------------------------------------------------- secondary
    print("\n" + "=" * 78)
    print("SECONDARY COMPARISONS (Holm-corrected)")
    print("=" * 78)
    labels, pv, detail = [], [], {}
    for slot in ("gender", "rationality"):
        for cond in ("C1",):
            a, _, _ = onset_of(slot, cond)
            b, _, _ = onset_of("number", cond)
            da, db = draws(slot, cond, "mean"), draws("number", cond, "mean")
            if a is None or b is None or da is None or db is None:
                continue
            if slot == "rationality" and cond == "C1":
                continue                       # that is the primary, not a secondary
            n = min(len(da), len(db))
            dd = da[:n] - db[:n]
            p = float(np.mean(dd <= 0))
            labels.append(f"Delta_onset({slot} - number), {cond}")
            pv.append(max(p, 1.0 / n))
            detail[labels[-1]] = {
                "delta": a - b,
                "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                "p_raw": p}
    # asymmetry between patching directions
    for slot in ("number", "rationality", "gender"):
        pm = [r for r in onsets if r["slot"] == slot and r["condition"] == "C1"]
        d = {r["direction"]: r for r in pm}
        if "+-" in d and "-+" in d:
            asym = float(d["+-"]["onset_patch"]) - float(d["-+"]["onset_patch"])
            detail[f"direction_asymmetry({slot})"] = {
                "onset_plus_to_minus": float(d["+-"]["onset_patch"]),
                "onset_minus_to_plus": float(d["-+"]["onset_patch"]),
                "asymmetry": asym,
                "max_r_plus_to_minus": float(d["+-"]["max_r"]),
                "max_r_minus_to_plus": float(d["-+"]["max_r"])}
            print(f"  asymmetry({slot}): +->- onset {float(d['+-']['onset_patch']):.3f} "
                  f"(max r {float(d['+-']['max_r']):.3f})  vs  "
                  f"-->+ onset {float(d['-+']['onset_patch']):.3f} "
                  f"(max r {float(d['-+']['max_r']):.3f})")

    if pv:
        adj = holm(np.array(pv), labels)
        for lab in labels:
            detail[lab]["p_holm"] = adj[lab]
            print(f"  {lab}: delta {detail[lab]['delta']:+.3f} "
                  f"CI {detail[lab]['ci95']}  p_holm {adj[lab]:.4f}")
    report["secondary"] = detail

    # ---------------------------------------------------------------- probe
    if probe_onsets:
        print("\n" + "=" * 78)
        print("PROBE ONSETS (excess_AUC >= "
              f"{thr['probe_excess_auc']}, surface-controlled)")
        print("=" * 78)
        report["probe"] = []
        for r in probe_onsets:
            on = float(r["onset_probe"])
            mx = float(r["max_excess_auc"])
            # "onset 1.0" means the excess_AUC criterion was NEVER met, which is a
            # different statement from "the feature is decodable, but only in the last
            # layers".  Collapsing the two would let an uninformative probe masquerade as
            # a late-onset finding, so they are banded separately.
            band = ("NEVER REACHED" if mx < thr["probe_excess_auc"]
                    else "EARLY" if on <= thr["early_depth"]
                    else "LATE" if on >= thr["late_depth"] else "INCONCLUSIVE")
            print(f"  {r['slot']:<12} {r['condition']}  {r['probe']:<6} "
                  f"onset {on:.3f}  max excess_AUC {mx:+.3f}  -> {band}")
            report["probe"].append({**r, "band": band})

    # ---------------------------------------------------------------- kill
    print("\n" + "=" * 78)
    print("KILL CRITERIA")
    print("=" * 78)
    kc = {}
    pr = report.get("primary", {})
    kc["primary_below_floor_or_ci_includes_0"] = bool(
        pr.get("verdict") == "NO DETECTABLE DISSOCIATION")
    kc["number_control_failed"] = not report.get("number_control", {}).get(
        "reproduced", False)
    bands = [p["band"] for p in report.get("probe", [])
             if p["slot"] in ("rationality", "gender") and p["probe"] == "P_out"]
    kc["both_unmarked_slots_inconclusive"] = (
        len(bands) >= 2
        and sum(b in ("INCONCLUSIVE", "NEVER REACHED") for b in bands) >= 2)
    kc["tuned_lens_missing"] = not os.path.exists(
        os.path.join(D, "tuned_lens", "translators.pt"))
    for k, v in kc.items():
        print(f"  {k}: {v}")
    fire = [k for k, v in kc.items() if v and k != "tuned_lens_missing"]
    print(f"\n  --> {'DROP the section' if fire else 'SECTION SURVIVES'}"
          + (f"  (triggered: {fire})" if fire else ""))
    report["kill_criteria"] = kc
    report["kill_verdict"] = "DROP" if fire else "KEEP"

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
