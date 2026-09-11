#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join the raw metric scores back onto the scoring set and produce
direction / magnitude / z / calibration, plus the tidy long frame used for the figures.

Joins are BY LINE INDEX via inputs/index.tsv, never by string equality: segments repeat across
systems and the empty variant is empty, so a string join would silently merge unrelated rows.

Outputs
  outputs/metric_blindness/blindness_long.csv   tidy frame, one row per (record, variant, metric)
  outputs/metric_blindness/aggregate.json       every cell's statistics
  outputs/metric_blindness/headline_sentences.txt  the paper's sentences with numbers filled in
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE = ROOT / "outputs/metric_blindness"

from tamillingbench.metrics.blindness import (                       # noqa: E402
    TOSHIP, calibrate, cluster_bootstrap_ci, direction, holm_bonferroni,
    magnitude, orient, z_floor)

NATIVE_SCALE = {
    "comet22": "0-1", "cometkiwi22": "0-1", "xcometxl": "0-1", "xcometxxl": "0-1",
    "chrf": "0-100", "chrfpp": "0-100",
    "metricx24xl": "0-25 (lower better; oriented)", "metricx24xlqe": "0-25 (lower better; oriented)",
    "metricx23large": "0-25 (lower better; oriented)",
}
MIN_CELL = 20      # pre-registered


IN_DIR, RAW_DIR, SUFFIX = "inputs", "raw", ""


def load_scores() -> dict[str, list[float]]:
    out = {}
    for p in sorted((BASE / RAW_DIR).glob("*.json")):
        d = json.loads(p.read_text())
        if "scores" not in d:
            continue
        out[d["metric"]] = d["scores"]
    return out


def load_index() -> list[dict]:
    with open(BASE / IN_DIR / "index.tsv") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-table", action="store_true")
    ap.add_argument("--config", choices=["main", "heldout"], default="main",
                    help="main = the reference IS the good variant (DEMETR construction); "
                         "heldout = the reference is a different system's verified-correct "
                         "output for the same item, so no variant is an exact copy of it")
    args = ap.parse_args()
    global IN_DIR, RAW_DIR, SUFFIX
    IN_DIR = "inputs" if args.config == "main" else "inputs_heldout"
    RAW_DIR = "raw" if args.config == "main" else "raw_heldout"
    SUFFIX = "" if args.config == "main" else "_heldout"

    # The held-out config reuses the controlled records but relabels the arm, so the record
    # lookup is keyed without the arm and the arm is carried on the index row instead.
    records = {(r["item_id"], r["system_id"], r["prompt_id"], r["arm"]): r
               for r in map(json.loads, open(BASE / "scoring_set.jsonl"))}
    if SUFFIX:
        records.update({(k[0], k[1], k[2], "heldout"): v for k, v in list(records.items())
                        if k[3] == "controlled"})
    index = load_index()
    scores = load_scores()
    print("metrics with scores: " + ", ".join(sorted(scores)))
    n_expect = len({int(row["line"]) for row in index})
    for m, s in scores.items():
        assert len(s) >= n_expect, (m, len(s), n_expect)

    # (record key, variant) -> {metric: oriented score}
    oriented = {m: orient(m, np.asarray(s)) for m, s in scores.items()}
    cell: dict[tuple, dict[str, dict[str, float]]] = defaultdict(dict)
    rows = []
    for row in index:
        key = (row["item_id"], row["system_id"], row["prompt_id"], row["arm"])
        line = int(row["line"])
        vals = {m: float(oriented[m][line]) for m in oriented}
        cell[key][row["variant"]] = vals
        for m, v in vals.items():
            rows.append({**row, "metric": m, "score": v,
                         "native_scale": NATIVE_SCALE.get(m, "?")})

    if args.emit_table:
        with open(BASE / f"blindness_long{SUFFIX}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote blindness_long{SUFFIX}.csv ({len(rows)} rows)")

    # ---------------------------------------------------------------- statistics
    #: (arm, comparison) -> the two variants whose scores are differenced. `good` is always the
    #: FST-verified correct form, so a positive delta always means the metric prefers correctness.
    COMPARISONS = {
        "slot_error": ("good", "bad"),
        "avoidance": ("good", "avoided"),
        "ctrl_rep": ("good", "ctrl_rep"),
        "ctrl_del": ("good", "ctrl_del"),
        "ctrl_ne": ("good", "ctrl_ne"),
        "ctrl_num": ("good", "ctrl_num"),
    }

    out: dict[str, dict] = {"cells": {}, "min_cell": MIN_CELL}
    pv: dict[str, dict[str, float]] = defaultdict(dict)

    def build(sel: list[tuple], label: str, comp: str, metric: str) -> dict | None:
        a, b = COMPARISONS[comp]
        g, x, e, cl = [], [], [], []
        for k in sel:
            v = cell[k]
            if a not in v or b not in v or "empty" not in v:
                continue
            g.append(v[a][metric]); x.append(v[b][metric]); e.append(v["empty"][metric])
            cl.append(records[k]["template_id"])
        if len(g) < MIN_CELL:
            return None
        g, x, e = map(np.asarray, (g, x, e))
        dr = direction(g, x)
        mg = magnitude(g, x, NATIVE_SCALE.get(metric, "?"))
        zf = z_floor(g, x, e, cl)
        dlo, dhi = cluster_bootstrap_ci(g - x, cl)
        cal = calibrate(metric, mg.mean)
        # Absolute LEVELS as well as deltas. A delta says how much the metric moves; a level
        # says what it actually reports for a translation carrying a critical grammatical
        # error, which is the number a practitioner would act on.
        levels = {"good": float(g.mean()), "wrong": float(x.mean()), "empty": float(e.mean())}
        pv[f"{label}|{comp}"][metric] = dr.p_binomial
        return {"metric": metric, "comparison": comp, "cell": label,
                "direction": dr.to_dict(), "magnitude": {**mg.to_dict(),
                                                         "mean_ci": [dlo, dhi]},
                "z_floor": zf.to_dict(), "calibration": cal.to_dict(),
                "levels": levels}

    keys_all = list(cell)
    groupings: dict[str, list[tuple]] = {"ALL|ALL|ALL": keys_all}
    for k in keys_all:
        r = records[k]
        for label in (f"{r['arm']}|ALL|ALL", f"{r['arm']}|{r['slot']}|ALL",
                      f"{r['arm']}|{r['slot']}|{r['system_id']}",
                      f"ALL|{r['slot']}|ALL"):
            groupings.setdefault(label, []).append(k)

    for label, sel in sorted(groupings.items()):
        for comp in COMPARISONS:
            for metric in sorted(oriented):
                res = build(sel, label, comp, metric)
                if res:
                    out["cells"][f"{label}||{comp}||{metric}"] = res

    out["splice_decomposition"] = splice_decomposition(cell, records, oriented)
    out["holm"] = {fam: holm_bonferroni(d) for fam, d in pv.items()}
    (BASE / f"aggregate{SUFFIX}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote aggregate{SUFFIX}.json ({len(out['cells'])} cells)")

    emit_headlines(out, SUFFIX)
    return 0


def splice_decomposition(cell, records, oriented) -> dict:
    """Separate the SLOT signal from a SPLICE/AUTHENTICITY bias. Not pre-registered; added because
    the arms are asymmetric in a way that would otherwise be a free objection.

    In the controlled arm `good` is the system's authentic output and `bad` is spliced. In the
    natural and avoidance arms it is the other way round: `bad` is authentic and `good` is
    spliced. So a metric that merely prefers unspliced text -- a fluency artefact, not blindness
    -- produces exactly the pattern we observe: above chance in the controlled arm, below chance
    in the natural arm. That alternative explanation has to be measured, not waved away.

    Writing the observed deltas as slot signal `s` plus a constant splice bias `b`,

        delta_controlled = s + b        delta_natural = s - b

    gives s = (d_c + d_n) / 2 and b = (d_c - d_n) / 2. The decomposition is computed on the
    ITEM INTERSECTION of the two arms, so it is within-item and not a difference of two
    differently-composed populations. `number` has an empty intersection by construction (a system
    is either right or wrong on a given number item) and reports null rather than a spurious value.
    """
    out = {}
    slots = sorted({records[k]["slot"] for k in cell})
    for metric in sorted(oriented):
        # "ALL" pools every shared item across slots. The pre-registration states P1's magnitude
        # clause on the metric's OVERALL mean delta, so that is the cell the prediction is scored
        # on; the per-slot rows are the detail, not the test.
        for slot in ["ALL"] + slots:
            def deltas(arm):
                acc = {}
                for k in cell:
                    r = records[k]
                    if r["arm"] != arm or (slot != "ALL" and r["slot"] != slot):
                        continue
                    v = cell[k]
                    if "good" not in v or "bad" not in v:
                        continue
                    acc.setdefault(r["item_id"], []).append(
                        v["good"][metric] - v["bad"][metric])
                return acc
            dc, dn = deltas("controlled"), deltas("natural")
            shared = sorted(set(dc) & set(dn))
            if len(shared) < MIN_CELL:
                out[f"{metric}|{slot}"] = {"n_shared_items": len(shared),
                                           "status": "insufficient item overlap"}
                continue
            c = float(np.mean([np.mean(dc[i]) for i in shared]))
            n = float(np.mean([np.mean(dn[i]) for i in shared]))
            sig, bias = (c + n) / 2, (c - n) / 2
            cal = calibrate(metric, sig)
            out[f"{metric}|{slot}"] = {
                "n_shared_items": len(shared),
                "delta_controlled": c, "delta_natural": n,
                "slot_signal": sig, "splice_bias": bias,
                "splice_bias_dominates": bool(abs(bias) > abs(sig)),
                # The de-confounded signal is the one that should be calibrated: it is what the
                # metric charges for the morpheme once the preference for unspliced text is out.
                # SIGNED. `calibrate` works on magnitude, so an unsigned ratio would score a
                # metric that PREFERS the wrong form as if it were highly sensitive to it.
                "slot_signal_ratio_75": (cal.ratio_75 * (1 if sig >= 0 else -1))
                                        if cal.ratio_75 is not None else None,
                "slot_signal_ratio_75_abs": cal.ratio_75,
                "slot_signal_calibrated": cal.available,
                "prefers_the_wrong_form": bool(sig < 0),
            }
    return out


def emit_headlines(out: dict, suffix: str = "") -> None:
    """The paper's sentences, generated. No number in §3 is transcribed by hand."""
    lines = ["# Generated headline sentences",
             "# Every number below comes from outputs/metric_blindness/aggregate.json.", ""]
    for key, c in sorted(out["cells"].items()):
        label, comp, metric = key.split("||")
        if comp != "slot_error" or not label.endswith("|ALL|ALL"):
            continue
        d, m, cal = c["direction"], c["magnitude"], c["calibration"]
        arm = label.split("|")[0]
        s = (f"[{arm}, {metric}] ranks the correct form higher "
             f"{100 * d['win_rate']:.1f}% of the time (n={d['n']}, "
             f"p={d['p_binomial']:.2g}), but the mean penalty is "
             f"{m['mean']:.5f} {m['native_scale']}")
        if cal["available"]:
            s += (f" — {cal['ratio_75']:.2f}x the difference at which humans are expected "
                  f"to agree three times in four (ToShip d_75 = {cal['d_75']:.4f}, fitted "
                  f"scale, x{cal['scale_factor']:g} from native).")
            if cal.get("saturated"):
                s += (f" [the implied agreement, {100 * cal['acc_at_observed']:.1f}%, is the "
                      f"fit's own ceiling a={100 * cal['asymptote']:.1f}% and carries no "
                      f"information here — quote the ratio, not the percentage]")
            else:
                s += (f" Implied human agreement with that ranking: "
                      f"{100 * cal['acc_at_observed']:.1f}%.")
            sd = out.get("splice_decomposition", {}).get(f"{metric}|ALL")
            if sd and sd.get("slot_signal_calibrated"):
                s += (f" DE-CONFOUNDED of the splice/authenticity preference, the slot signal is "
                      f"{sd['slot_signal']:+.5g} = {sd['slot_signal_ratio_75']:+.2f}x d_75.")
        else:
            s += " — no published human-agreement calibration exists for this metric (—)."
        lines.append(s)
    (BASE / f"headline_sentences{suffix}.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote headline_sentences{suffix}.txt ({len(lines) - 3} sentences)")


if __name__ == "__main__":
    sys.exit(main())
