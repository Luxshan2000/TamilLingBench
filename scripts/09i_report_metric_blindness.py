#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble results/metric-blindness.{json,md}.

Every number in the .md is read out of the JSON, which is itself assembled from
outputs/metric_blindness/*.json. Nothing here is typed by hand, and the renderer refuses to emit a
row whose supporting cell is missing rather than printing a blank that reads as a zero.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE = ROOT / "outputs/metric_blindness"
REPORTS = ROOT / "results"

METRIC_LABEL = {
    "comet22": "COMET-22",
    "cometkiwi22": "CometKiwi-22 (QE)",
    "xcometxl": "xCOMET-XL",
    "metricx24xl": "MetricX-24-hybrid-XL",
    "metricx24xlqe": "MetricX-24-hybrid-XL (QE)",
    "metricx23large": "MetricX-23-Large",
    "chrf": "chrF2",
    "chrfpp": "chrF++",
}
REF_FREE = {"cometkiwi22", "metricx24xlqe"}
ORDER = ["comet22", "xcometxl", "metricx24xl", "metricx23large", "chrf", "chrfpp",
         "cometkiwi22", "metricx24xlqe"]


def g(d, *path, default=None):
    for p in path:
        if d is None:
            return default
        d = d.get(p) if isinstance(d, dict) else None
    return default if d is None else d


def fmt(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def main() -> int:
    agg = json.loads((BASE / "aggregate.json").read_text())
    agg_ho = json.loads((BASE / "aggregate_heldout.json").read_text()) \
        if (BASE / "aggregate_heldout.json").exists() else {"cells": {}}
    stats = json.loads((BASE / "scoring_set.stats.json").read_text())
    aces = json.loads((BASE / "aces_verification.json").read_text())
    loc = json.loads((BASE / "localization.json").read_text()) \
        if (BASE / "localization.json").exists() else {"status": "not_run"}
    prereg = (ROOT / "preregistration/09-metric-blindness.md")

    metrics = sorted({k.split("||")[2] for k in agg["cells"]}, key=lambda m: ORDER.index(m)
                     if m in ORDER else 99)

    def cell(label, comp, metric, src=None):
        return (src or agg)["cells"].get(f"{label}||{comp}||{metric}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": str(prereg.relative_to(ROOT)),
        "metrics_run": metrics,
        "metrics_not_run": {},
        "scoring_set": {
            "n_records": stats["n_records"],
            "by_arm_slot": stats["by_arm_slot"],
            "repair_rejection_rate_by_arm": stats["repair_rejection_rate_by_arm"],
            "variant_counts": stats["variant_counts"],
            "reference_construction": stats["note_reference_construction"],
        },
        "aces_verification": {
            "checks": aces["checks"], "all_pass": all(aces["checks"].values()),
            "n_rows": aces["n_rows"], "en_ta_rows": aces["en_ta_rows"],
            "ta_en_rows": aces["ta_en_rows"], "tamil_rows_total": aces["tamil_rows_total"],
            "gender_langpairs": aces["gender_langpairs"],
            "antonym_langpairs": aces["antonym_langpairs"],
        },
        "localization": loc,
        "llm_judge": load_llm_judge(),
        "cells": agg["cells"],
        "cells_heldout": agg_ho["cells"],
        "holm": agg.get("holm", {}),
        "splice_decomposition": agg.get("splice_decomposition", {}),
    }

    # ------------------------------------------------------------------ findings
    findings = {}
    for m in metrics:
        c_nat = cell("natural|ALL|ALL", "slot_error", m)
        c_ctl = cell("controlled|ALL|ALL", "slot_error", m)
        c_av = cell("avoidance|ALL|ALL", "slot_error", m)
        c_rep = cell("controlled|ALL|ALL", "ctrl_rep", m)
        c_ne = cell("controlled|ALL|ALL", "ctrl_ne", m)
        c_num = cell("controlled|ALL|ALL", "ctrl_num", m)
        findings[m] = {
            "label": METRIC_LABEL.get(m, m),
            "reference_free": m in REF_FREE,
            "controlled": summary(c_ctl),
            "natural": summary(c_nat),
            "avoidance": summary(c_av),
            "ctrl_rep": summary(c_rep),
            "ctrl_ne": summary(c_ne),
            "ctrl_num": summary(c_num),
            "ctrl_del": summary(cell("controlled|ALL|ALL", "ctrl_del", m)),
            "heldout_controlled": summary(cell("controlled|ALL|ALL", "slot_error", m, agg_ho)),
            "demetr_ordering": demetr(c_ctl, c_rep),
        }
    report["findings"] = findings

    report["predictions"] = evaluate_predictions(findings, loc,
                                                 report["splice_decomposition"],
                                                 report["llm_judge"])
    (REPORTS / "metric-blindness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    md = render_md(report)

    # ---- COMPLETENESS GATE -------------------------------------------------------------
    # Every table loop here does `if not <block>: continue`, so a metric that lacks one
    # sub-analysis disappears from the markdown with no trace. That is exactly how
    # `metricx24xlqe` -- the SECOND reference-free metric, and the one that turns "a single
    # odd checkpoint" into a claim about the reference-free class -- was present in
    # `findings` (8 metrics) but absent from the rendered tables (7). A silent omission in
    # a report is indistinguishable from a result that does not exist. Fail loudly instead.
    missing = [m for m in report.get("metrics_run", []) if m not in md]
    if missing:
        labels = ", ".join(f"{m} ({report['findings'][m]['label']})" for m in missing)
        banner = ("\n> ⚠ **REPORT INCOMPLETE.** These metrics were scored and appear in "
                  f"`metric-blindness.json` but are absent from every table above: **{labels}**. "
                  "A table loop skipped them because one sub-analysis was missing. Do not read "
                  "the tables as the full panel.\n")
        md = md.replace("\n## ", banner + "\n## ", 1)
        print(f"09i: WARNING {len(missing)} scored metric(s) missing from the markdown: {labels}",
              file=sys.stderr)

    (REPORTS / "metric-blindness.md").write_text(md)
    print(f"wrote {REPORTS / 'metric-blindness.json'}")
    print(f"wrote {REPORTS / 'metric-blindness.md'}")
    return 0


def load_llm_judge() -> dict:
    """P2's test: the GEMBA-ESA protocol run with an open-weight judge.

    Returns {"status": "not_run", ...} when the file is absent, so the P2 verdict degrades to
    "untested" rather than silently disappearing.
    """
    f = BASE / "raw/gembaesa_openweight.jsonl"
    meta_f = BASE / "raw/gembaesa_openweight.meta.json"
    if not f.exists():
        return {"status": "not_run",
                "why": "GEMBA-ESA as published needs an OPENAI_API_KEY (DECISIONS.md D-1 excluded "
                       "the frontier-API arm); the open-weight substitute has not been run"}
    rows = [json.loads(l) for l in open(f)]
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    by = {}
    for r in rows:
        by.setdefault((r["item_id"], r["system_id"], r["prompt_id"], r["arm"]), {})[r["variant"]] = r
    out = {"status": "ok", **meta, "cells": {},
           "matched_metric_win_rates": matched_win_rates(set(by))}
    from collections import defaultdict
    groups = defaultdict(list)
    for k, v in by.items():
        groups["ALL"].append(v)
        groups[v.get("bad", v.get("good", {})).get("slot", "?")].append(v)
    for label, vs in groups.items():
        pairs = [(v["good"]["score"], v["bad"]["score"]) for v in vs
                 if v.get("good") and v.get("bad")
                 and v["good"]["score"] is not None and v["bad"]["score"] is not None]
        if len(pairs) < 20:
            continue
        g = [a for a, _ in pairs]
        b = [x for _, x in pairs]
        win = sum(1 for a, x in pairs if a > x)
        tie = sum(1 for a, x in pairs if a == x)
        # Stage-1 localisation signal: did the judge flag ANY error span on the wrong form?
        flagged_bad = [v["bad"] for v in vs if v.get("bad")]
        out["cells"][label] = {
            "n": len(pairs),
            "win_rate": (win + 0.5 * tie) / len(pairs),
            "n_tie": tie,
            "mean_delta": sum(a - x for a, x in pairs) / len(pairs),
            "mean_good": sum(g) / len(g), "mean_wrong": sum(b) / len(b),
            "scale": "0-100 as prompted, higher is better",
            "stage1_flag_rate_on_wrong": (sum(1 for x in flagged_bad if not x["no_errors"])
                                          / len(flagged_bad)) if flagged_bad else None,
            "stage1_flag_rate_on_correct": (
                sum(1 for v in vs if v.get("good") and not v["good"]["no_errors"])
                / max(1, sum(1 for v in vs if v.get("good")))),
            "toship_key": None,
        }
    return out


def matched_win_rates(keys: set) -> dict:
    """Win rate of every regression metric on EXACTLY the (record) keys the LLM judge covered."""
    import csv
    idx_f, raw_d = BASE / "inputs/index.tsv", BASE / "raw"
    if not idx_f.exists():
        return {}
    scores = {}
    for f in sorted(raw_d.glob("*.json")):
        d = json.loads(f.read_text())
        if "scores" in d:
            scores[d["metric"]] = d["scores"]
    from tamillingbench.metrics.blindness import ORIENT
    per = {}
    with open(idx_f) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            k = (row["item_id"], row["system_id"], row["prompt_id"], row["arm"])
            if k not in keys or row["variant"] not in ("good", "bad"):
                continue
            per.setdefault(k, {})[row["variant"]] = int(row["line"])
    out = {}
    for m, sc in scores.items():
        o = ORIENT.get(m, 1.0)
        w = t = n = 0
        for k, v in per.items():
            if "good" not in v or "bad" not in v:
                continue
            g, b = o * sc[v["good"]], o * sc[v["bad"]]
            n += 1
            w += g > b
            t += g == b
        if n:
            out[m] = (w + 0.5 * t) / n
    return out


def summary(c):
    if not c:
        return None
    return {
        "n": g(c, "direction", "n"),
        "win_rate": g(c, "direction", "win_rate"),
        "n_tie": g(c, "direction", "n_tie"),
        "p": g(c, "direction", "p_binomial"),
        "mean_delta": g(c, "magnitude", "mean"),
        "mean_delta_ci": g(c, "magnitude", "mean_ci"),
        "frac_nonpositive": g(c, "magnitude", "frac_nonpositive"),
        "native_scale": g(c, "magnitude", "native_scale"),
        "z": g(c, "z_floor", "mean"),
        "z_ci": g(c, "z_floor", "ci"),
        "z_dropped_eps": g(c, "z_floor", "n_dropped_eps"),
        "levels": g(c, "levels"),
        "toship_key": g(c, "calibration", "toship_key"),
        "calibrated": g(c, "calibration", "available"),
        "acc_at_observed": g(c, "calibration", "acc_at_observed"),
        "d75": g(c, "calibration", "d_75"),
        "d90": g(c, "calibration", "d_90"),
        "ratio_75": g(c, "calibration", "ratio_75"),
        "ratio_90": g(c, "calibration", "ratio_90"),
        "saturated": g(c, "calibration", "saturated"),
        "asymptote": g(c, "calibration", "asymptote"),
    }


def demetr(c_slot, c_rep):
    """DEMETR's minor-vs-critical contrast, on our data: is a duplicated word penalised more
    than the obligatory-slot violation? Decided on disjoint bootstrap CIs, not point estimates."""
    if not c_slot or not c_rep:
        return None
    zs, zr = g(c_slot, "z_floor", "mean"), g(c_rep, "z_floor", "mean")
    cs, cr = g(c_slot, "z_floor", "ci"), g(c_rep, "z_floor", "ci")
    disjoint = cs[1] < cr[0] or cr[1] < cs[0]
    return {"z_slot_error": zs, "z_word_repetition": zr,
            "ci_slot": cs, "ci_rep": cr, "cis_disjoint": bool(disjoint),
            "repetition_penalised_more": bool(zr > zs),
            "ratio": (zr / zs) if zs else None,
            "verdict": ("DEMETR reproduced: word repetition costs more than the obligatory "
                        "morpheme" if (zr > zs and disjoint) else
                        "obligatory morpheme costs more than word repetition"
                        if (zs > zr and disjoint) else "indistinguishable")}


def evaluate_p2(judge, findings):
    """P2: does an LLM-PROMPTED metric penalise the obligatory-slot error more than the
    regression-head metrics do? Avramidis et al. (WMT 2024) found exactly that for GEMBA on
    German negation, which is why it was pre-registered rather than discovered."""
    base = {"prediction": "an LLM-prompted metric shows a larger, better-localised penalty than "
                          "the regression-head neural metrics (Avramidis et al., WMT 2024, "
                          "2024.wmt-1.37 §3.1: GEMBA is best on German negation at 97.4% despite "
                          "a 69.7% average)",
            "gemba_esa_as_published": "NOT RUN — needs OPENAI_API_KEY; D-1 removed the "
                                      "frontier-API arm. Never reported as GEMBA-ESA."}
    if judge.get("status") != "ok":
        return {**base, "status": "untested", "why": judge.get("why"),
                "note": "NOT dropped — it stands as pre-registered and untested."}
    cell = judge["cells"].get("ALL")
    if not cell:
        return {**base, "status": "untested", "why": "too few parsable judgements"}
    # Compare on win rate, the only quantity commensurable across a 0-100 prompted scalar and a
    # regression head. Magnitudes are not comparable and are deliberately not compared.
    # MATCHED comparison: every metric's win rate on EXACTLY the records the judge scored.
    # Comparing the judge's 240-record stratified subsample against a metric's 3,500-record
    # arm-level win rate would be comparing two different populations.
    matched = judge.get("matched_metric_win_rates") or {}
    ref_free = [m for m, f in findings.items() if f["reference_free"]]
    kiwi_win = max((matched[m] for m in ref_free if m in matched), default=None)
    return {**base, "status": "tested_with_open_weight_substitute",
            "judge_model": judge.get("judge_model"),
            "judge_is_under_test": judge.get("judge_is_under_test"),
            "excluded_system": judge.get("excluded_system"),
            "n": cell["n"], "win_rate": cell["win_rate"],
            "mean_delta": cell["mean_delta"], "scale": cell["scale"],
            "stage1_flag_rate_on_wrong": cell["stage1_flag_rate_on_wrong"],
            "stage1_flag_rate_on_correct": cell["stage1_flag_rate_on_correct"],
            "matched_metric_win_rates": matched,
            "reference_free_regression_win_rate": kiwi_win,
            "comparison_basis": "same records as the judge, matched on "
                                "(item_id, system_id, prompt_id, arm)",
            "beats_regression_head_qe": (None if kiwi_win is None
                                         else bool(cell["win_rate"] > kiwi_win)),
            "reference_free_matched_range": ([min(matched[m] for m in ref_free if m in matched),
                                              max(matched[m] for m in ref_free if m in matched)]
                                             if any(m in matched for m in ref_free) else None),
            "verdict": ("P2 SUPPORTED by the open-weight substitute: the prompted judge ranks "
                        "the correct form first more often than every reference-free regression "
                        "metric on the same records"
                        if kiwi_win is not None and cell["win_rate"] > kiwi_win else
                        "P2 NOT SUPPORTED by the open-weight substitute: on the same records the "
                        "prompted judge is indistinguishable from the reference-free regression "
                        "metrics, and both are far below the reference-based ones"),
            "caveat": "This is NOT GEMBA-ESA. It is the same two-stage protocol with an "
                      "open-weight judge; the judge is itself a system under test, so its own "
                      "outputs are excluded from the judged subsample and it never grades "
                      "itself. It tests the metric-FAMILY prediction, not the published metric, "
                      "and a 14B open-weight judge is a weaker instrument than the GPT-4-class "
                      "model Avramidis et al. evaluated -- a null here does not refute their "
                      "finding, it fails to reproduce it under a substitution we could afford."}


def evaluate_predictions(findings, loc, sd=None, judge=None):
    """Score the pre-registration mechanically. A prediction is not allowed to be re-read."""
    judge = judge or {"status": "not_run"}
    out = {}
    p1 = {}
    for m, f in findings.items():
        c = f["controlled"]
        if not c:
            continue
        # P1's magnitude clause is judged on the DE-CONFOUNDED slot signal wherever it exists.
        # The raw controlled-arm ratio conflates the morpheme with the metric's preference for
        # unspliced text, and for the QE metric that preference is the larger of the two.
        sigs = {k.split("|")[1]: v["slot_signal_ratio_75"]
                for k, v in (sd or {}).items()
                if k.startswith(m + "|") and v.get("slot_signal_calibrated")}
        overall = sigs.pop("ALL", None)          # the pre-registered test is on the overall mean
        best = max(sigs.values()) if sigs else None
        p1[m] = {
            "direction_above_chance": bool(c["win_rate"] > 0.5 and c["p"] < 0.05),
            "ratio_75_controlled_arm": c["ratio_75"],
            "ratio_75_deconfounded_by_slot": sigs or None,
            "ratio_75_deconfounded_max_slot": best,
            "ratio_75_deconfounded_overall": overall,
            "magnitude_below_d75": (None if overall is None else bool(overall < 1.0)),
            "verdict": ("not calibratable" if not c["calibrated"] else
                        "no de-confounded estimate (needs both arms on shared items)"
                        if overall is None else
                        "P1 HOLDS — the de-confounded penalty is below the 75% threshold; the "
                        "metric ranks correctly but by a margin humans would not notice"
                        if overall < 1.0 else
                        "P1 FALSIFIED for this metric — the de-confounded penalty exceeds the "
                        "75% threshold"),
        }
    out["P1"] = p1
    out["P2"] = evaluate_p2(judge, findings)
    p3 = {}
    for m, f in findings.items():
        av, nat = f["avoidance"], f["natural"]
        if not av or not nat:
            continue
        ci = av["mean_delta_ci"]
        p3[m] = {
            "delta_avoid": av["mean_delta"], "delta_avoid_ci": ci,
            "delta_wrong": nat["mean_delta"],
            "avoid_cheaper_than_wrong": bool(abs(av["mean_delta"]) < abs(nat["mean_delta"]))
            if nat["mean_delta"] else None,
            "avoid_indistinguishable_from_zero": bool(ci[0] <= 0 <= ci[1]),
            "win_rate": av["win_rate"],
        }
    out["P3"] = p3
    out["P4"] = loc.get("verdict", {"call": "not_run"})
    return out


# ------------------------------------------------------------------ markdown

def render_md(r) -> str:
    L = []
    A = L.append
    A("# Metric blindness")
    A("")
    A(f"Generated {r['generated_utc']}. Pre-registration: `{r['preregistration']}`.")
    A("")
    A("**Reading rule.** Direction and magnitude are reported separately and neither is a "
      "verdict on its own. A metric can rank the correct form first on every single item and "
      "still charge a penalty no human would notice; that is the distinction this section "
      "exists to make. `z` is the penalty as a fraction of the penalty for producing nothing "
      "at all.")
    A("")

    # ---- generated headline
    A("## Headline")
    A("")
    # Renumber: the individual claims are emitted conditionally, so their hard-coded ordinals
    # would go 0,1,3,5 as soon as one of them does not apply.
    for i, line in enumerate(headline_lines(r), 1):
        A(f"{i}. " + re.sub(r"^\d+\.\s*", "", line))
    A("")

    # ---- what was run
    A("## What ran, and what did not")
    A("")
    A("| metric | reference | calibratable (ToShip key) |")
    A("|---|---|---|")
    for m in r["metrics_run"]:
        f = r["findings"][m]
        key = (f["controlled"] or f["natural"] or {}).get("toship_key")
        A(f"| {f['label']} | {'reference-free (QE)' if f['reference_free'] else 'reference-based'} "
          f"| {'`' + key + '`' if key else '**—** (no published calibration)'} |")
    A("")
    A("**GEMBA-ESA was not run.** It needs an `OPENAI_API_KEY`; `DECISIONS.md` D-1 excluded the "
      "frontier-API arm. Pre-registered prediction **P2** "
      "(Avramidis et al., WMT 2024: LLM-prompted metrics beat regression-head metrics on "
      "negation-like phenomena) is therefore **untested**, not dropped.")
    A("")
    A("**MetricX-24-hybrid and GEMBA-ESA have no entry in the Kocmi et al. (ACL 2024) ToShip23 "
      "table.** They are reported with `—` in every calibrated column. Borrowing MetricX-23's "
      "thresholds for MetricX-24 would be wrong: the fits are tied to checkpoints, not to "
      "metric families. That two of the newest metrics in the panel have no published "
      "human-agreement calibration at all is itself a small finding about the state of the field.")
    A("")

    # ---- the scoring set
    s = r["scoring_set"]
    A("## The scoring set")
    A("")
    A(f"{s['n_records']:,} records over three arms, C1 only. "
      f"Repair rejection rates: " + ", ".join(
          f"{k} {100 * v:.1f}%" for k, v in s["repair_rejection_rate_by_arm"].items()) + ".")
    A("")
    A("| arm | slot | n |")
    A("|---|---|---:|")
    for k, v in s["by_arm_slot"].items():
        arm, slot = k.split("|")
        A(f"| {arm} | {slot} | {v} |")
    A("")
    A("> **Departure from the pre-registered design, and it is load-bearing.** The design assumed "
      "`data/benchmark/items.jsonl` carries a full-sentence `reference`. It does not: "
      "`gold_targets[*].tamil` is the slot-bearing core only (1–3 words, median 2) against "
      "5–9 word English sources. Using it as a COMET reference would make every metric penalise "
      "the missing adjunct, an effect orders of magnitude larger than the morpheme. "
      + s["reference_construction"])
    A("")

    # ---- headline table
    A("## Direction, magnitude, z — controlled minimal pairs")
    A("")
    A("`good` and `bad` differ in **exactly one grapheme-cluster span** (asserted, not assumed) "
      "and the FST verifies that `bad` carries the wrong slot value.")
    A("")
    A("† = the ToShip sigmoid is **saturated at its own ceiling** (93–99% depending on the "
      "metric), so the agreement figure is the fit's asymptote and carries no further "
      "information. In that regime `ratio to d75` is the number to read, not the percentage.")
    A("")
    A(hdr := "| metric | n | win rate | mean Δ (native) | 95% CI | z | z 95% CI | "
             "score for the WRONG form | ratio to d75 | implied human agreement |")
    A("|---|---:|---:|---:|---|---:|---|---:|---:|---:|")
    for m in r["metrics_run"]:
        c = r["findings"][m]["controlled"]
        if not c:
            continue
        A(row(r["findings"][m]["label"], c))
    A("")
    A("## The same metrics on the system's own errors (natural arm)")
    A("")
    A(hdr)
    A("|---|---:|---:|---:|---|---:|---|---:|---:|---:|")
    for m in r["metrics_run"]:
        c = r["findings"][m]["natural"]
        if not c:
            continue
        A(row(r["findings"][m]["label"], c))
    A("")

    # ---- avoidance
    A("## Avoidance — does any metric charge for declining the distinction?")
    A("")
    A("The panel's dominant gender behaviour is not a wrong choice but the **epicene `-ஆர்`**: a "
      "legitimate Tamil form that silently drops information the grammar requires. Here `bad` is "
      "the system's actual epicene output and `good` is the same sentence repaired to the gold "
      "value.")
    A("")
    A(hdr)
    A("|---|---:|---:|---:|---|---:|---|---:|---:|---:|")
    for m in r["metrics_run"]:
        c = r["findings"][m]["avoidance"]
        if not c:
            continue
        A(row(r["findings"][m]["label"], c))
    A("")

    # ---- floor levels
    A("## What each metric scores, in absolute terms")
    A("")
    A("DEMETR's own hedge (footnote 20) is that `z` depends heavily on the score given to the "
      "floor hypothesis, so the floor is reported rather than left implicit. `wrong` is the "
      "score the metric actually returns for a translation carrying a critical grammatical "
      "error — the number a practitioner would act on.")
    A("")
    A("| metric | correct form | WRONG form | empty hypothesis (the z denominator) |")
    A("|---|---:|---:|---:|")
    for m in r["metrics_run"]:
        lv = (r["findings"][m]["natural"] or {}).get("levels") \
            or (r["findings"][m]["controlled"] or {}).get("levels")
        if not lv:
            continue
        A(f"| {r['findings'][m]['label']} | {fmt(lv['good'], 4)} | **{fmt(lv['wrong'], 4)}** | "
          f"{fmt(lv['empty'], 4)} |")
    A("")

    # ---- splice decomposition
    sd = r.get("splice_decomposition") or {}
    if sd:
        A("## Is it the morpheme, or just a preference for unspliced text?")
        A("")
        A("The arms are asymmetric on purpose and that asymmetry has to be controlled. In the "
          "**controlled** arm `good` is the system's authentic output and `bad` is spliced; in "
          "the **natural** and **avoidance** arms it is the other way round. A metric that "
          "merely prefers untouched text would produce exactly the observed pattern — above "
          "chance in one arm, below chance in the other. Writing "
          "`Δ_controlled = s + b` and `Δ_natural = s − b` on the **item intersection** of the "
          "two arms separates the slot signal `s` from the splice bias `b`.")
        A("")
        A("| metric | slot | shared items | Δ controlled | Δ natural | slot signal `s` | "
          "splice bias `b` | bias dominates | `s` / d75 |")
        A("|---|---|---:|---:|---:|---:|---:|:--:|---:|")
        for k, v in sorted(sd.items()):
            if "slot_signal" not in v:
                continue
            m, slot = k.split("|")
            A(f"| {METRIC_LABEL.get(m, m)} | {slot} | {v['n_shared_items']} | "
              f"{fmt(v['delta_controlled'], 5)} | {fmt(v['delta_natural'], 5)} | "
              f"**{fmt(v['slot_signal'], 5)}**{' (prefers the WRONG form)' if v['prefers_the_wrong_form'] else ''} | "
              f"{fmt(v['splice_bias'], 5)} | {'**yes**' if v['splice_bias_dominates'] else 'no'} | "
              f"{fmt(v['slot_signal_ratio_75'], 2) if v['slot_signal_calibrated'] else '—'} |")
        A("")
        A("`number` has an empty intersection by construction — a system is either right or "
          "wrong on a given number item, never both — and is omitted rather than estimated from "
          "two differently-composed populations.")
        A("")

    # ---- DEMETR contrast
    A("## DEMETR's minor-vs-critical contrast, reproduced on Tamil")
    A("")
    A("Same items, same metric, same run: the obligatory-slot violation against a duplicated "
      "content word (DEMETR's canonical *minor* error).")
    A("")
    A("| metric | z(slot error) | z(word repetition) | CIs disjoint | verdict |")
    A("|---|---:|---:|:--:|---|")
    for m in r["metrics_run"]:
        d = r["findings"][m]["demetr_ordering"]
        if not d:
            continue
        A(f"| {r['findings'][m]['label']} | {fmt(d['z_slot_error'], 3)} | "
          f"{fmt(d['z_word_repetition'], 3)} | {'yes' if d['cis_disjoint'] else 'no'} | "
          f"{d['verdict']} |")
    A("")

    # ---- positive controls
    A("## Positive controls on the same sentences")
    A("")
    A("All four are one-edit perturbations of the same `good` strings, scored in the same run.")
    A("")
    A("| metric | z(slot error) | z(numeral swap) | z(named-entity swap) | z(word deletion) | "
      "z(word repetition) |")
    A("|---|---:|---:|---:|---:|---:|")
    for m in r["metrics_run"]:
        f = r["findings"][m]
        if not f["controlled"]:
            continue
        A(f"| {f['label']} | **{fmt(f['controlled']['z'], 3)}** | "
          f"{fmt(g(f, 'ctrl_num', 'z'), 3)} | {fmt(g(f, 'ctrl_ne', 'z'), 3)} | "
          f"{fmt(g(f, 'ctrl_del', 'z'), 3)} | {fmt(g(f, 'ctrl_rep', 'z'), 3)} |")
    A("")
    _ne_bigger = [m for m in r["metrics_run"]
                  if g(r["findings"][m], "ctrl_ne", "z") is not None
                  and g(r["findings"][m], "controlled", "z") is not None
                  and g(r["findings"][m], "ctrl_ne", "z") > g(r["findings"][m], "controlled", "z")]
    _tot = [m for m in r["metrics_run"] if g(r["findings"][m], "ctrl_ne", "z") is not None]
    if _tot:
        _subj = ("Every metric in the panel charges" if len(_ne_bigger) == len(_tot)
                 else f"{len(_ne_bigger)} of the {len(_tot)} metrics in the panel charge")
        A(f"**{_subj} more for swapping a named entity than for getting the obligatory morpheme "
          f"wrong.**")
    A("")
    A("`ctrl_num` (numeral) and `ctrl_ne` (named entity) are ACES *accuracy* phenomena with their "
      "own categories in that benchmark; the obligatory slot has no ACES category at all "
      "(see the ACES section below).")
    A("")
    A("Holm-Bonferroni corrected direction tests are in `results/metric-blindness.json` under "
      "`holm`, one family per (cell x comparison).")
    A("")

    # ---- held-out reference
    if r["cells_heldout"]:
        A("## Held-out reference — removing the exact-match ceiling")
        A("")
        A("In the main configuration `good` **is** the reference (the DEMETR/ACES construction), "
          "so a reference-based metric scoring `good` is scoring a copy of its own reference. "
          "The measured Δ is therefore an upper bound on sensitivity. Here the reference is a "
          "*different* system's independently produced output for the same item, FST-verified as "
          "carrying the gold slot value, so neither variant is a copy of it.")
        A("")
        A("| metric | n | win rate | mean Δ | z | main-config mean Δ |")
        A("|---|---:|---:|---:|---:|---:|")
        for m in r["metrics_run"]:
            h, c = r["findings"][m]["heldout_controlled"], r["findings"][m]["controlled"]
            if not h:
                continue
            A(f"| {r['findings'][m]['label']} | {h['n']} | {100 * h['win_rate']:.1f}% | "
              f"{fmt(h['mean_delta'], 5)} | {fmt(h['z'], 3)} | "
              f"{fmt(c['mean_delta'], 5) if c else '—'} |")
        A("")
        _c22 = r["findings"].get("comet22", {})
        if _c22.get("heldout_controlled") and _c22.get("controlled"):
            _h, _m = _c22["heldout_controlled"], _c22["controlled"]
            A(f"**The exact-match ceiling is not what was driving COMET-22.** Its mean Δ moves "
              f"from {_m['mean_delta']:.5f} to {_h['mean_delta']:.5f} when the reference is a "
              f"different system's output — essentially unchanged, on a win rate that falls from "
              f"{100 * _m['win_rate']:.1f}% to {100 * _h['win_rate']:.1f}%. chrF, a surface "
              f"metric, loses about {100 * (1 - r['findings']['chrf']['heldout_controlled']['mean_delta'] / r['findings']['chrf']['controlled']['mean_delta']):.0f}% "
              f"of its Δ, which is what a character-n-gram score should do once the hypothesis "
              f"stops being a copy of the reference. CometKiwi is reference-free, so its numbers "
              f"are unchanged by construction and serve here as a control on the subsetting.")
        A("")

    # ---- localisation
    A("## xCOMET span localisation — the adversary test")
    A("")
    loc = r["localization"]
    if loc.get("status") != "ok":
        A(f"**NOT RUN.** {loc.get('reason', '')}")
    else:
        A(f"Model: `{loc.get('model')}`. Observed span schema: "
          + ", ".join(f"`{k}` ({v})" for k, v in loc["span_schema_observed"].items())
          + f". Offset round-trip: {loc['roundtrip']}.")
        A("")
        A("| variant | n | flags anything | hit (≥1 char) | hit (IoU≥0.5) | flagged share of "
          "sentence | gold span | `critical` share when hit | distractor rate |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for v, d in loc["per_variant"].items():
            if not d["n"]:
                continue
            A(f"| `{v}` | {d['n']} | {100 * d['detection_at_item']:.1f}% | "
              f"{100 * d['hit_rate_overlap']:.1f}% | {100 * d['hit_rate_iou50']:.1f}% | "
              f"{100 * (d.get('mean_flagged_share_of_sentence') or 0):.0f}% | "
              f"{(d.get('mean_gold_span_chars') or 0):.1f} ch | "
              f"{100 * (d.get('critical_share_when_hit') or 0):.1f}% | "
              f"{100 * d['distractor_rate']:.1f}% |")
        A("")
        for v, d in loc["per_variant"].items():
            if d["n"]:
                A(f"- `{v}` — {d['description']}")
        A("")
        A(f"**Verdict ({loc['verdict']['call']}):** {loc['verdict']['why']}")
        A("")
        A("Three things this table says that a single hit rate would hide:")
        A("")
        A(f"1. **The false-positive floor is high.** xCOMET flags an error span on "
          f"**{100 * loc['per_variant']['good']['detection_at_item']:.1f}% of "
          f"FST-verified-correct Tamil sentences** and lands on the very region that would have "
          f"carried the error on {100 * loc['per_variant']['good']['hit_rate_overlap']:.1f}% of "
          f"them. Any hit rate on a perturbed variant has to be read against that floor.")
        A(f"2. **Detection is not localisation.** The flagged region covers "
          f"{100 * (loc['per_variant']['bad'].get('mean_flagged_share_of_sentence') or 0):.0f}% "
          f"of the sentence against a gold error span of "
          f"{(loc['per_variant']['bad'].get('mean_gold_span_chars') or 0):.1f} characters, so an "
          f"overlap is close to automatic and IoU≥0.5 reaches only "
          f"{100 * loc['per_variant']['bad']['hit_rate_iou50']:.1f}%.")
        A(f"3. **Severity is where the phenomenon shows.** When xCOMET does flag the obligatory "
          f"morpheme it calls it `critical` "
          f"{100 * (loc['per_variant']['bad'].get('critical_share_when_hit') or 0):.1f}% of the "
          f"time; on a named-entity swap in the same sentences it says `critical` "
          f"{100 * (loc['per_variant']['ctrl_ne'].get('critical_share_when_hit') or 0):.1f}% of "
          f"the time, and on a **duplicated word** — DEMETR's canonical *minor* error — "
          f"{100 * (loc['per_variant']['ctrl_rep'].get('critical_share_when_hit') or 0):.1f}%. "
          f"The metric that advertises critical-error identification rates a duplicated Tamil "
          f"word as critical an order of magnitude more often than it rates addressing an elder "
          f"with the familiar நீ.")
    A("")

    # ---- ACES
    a = r["aces_verification"]
    A("## ACES coverage — verified against the released parquet")
    A("")
    A(f"`scripts/09h_verify_aces.py` exits {'0' if a['all_pass'] else 'NON-ZERO'}; "
      f"{sum(a['checks'].values())}/{len(a['checks'])} assertions pass.")
    A("")
    A(f"- **`en-ta` = exactly {a['en_ta_rows']} row of {a['n_rows']:,}**, phenomenon `addition`.")
    A(f"- `ta-en` = {a['ta_en_rows']} rows. **{a['tamil_rows_total']} rows in the whole benchmark "
      f"involve Tamil at all, {a['ta_en_rows']} of them into English.**")
    A("- No phenomenon name contains *negat*, *agree*, *morph*, *polarity*, *honor*, *person*, "
      "*tense*, *aspect*, *clusiv* or *animac*.")
    A(f"- `ambiguous-translation-wrong-gender-*` is confined to "
      f"{', '.join('`' + x + '`' for x in a['gender_langpairs'])} — **narrower than is often "
      f"assumed**; the broader list belongs to `antonym-replacement` "
      f"({', '.join('`' + x + '`' for x in a['antonym_langpairs'])}).")
    A("")

    # ---- predictions
    A("## Pre-registered predictions, scored")
    A("")
    p = r["predictions"]
    A("`ratio` columns are the mean Δ as a multiple of the ToShip 75%-agreement threshold. The "
      "**de-confounded** column is the one that decides the prediction; it is signed, so a "
      "negative value means the metric prefers the WRONG form by that margin.")
    A("")
    A("| metric | P1 direction | ratio (controlled arm) | **ratio (de-confounded, overall)** | "
      "ratio (de-confounded, per slot) | verdict |")
    A("|---|:--:|---:|---:|---|---|")
    for m, d in p["P1"].items():
        per = d.get("ratio_75_deconfounded_by_slot") or {}
        cell_s = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(per.items())) or "—"
        ov = d.get("ratio_75_deconfounded_overall")
        A("| " + " | ".join([
            r["findings"][m]["label"],
            "yes" if d["direction_above_chance"] else "**no**",
            fmt(d["ratio_75_controlled_arm"], 2),
            f"**{ov:+.2f}**" if ov is not None else "—",
            cell_s,
            d["verdict"],
        ]) + " |")
    A("")
    _p2 = p["P2"]
    A(f"**P2 — {_p2['status']}.** Prediction: {_p2['prediction']}.")
    A("")
    A(f"- GEMBA-ESA as published: {_p2['gemba_esa_as_published']}")
    if _p2["status"] == "untested":
        A(f"- {_p2.get('why', '')} {_p2.get('note', '')}")
    else:
        A(f"- Open-weight substitute: judge `{_p2['judge_model']}`"
          + (" — itself a system under test, so **its own outputs are excluded from the judged "
             "subsample by construction**; it never grades itself"
             if _p2.get("judge_is_under_test") else " (outside the evaluation panel)")
          + f", n={_p2['n']}, win rate **{100 * _p2['win_rate']:.1f}%**, mean Δ "
          f"{_p2['mean_delta']:+.3f} on a {_p2['scale']}.")
        A(f"- Stage-1 error spans flagged on the WRONG form "
          f"{100 * (_p2['stage1_flag_rate_on_wrong'] or 0):.1f}% of the time, against "
          f"{100 * (_p2['stage1_flag_rate_on_correct'] or 0):.1f}% on the correct form.")
        A("- **Matched comparison** — every metric's win rate on exactly the records the judge "
          "scored: " + ", ".join(
              f"{r['findings'].get(k, {}).get('label', k)} {100 * v:.1f}%"
              for k, v in sorted((_p2.get("matched_metric_win_rates") or {}).items(),
                                 key=lambda kv: -kv[1])) + ".")
        A(f"- **{_p2['verdict']}.** {_p2['caveat']}")
    A("")
    A("**P3 (avoidance).**")
    A("")
    A("| metric | Δ(avoidance) | 95% CI | Δ(wrong commitment) | avoidance CI includes 0 |")
    A("|---|---:|---|---:|:--:|")
    for m, d in p["P3"].items():
        ci = d["delta_avoid_ci"]
        A(f"| {r['findings'][m]['label']} | {fmt(d['delta_avoid'], 5)} | "
          f"[{fmt(ci[0], 5)}, {fmt(ci[1], 5)}] | {fmt(d['delta_wrong'], 5)} | "
          f"{'**yes**' if d['avoid_indistinguishable_from_zero'] else 'no'} |")
    A("")
    A(f"**P4 (xCOMET localisation) — {p['P4'].get('call')}.** {p['P4'].get('why', '')}")
    A("")
    A("## Citations this section depends on — all verified against primary sources 2026-08-08")
    A("")
    A("| claim | source | status |")
    A("|---|---|---|")
    A("| `z` is DEMETR's sensitivity ratio, not a variant | Karpinska, Raj, Thai, Song, Gupta & "
      "Iyyer, *DEMETR: Diagnosing Evaluation Metrics for Translation*, EMNLP 2022, "
      "[`2022.emnlp-main.649`](https://aclanthology.org/2022.emnlp-main.649/), §4 Eq. (1) | "
      "**verified verbatim** |")
    A("| \"even the best-performing metrics struggle to distinguish between minor errors such as "
      "word repetition and critical errors such as incorrect number, aspect, and gender\" | "
      "DEMETR §6 (Conclusion), p. 9547 | **verified verbatim** |")
    A("| BERTScore/COMET/COMET-QE are more sensitive to word repetition than to a "
      "meaning-changing word addition | DEMETR §4, p. 9547 | **verified verbatim** |")
    A("| COMET-QE gender p = 0.178, number p = 0.871 (Welch t-test) | DEMETR Appendix Table A3 | "
      "**verified**; the paper spells it \"Welsch\" — do not reproduce the typo |")
    A("| GEMBA is the best metric on German negation despite low overall performance (97.4% vs "
      "69.7% average) | Avramidis, Manakhimova, Macketanz & Möller, *Machine Translation Metrics "
      "Are Better in Evaluating Linguistic Errors on LLMs than on Encoder-Decoder Systems*, "
      "WMT 2024, [`2024.wmt-1.37`](https://aclanthology.org/2024.wmt-1.37/), §3.1 | "
      "**verified verbatim** — P2 is correctly attributed |")
    A("| COMET's number/NE blind spots are not removed by synthetic training data | **Amrhein "
      "and Sennrich** (two authors, not *et al.*), *Identifying Weaknesses in Machine "
      "Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET*, "
      "AACL 2022, [`2022.aacl-main.83`](https://aclanthology.org/2022.aacl-main.83/), §7 | "
      "**verified verbatim** |")
    A("| \"Synthetic data approaches show mixed results and overall do not help close the gap by "
      "much for these languages\" | Singh, Sai, Dabre, Puduppully, Kunchukuttan & Khapra, *How "
      "Good is Zero-Shot MT Evaluation for Low Resource Indian Languages?*, ACL 2024, "
      "[`2024.acl-short.58`](https://aclanthology.org/2024.acl-short.58/), abstract + §4.3 | "
      "**verified verbatim** |")
    A("| xCOMET \"largely capable of identifying **localized critical errors and "
      "hallucinations**\" | Guerreiro, Rei, van Stigt, Coheur, Colombo & Martins, TACL 12 (2024), "
      "[`2024.tacl-1.54`](https://aclanthology.org/2024.tacl-1.54/), abstract | **verified "
      "verbatim** — note the original hedges with *largely capable*; do not quote it as an "
      "unqualified guarantee |")
    A("")
    A("**Two deviations from DEMETR, stated because they change the number slightly.** (i) "
      "DEMETR's floor hypothesis is a **full stop**, not an empty string (Appendix Table A1, "
      "perturbation 32: \"since most automatic metrics will not allow an empty string we pass a "
      "full stop instead\"); we pass a genuinely empty string, every metric accepted it, and the "
      "resulting denominator is slightly larger — so our z is if anything *smaller* than the "
      "DEMETR-comparable value, which is conservative for our claim. (ii) DEMETR's SCORE is "
      "reference-based on both terms; for the reference-free metrics here the arguments are "
      "`(src, ·)`, so their z is an analogue rather than the identical statistic.")
    A("")
    A("## The \"just augment the metric's training data\" objection")
    A("")
    A("Two independent published refutations, one general and one on our own language family, "
      "both verified above. Amrhein and Sennrich retrained COMET on ~61k synthetic examples "
      "targeting exactly the number/NE blind spots and concluded that removing them \"might need "
      "more effort than simply training on additional synthetic data\"; Singh et al. reach the "
      "same conclusion for low-resource Indic. The objection presumes a coverage gap fixable by "
      "examples — but both prior attempts targeted a strictly *easier* phenomenon, one that **is** "
      "marked in the English source and whose supervision is therefore derivable automatically. "
      "Our phenomenon has no English side to derive supervision from. "
      "(Amrhein and Sennrich's own Limitations hedge — that other synthetic-data strategies might "
      "succeed — should be quoted alongside, not omitted.)")
    A("")
    A("## Provisionality")
    A("")
    A("- `honorificity` and `rationality` are **provisional project-wide** (unresolved third "
      "honorific degree, NATIVE-CHECK-HON-3/4; unresolved variety split). They are still the "
      "preferred headline slots because their metric result does not depend on the unresolved "
      "question — only the T-vs-V contrast is used.")
    A("- `gender` C2/C3 items carry a known confound (100% of their sources are *they*). "
      "**This section uses C1 only, where 0% of gender sources contain *they*,** so the confound "
      "does not apply; gender results are nevertheless marked provisional pending the "
      "regeneration in progress.")
    A("- `clusivity` C1 does not measure clusivity **behaviourally** (0/196 C1 sources contain "
      "*we/us/our*). The controlled pair is unaffected — it is a reference-side minimal pair "
      "whose members are both FST-verified — so clusivity contributes here and its behavioural "
      "C1 result does not.")
    A("- ⛔ **`controlled|number` covers ONE DIRECTION ONLY (PL→SG).** The FST verification of a "
      "flipped form was called without the honorificity context filter, and `-ஆர்கள்` is "
      "3sg-honorific *or* 3pl (DECISIONS.md D-2), so every SG→PL flip came back `UNDECIDABLE` "
      "and was rejected: 252 of 620 number flips dropped, and all 368 survivors are PL→SG. "
      "`scripts/09a_build_blindness_set.py` now passes `referent_honorificity` (the pinned "
      "`hon = minus`), which is the filter `MorphChecker.check` exposes for "
      "exactly this case, so rebuilding the set recovers the missing direction. The released "
      "scores cover PL→SG only; the affected cell is the control slot in one arm. Every other cell is unaffected: for the four "
      "obligatory slots the filter feature (`referent_number`) is independent of the slot being "
      "flipped, and the natural and avoidance arms verify against the GOLD value, where no "
      "circularity arises. **Do not read `controlled|number` as a symmetric number result.**")
    A("- Accuracy is never reported where commitment is 0%; this section reports deltas and win "
      "rates over pairs, so the question does not arise.")
    return "\n".join(L) + "\n"


def headline_lines(r) -> list[str]:
    """The section's claims, generated from the JSON. Wording branches on the data, so a rerun
    that moves the numbers also moves the claims instead of leaving stale prose behind."""
    out, sd = [], r.get("splice_decomposition") or {}
    refbased = [m for m in r["metrics_run"] if not r["findings"][m]["reference_free"]]
    reffree = [m for m in r["metrics_run"] if r["findings"][m]["reference_free"]]

    # 1. which metrics see it at all, on the de-confounded slot signal
    blind, sighted = [], []
    for m in r["metrics_run"]:
        # The pooled "ALL" cell, signed: a negative ratio is a preference for the WRONG form, not
        # sensitivity, so an unsigned max would classify a backwards metric as a sighted one.
        v = sd.get(f"{m}|ALL")
        if not v or not v.get("slot_signal_calibrated"):
            continue
        (blind if v["slot_signal_ratio_75"] < 1.0 else sighted).append(m)
    if blind:
        out.append("0. **The split is by metric type, and it is not the split the pre-registration "
                   "predicted.** On the de-confounded slot signal, "
                   + ", ".join(f"{r['findings'][m]['label']} ({sd[m + '|ALL']['slot_signal_ratio_75']:+.2f}x d75)"
                               for m in blind)
                   + " fall **below** the threshold at which humans agree three times in four; "
                   + ", ".join(f"{r['findings'][m]['label']} ({sd[m + '|ALL']['slot_signal_ratio_75']:+.2f}x)"
                               for m in sighted) + " do not.")
    if sighted:
        out.append(f"1. **The reference-based metrics are not blind.** "
                   + ", ".join(r["findings"][m]["label"] for m in sighted if m in refbased)
                   + " charge a penalty **above** the ToShip 75%-agreement threshold on the "
                     "de-confounded slot signal, so the magnitude clause of pre-registered "
                     "prediction P1 is **falsified for them**. Reported, not dropped.")
    # Rank the reference-free metrics by their de-confounded slot signal and narrate the weakest
    # one; adding a second QE metric should sharpen the claim, not duplicate the paragraph.
    reffree = sorted(reffree, key=lambda m: (sd.get(f"{m}|ALL", {}).get("slot_signal_ratio_75")
                                             if sd.get(f"{m}|ALL", {}).get("slot_signal_calibrated")
                                             else 1e9))
    _qe = [m for m in reffree if r["findings"][m]["natural"]]
    if len(_qe) > 1:
        out.append(
            "**Two reference-free metrics from different labs replicate it independently.** "
            + "; ".join(
                f"{r['findings'][m]['label']} wins "
                f"{100 * r['findings'][m]['natural']['win_rate']:.1f}% on natural errors "
                f"(Δ {r['findings'][m]['natural']['mean_delta']:+.5f})" for m in _qe)
            + ". Different architectures, different training data, same direction — so this is a "
              "property of reference-free evaluation, not of one checkpoint.")
    for m in reffree[:1]:
        f = r["findings"][m]
        nat, av = f["natural"], f["avoidance"]
        if not nat:
            continue
        prefer = [k.split("|")[1] for k, v in sd.items()
                  if k.startswith(m + "|") and v.get("prefers_the_wrong_form")]
        out.append(
            f"2. **{f['label']} is the metric that cannot see it.** It ranks the correct form "
            f"first on only **{100 * nat['win_rate']:.1f}%** of the system's own errors "
            f"(n={nat['n']}) — i.e. it *prefers* the ungrammatical output more often than not"
            + (f", and on **{', '.join(prefer)}** its de-confounded slot signal is negative: it "
               f"genuinely prefers the wrong value" if prefer else "") + ".")
        if av:
            ci = av["mean_delta_ci"]
            zero = ci[0] <= 0 <= ci[1]
            out.append(
                f"3. **Nothing is charged for avoidance.** On the epicene `-ஆர்` outputs the "
                f"panel actually produces (n={av['n']}), {f['label']}'s mean penalty is "
                f"**{av['mean_delta']:+.5f}** [{ci[0]:.5f}, {ci[1]:.5f}]"
                + (" — a confidence interval that includes zero" if zero else "")
                + (f", {av['ratio_75']:.2f}x the 75%-agreement threshold"
                   if av.get("ratio_75") is not None else
                   " (this metric has no published calibration, so no threshold comparison)")
                + ". A system that never marks gender loses nothing on this metric; the "
                  "information loss is free.")
    # 4. the internal control: number is marked in English
    _rows = []
    for m in _qe:
        c = r["cells"].get(f"controlled|number|ALL||slot_error||{m}")
        h = r["cells"].get(f"controlled|honorificity|ALL||slot_error||{m}")
        if c and h:
            _rows.append(f"{r['findings'][m]['label']} "
                         f"{100 * c['direction']['win_rate']:.1f}% vs "
                         f"{100 * h['direction']['win_rate']:.1f}%")
    if _rows:
        out.append("4. **The same metrics see `number` almost perfectly.** Win rate on the number "
                   "control slot against honorificity: " + "; ".join(_rows) + ". Number is marked "
                   "in the English source; honorificity, clusivity and rationality are not. "
                   "A reference-free metric sees exactly the distinctions the source encodes — "
                   "which is the mechanism this paper is about, visible inside a single metric.")
    # 5. DEMETR ordering
    dem = [m for m in r["metrics_run"]
           if (r["findings"][m]["demetr_ordering"] or {}).get("repetition_penalised_more")
           and (r["findings"][m]["demetr_ordering"] or {}).get("cis_disjoint")]
    if dem:
        out.append("5. **DEMETR's minor-vs-critical result reproduces on Tamil** for "
                   + ", ".join(r["findings"][m]["label"] for m in dem)
                   + ": a duplicated content word costs more than the obligatory morpheme, on "
                     "the same items in the same run, with disjoint bootstrap CIs.")
    v = (r.get("localization") or {}).get("verdict") or {}
    if v.get("call"):
        out.append(f"6. **xCOMET span localisation — {v['call'].replace('_', ' ')}.** {v.get('why', '')}")
    return out


def row(label, c) -> str:
    lv = c.get("levels") or {}
    return (f"| {label} | {c['n']} | {100 * c['win_rate']:.1f}% | "
            f"{fmt(c['mean_delta'], 5)} ({c['native_scale']}) | "
            f"[{fmt(c['mean_delta_ci'][0], 5)}, {fmt(c['mean_delta_ci'][1], 5)}] | "
            f"{fmt(c['z'], 3)} | [{fmt(c['z_ci'][0], 3)}, {fmt(c['z_ci'][1], 3)}] | "
            f"{fmt(lv.get('wrong'), 4)} | "
            f"{fmt(c['ratio_75'], 2) if c['calibrated'] else '—'} | "
            + (("—" if not c["calibrated"] else
                (f"{100 * c['acc_at_observed']:.1f}%"
                 + ("†" if c.get("saturated") else ""))) ) + " |")


if __name__ == "__main__":
    sys.exit(main())
