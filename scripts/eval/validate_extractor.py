#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate the scoring extractor against the benchmark's OWN gold and contrast targets.

This is the check that licenses every number in `results/eval-results.*`. Two directions, both
required, because either alone is easy to pass by cheating:

  * **recall** — feeding an item's `gold_targets[].tamil` must yield `CORRECT`;
  * **precision** — feeding its `contrast_targets[].tamil` (the minimally different WRONG form)
    must NOT yield `CORRECT`.

A locus-selection bug shows up as a recall failure; an over-permissive matcher shows up as a
precision failure. Writes `results/extractor-validation.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tamillingbench.eval.extract import SlotExtractor, context_filters  # noqa: E402


def ablation(raw_root: Path, items: dict) -> dict:
    """Measure what the two SCORING-LAYER repairs actually change on real model output.

    Both are reported rather than assumed, because a repair that quietly moves the headline
    number is indistinguishable from a thumb on the scale unless its size is published.
    """
    from tamillingbench.eval.extract import SlotExtractor as SE
    canon = {"C0": "P0_specified", "C1": "P1_minimal", "C2": "P4_context",
             "C3": "P1_minimal"}
    recs = []
    if raw_root.is_dir():
        for mdir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            for fp in sorted(mdir.glob("*.jsonl")):
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["prompt_id"] in (canon.get(r["condition"]), "nmt"):
                        recs.append(r)
    if not recs:
        return {"n": 0, "note": "no raw outputs available"}
    out = {}
    for name, kw in (("with_repairs", {"repair_anar": True, "repair_sandhi": True}),
                     ("without_anar_repair", {"repair_anar": False, "repair_sandhi": True}),
                     ("without_sandhi_repair", {"repair_anar": True, "repair_sandhi": False}),
                     ("without_any_repair", {"repair_anar": False, "repair_sandhi": False})):
        e = SE(**kw)
        c: Counter = Counter()
        for r in recs:
            it = items[r["item_id"]]
            rn, rh = context_filters(it)
            j = e.judge(r["hypothesis"], r["slot"], it.get("gold_value"),
                        referent_number=rn, referent_honorificity=rh)
            c[(r["slot"], j.outcome)] += 1
        e.close()
        out[name] = {f"{s}|{o}": n for (s, o), n in sorted(c.items())}
    deltas = {}
    for arm in ("without_anar_repair", "without_sandhi_repair", "without_any_repair"):
        d = {}
        for k in set(out["with_repairs"]) | set(out[arm]):
            v = out["with_repairs"].get(k, 0) - out[arm].get(k, 0)
            if v:
                d[k] = v
        deltas[f"vs_{arm}"] = d
    return {"n_records": len(recs), **out, "deltas": deltas,
            "delta_with_minus_without": deltas["vs_without_any_repair"],
            "note": "Two SCORING-layer corrections of documented ThamizhiMorph defects: "
                    "(a) -அனர் is tagged 3sge (singular) though it is the unambiguously "
                    "plural rational form; (b) a word-final sandhi consonant "
                    "(எங்களைப் < எங்களை + ப்) blocks the pronoun analysis and returns a "
                    "spurious verb reading instead. The validated checker is NOT modified — "
                    "SlotExtractor(repair_anar=False, repair_sandhi=False) reproduces the "
                    "unrepaired numbers, and each arm is reported separately above."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--items", default="data/benchmark/items.jsonl")
    ap.add_argument("--out", default="results/extractor-validation.json")
    args = ap.parse_args()

    ex = SlotExtractor()
    recall: Counter = Counter()
    prec: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    for line in (ROOT / args.items).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if not it.get("gold_value") or not it.get("gold_targets"):
            continue
        rn, rh = context_filters(it)
        j = ex.judge(it["gold_targets"][0]["tamil"], it["slot"], it["gold_value"],
                     referent_number=rn, referent_honorificity=rh)
        recall[(it["slot"], it["gold_value"], j.outcome)] += 1
        if j.outcome != "CORRECT" and len(examples[f"recall:{it['slot']}:{j.outcome}"]) < 3:
            examples[f"recall:{it['slot']}:{j.outcome}"].append(
                {"item_id": it["item_id"], "gold_value": it["gold_value"],
                 "target": it["gold_targets"][0]["tamil"], "why": j.why[:220]})
        if it.get("contrast_targets"):
            j2 = ex.judge(it["contrast_targets"][0]["tamil"], it["slot"], it["gold_value"],
                          referent_number=rn, referent_honorificity=rh)
            prec[(it["slot"], j2.outcome)] += 1
            if j2.outcome == "CORRECT" and len(examples[f"precision:{it['slot']}"]) < 3:
                examples[f"precision:{it['slot']}"].append(
                    {"item_id": it["item_id"],
                     "contrast": it["contrast_targets"][0]["tamil"], "why": j2.why[:220]})
    ex.close()

    by_slot: dict[str, dict] = {}
    for slot in ("clusivity", "gender", "honorificity", "number", "rationality"):
        r = {o: n for (s, g, o), n in recall.items() if s == slot}
        rr: Counter = Counter()
        for (s, g, o), n in recall.items():
            if s == slot:
                rr[o] += n
        pp = {o: n for (s, o), n in prec.items() if s == slot}
        n_r, n_p = sum(rr.values()), sum(pp.values())
        by_slot[slot] = {
            "n_gold_targets": n_r,
            "recall_correct": rr["CORRECT"],
            "recall_rate": round(rr["CORRECT"] / n_r, 4) if n_r else None,
            "recall_outcomes": dict(rr),
            "recall_by_gold_value": {f"{g}->{o}": n for (s, g, o), n in sorted(recall.items())
                                     if s == slot},
            "n_contrast_targets": n_p,
            "contrast_false_positives": pp.get("CORRECT", 0),
            "contrast_fp_rate": round(pp.get("CORRECT", 0) / n_p, 4) if n_p else None,
            "contrast_outcomes": pp,
        }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "what_this_is": "Extractor validated against the benchmark's own gold and contrast "
                        "targets. Recall = gold target scores CORRECT. Precision = contrast "
                        "target does NOT score CORRECT.",
        "by_slot": by_slot,
        "totals": {
            "n_gold": sum(v["n_gold_targets"] for v in by_slot.values()),
            "n_gold_correct": sum(v["recall_correct"] for v in by_slot.values()),
            "n_contrast": sum(v["n_contrast_targets"] for v in by_slot.values()),
            "n_contrast_false_positive": sum(v["contrast_false_positives"]
                                             for v in by_slot.values()),
        },
        "known_structural_failure": {
            "slot": "honorificity", "gold_value": "VV",
            "note": "Every VV gold target fails, and this is not a bug in the extractor. "
                    "ThamizhiMorph licenses no DEFERENTIAL reading (தாங்கள் is "
                    "+pron+3pl+refl; the optative வருக/செய்க carries no honorificity tag). "
                    "The benchmark's own generator recorded the same failure: 106/106 VV "
                    "items have checker_verified=false. This is the concrete form of the "
                    "open NATIVE-CHECK-HON-3/4 question.",
        },
        "failure_examples": dict(examples),
        "scoring_repair_ablation": ablation(
            ROOT / args.raw,
            {json.loads(l)["item_id"]: json.loads(l)
             for l in (ROOT / args.items).read_text(
                 encoding="utf-8").splitlines() if l.strip()}),
    }
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["totals"].items()}, indent=1))
    for slot, v in by_slot.items():
        print(f"  {slot:14s} recall {v['recall_correct']}/{v['n_gold_targets']} "
              f"({v['recall_rate']})  contrast-FP {v['contrast_false_positives']}"
              f"/{v['n_contrast_targets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
