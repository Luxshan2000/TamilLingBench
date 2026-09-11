#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scoring pass — re-runnable, never destructive. RUNS LOCALLY (hfst/foma have no aarch64
wheel, so the morphology cannot move to the aarch64 GPU machine).

Reads  `<raw>/{model}/{slot}_{condition}_{prompt}.jsonl`   (never written to)
Writes `<scored>/{model}/{slot}_{condition}_{prompt}.jsonl` (freely overwritten)

Re-scoring is the whole point of the split: when NATIVE-CHECK-HON-3/4 settles which Tamil form
is the third honorific degree, or when the Indian/Sri Lankan rationality question settles, this
script re-runs in minutes and no GPU is touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from postprocess import clean as clean_output  # noqa: E402

from tamillingbench.eval.extract import (  # noqa: E402
    AVOIDANT_OUTCOMES, COMMITTED_OUTCOMES, SlotExtractor, context_filters, tamil_share)


def load_items(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            out[it["item_id"]] = it
    return out


def gold_surface_hit(hyp: str, item: dict) -> bool | None:
    """A DIAGNOSTIC string match, never a verdict.

    It exists for exactly one reason: the validated checker cannot license a DEFERENTIAL
    reading at all (ThamizhiMorph analyses தாங்கள் as `+pron+3pl+refl`, and the optative
    வருக/செய்க carries no honorificity), so the VV degree would otherwise be invisible rather
    than merely unverified. Reported as its own column, labelled as a string match, and never
    substituted for the morphological verdict.
    """
    tgts = item.get("gold_targets") or []
    surfaces = [t.get("surface_morpheme") for t in tgts if t.get("surface_morpheme")]
    words = [t["tamil"].rstrip(".!?") for t in tgts if t.get("tamil")]
    if not surfaces and not words:
        return None
    return any(s and s in hyp for s in surfaces) or any(w and w in hyp for w in words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--scored", default="outputs/scored")
    ap.add_argument("--items", default="data/benchmark/items.jsonl")
    ap.add_argument("--models", default="", help="comma-separated; default = all found")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    raw_root, scored_root = ROOT / args.raw, ROOT / args.scored
    items = load_items(ROOT / args.items)
    models = (args.models.split(",") if args.models
              else sorted(p.name for p in raw_root.iterdir() if p.is_dir()))

    ex = SlotExtractor()
    summary: dict[str, dict] = {}
    t0 = time.time()
    for model in models:
        mdir = raw_root / model
        if not mdir.is_dir():
            print(f"  ! no raw dir for {model}", flush=True)
            continue
        odir = scored_root / model
        odir.mkdir(parents=True, exist_ok=True)
        counts: Counter = Counter()
        n_files = 0
        for fp in sorted(mdir.glob("*.jsonl")):
            recs_out = []
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                item = items[r["item_id"]]
                rn, rh = context_filters(item)
                # Re-clean from the PRESERVED raw string rather than trusting the
                # generation-time clean. This is what makes a cleaning bug a re-scoring job
                # instead of a re-generation job.
                hyp, ops = clean_output(r["hypothesis_raw"])
                if not hyp:
                    hyp, ops = r["hypothesis"], list(r.get("clean_ops") or [])
                rescored_clean = hyp != r["hypothesis"]
                j = ex.judge(hyp, r["slot"], item.get("gold_value"),
                             referent_number=rn, referent_honorificity=rh,
                             third_degree=item.get("third_degree"),
                             slot_family=item.get("slot_family"))
                span = list(j.morpheme_char_span) if j.morpheme_char_span else None
                if span and j.surface_morpheme:
                    # the span must slice the hypothesis exactly.
                    if hyp[span[0]:span[1]] != j.surface_morpheme:
                        # the morpheme is a SUFFIX of the token; narrow the span to it
                        tokstr = hyp[span[0]:span[1]]
                        if tokstr.endswith(j.surface_morpheme):
                            span = [span[1] - len(j.surface_morpheme), span[1]]
                        else:
                            span = None
                has_gold = item.get("gold_value") is not None
                out = {
                    "item_id": r["item_id"], "set_id": r["set_id"],
                    "template_id": r["template_id"], "slot": r["slot"],
                    "slot_family": r["slot_family"], "condition": r["condition"],
                    "k": r["k"], "chance_rate": r["chance_rate"],
                    "stratum": r["stratum"], "register": r["register"],
                    "system_id": r["system_id"], "prompt_id": r["prompt_id"],
                    "revision": r["revision"], "decoding_id": r["decoding_id"],
                    "gold_value": item.get("gold_value"),
                    # --- judgment ---
                    "outcome": j.outcome,
                    "emitted_value": j.emitted_value,
                    # ⛔ None, not False: C3 has no gold, so "correct" is undefined. A False
                    # here would be silently counted as an error by any aggregate.
                    "correct": (j.outcome == "CORRECT") if has_gold else None,
                    "has_gold": has_gold,
                    "universal": j.universal, "existential": j.existential,
                    "undecidable": j.undecidable,
                    "committed": j.outcome in COMMITTED_OUTCOMES,
                    "avoidant": j.outcome in AVOIDANT_OUTCOMES,
                    "n_candidate_readings": j.n_candidate_readings,
                    "n_surviving": j.n_surviving, "n_matching": j.n_matching,
                    "all_candidate_values": j.all_candidate_values,
                    "neutralising_value": j.neutralising_value,
                    "surface_morpheme": j.surface_morpheme,
                    "morpheme_char_span": span,
                    "decision_source": j.decision_source,
                    "why": j.why,
                    "analysable_rate": round(j.analysable_rate, 4),
                    "tamil_char_share": round(j.tamil_char_share, 4),
                    "unparsed_reason": j.unparsed_reason,
                    "provisional": j.provisional,
                    "provisional_reason": j.provisional_reason,
                    # --- diagnostics, never verdicts ---
                    "gold_surface_hit": gold_surface_hit(hyp, item),
                    "overt_environment": (item.get("avoidance") or {}).get("overt_environment"),
                    "third_degree_form": (item.get("third_degree") or {}).get("form"),
                    "variety_assumed": item.get("variety_assumed"),
                    "hypothesis": hyp,
                    "hypothesis_at_generation": r["hypothesis"],
                    "clean_ops": ops,
                    "recleaned_at_scoring": rescored_clean,
                    "tamil_char_share_clean": round(tamil_share(hyp), 4),
                    "n_out_tokens": r["n_out_tokens"],
                    "cleaned": bool(ops),
                }
                recs_out.append(out)
                counts[(r["slot"], r["condition"], r["prompt_id"], j.outcome)] += 1
            (odir / fp.name).write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs_out),
                encoding="utf-8")
            n_files += 1
            if not args.quiet:
                print(f"  {model}/{fp.name}: {len(recs_out)}", flush=True)
        summary[model] = {
            "n_files": n_files,
            "n_records": sum(counts.values()),
            "outcomes": {f"{s}|{c}|{p}|{o}": n for (s, c, p, o), n in sorted(counts.items())},
        }
        agg = Counter()
        for (s, c, p, o), n in counts.items():
            agg[o] += n
        print(f"{model}: {sum(counts.values())} records  {dict(agg)}", flush=True)

    ex.close()
    meta = {"scored_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.time() - t0, 1),
            "raw_root": str(raw_root), "scored_root": str(scored_root),
            "checker": "tamillingbench.morph.MorphChecker (existing, validated; not rebuilt)",
            "policy": "DECISIONS.md D-2 — universal after context filtering (headline); "
                      "existential and undecidable reported separately",
            "models": summary}
    scored_root.mkdir(parents=True, exist_ok=True)
    (scored_root / "_scoring_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
