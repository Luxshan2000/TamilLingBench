#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The adversary test: does xCOMET localise the obligatory morpheme?

xCOMET is marketed as identifying "localized critical errors and hallucinations". BLEU failing on
morphology surprises nobody; xCOMET failing is a falsifiable claim about a specific published
capability. This script tests it directly and, per the pre-registration, IT DOES NOT REPORT A
PHENOMENON-SPECIFIC FAILURE WITHOUT THE POSITIVE CONTROL: the identical analysis is run on the
numeral-swap and named-entity-swap variants built on the same Tamil sentences. If xCOMET misses
those too, the finding is declared inconclusive for Tamil rather than claimed.

The gold error span is computed as the differing grapheme-cluster region between `good` and the
perturbed variant, on NORMALISED text -- not read off the scorer's `morpheme_char_span`, which is an
offset into the raw system output. Every predicted span is round-trip checked against
`text[start:end]` before it is used, because a silent offset drift looks exactly like a miss.
The check is deliberately two-tier: xCOMET's `text` field is DETOKENISED
(`tokenizer.decode(span["tokens"])`), so strict equality fails on correctly-aligned spans whenever
SentencePiece normalises whitespace. Exact and relaxed (containment) matches are counted
separately, and only a genuine non-overlap is discarded.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE = Path(os.environ.get("MB_BASE", ROOT / "outputs/metric_blindness"))

from tamillingbench.metrics.surgery import letters                   # noqa: E402

#: Which variants get the localisation analysis, and what each one is for.
TARGETS = {
    "good":     "BASELINE — the FST-verified CORRECT sentence, which contains no injected error. "
                "Whatever xCOMET flags here is its false-positive floor, and a hit rate on `bad` "
                "is only meaningful above it.",
    "bad":      "the obligatory-slot error under test",
    "avoided":  "the epicene avoidance form (information dropped, not wrong)",
    "ctrl_num": "POSITIVE CONTROL — numeral swap (DEMETR critical / ACES hallucination-number)",
    "ctrl_ne":  "POSITIVE CONTROL — named-entity swap (ACES hallucination-named-entity)",
    "ctrl_rep": "NEGATIVE CONTROL — word duplication (DEMETR minor)",
}

#: `ctrl_del` is deliberately absent. A deletion leaves a ZERO-WIDTH region in the hypothesis
#: (measured: 3500/3500 give start == end), and a span-labelling model cannot tag characters that
#: are not there. It is a valid SCORE control and is reported as one; it is not a span control.
#: A slot flip that removes a suffix has the same property and lands in `skipped_no_gold_span`,
#: which is reported rather than folded into the miss rate -- counting an untaggable item as a
#: miss would manufacture the result this section is trying to test.
EXCLUDED_FROM_LOCALISATION = {"ctrl_del": "deletion leaves a zero-width span in the hypothesis"}


def diff_span(good: str, bad: str) -> tuple[int, int] | None:
    """Character span in `bad` covering the single differing grapheme-cluster region."""
    la, lb = letters(good), letters(bad)
    i = 0
    while i < len(la) and i < len(lb) and la[i] == lb[i]:
        i += 1
    j = 0
    while (j < len(la) - i) and (j < len(lb) - i) and la[-1 - j] == lb[-1 - j]:
        j += 1
    start = sum(len(x) for x in lb[:i])
    end = sum(len(x) for x in lb[:len(lb) - j])
    if end < start:
        return None
    return (start, end)


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def main() -> int:
    raw = BASE / "raw/xcometxl.json"
    if not raw.exists():
        print("xcometxl.json not present — xCOMET has not been scored. "
              "Reporting the localisation section as NOT RUN.", file=sys.stderr)
        (BASE / "localization.json").write_text(json.dumps(
            {"status": "not_run", "reason": "outputs/metric_blindness/raw/xcometxl.json missing"},
            indent=1))
        return 1
    d = json.loads(raw.read_text())
    spans_by_line = d.get("error_spans")
    if spans_by_line is None:
        print("xCOMET output carries no error_spans field — the installed unbabel-comet did not "
              "return metadata.error_spans. Localisation cannot be run.", file=sys.stderr)
        (BASE / "localization.json").write_text(json.dumps(
            {"status": "not_run", "reason": "no error_spans in xcometxl.json",
             "observed_keys": sorted(d)}, indent=1))
        return 1

    import csv
    with open(BASE / "inputs/index.tsv") as fh:
        index = list(csv.DictReader(fh, delimiter="\t"))
    line_of = {(r["item_id"], r["system_id"], r["prompt_id"], r["arm"], r["variant"]): int(r["line"])
               for r in index}
    mt = (BASE / "inputs/segments.mt.txt").read_text().splitlines()
    recs = [json.loads(l) for l in open(BASE / "scoring_set.jsonl")]

    # Record the observed span schema rather than assuming field names.
    schema = Counter()
    for segspans in spans_by_line:
        for s in (segspans or []):
            schema[tuple(sorted(s))] += 1

    per_variant: dict[str, dict] = {}
    per_slot: dict[str, dict] = defaultdict(lambda: defaultdict(Counter))
    roundtrip = Counter()

    for var in TARGETS:
        agg = Counter()
        sev = Counter()
        prec_num = prec_den = rec_den = 0
        for r in recs:
            if var not in r["variants"]:
                continue
            key = (r["item_id"], r["system_id"], r["prompt_id"], r["arm"], var)
            if key not in line_of:
                continue
            line = line_of[key]
            text = mt[line]
            if var == "good":
                # No injected error: the gold span is the region that WOULD have carried it, so
                # a "hit" here is a false positive on an unperturbed sentence.
                gold = diff_span(r["variants"]["good"], r["variants"].get("bad", ""))
                gold = (min(gold[0], len(text)), min(gold[1], len(text))) if gold else None
            else:
                gold = diff_span(r["variants"]["good"], r["variants"][var])
            if gold is None or gold[1] <= gold[0]:
                agg["skip_no_gold_span"] += 1
                continue
            pred = []
            for s in (spans_by_line[line] or []):
                a, b = int(s.get("start", -1)), int(s.get("end", -1))
                if not (0 <= a <= b <= len(text)):
                    roundtrip["out_of_range"] += 1
                    continue
                # The `text` field is the DETOKENISED span, not a raw slice: xCOMET builds it
                # with `tokenizer.decode(span["tokens"])`, so SentencePiece markers and leading
                # spaces make strict equality fail on correctly-aligned spans. Strict equality is
                # still counted (it is the signal that would catch a genuine offset drift); the
                # relaxed containment check is what gates use.
                slice_txt, span_txt = text[a:b].strip(), str(s.get("text", "")).strip()
                if not span_txt:
                    roundtrip["no_text_field"] += 1
                elif slice_txt == span_txt:
                    roundtrip["ok_exact"] += 1
                elif span_txt in slice_txt or slice_txt in span_txt:
                    roundtrip["ok_relaxed"] += 1
                else:
                    roundtrip["text_mismatch"] += 1
                    continue
                pred.append(((a, b), s.get("severity") or s.get("confidence")))
            agg["n"] += 1
            if pred:
                agg["flagged_any"] += 1
            hits = [(p, sv) for p, sv in pred if overlap(p, gold) >= 1]
            hits_iou = [p for p, _ in pred if iou(p, gold) >= 0.5]
            if hits:
                agg["hit_overlap"] += 1
                for _, sv in hits:
                    sev[str(sv)] += 1
            if hits_iou:
                agg["hit_iou50"] += 1
            if pred and not hits:
                agg["distractor_only"] += 1          # flagged something, missed the actual error
            prec_num += len(hits)
            prec_den += len(pred)
            rec_den += 1
            # Span WIDTH is the quantity that decides whether "overlap" means anything: a model
            # that flags the whole sentence overlaps every error by construction.
            if pred:
                cov = sum(b2 - a2 for (a2, b2), _ in pred)
                agg["cov_num"] += min(cov, len(text))
                agg["cov_den"] += len(text)
                agg["nspans"] += len(pred)
                agg["gold_len"] += gold[1] - gold[0]
            per_slot[var][r["slot"]]["n"] += 1
            per_slot[var][r["slot"]]["flagged"] += bool(pred)
            per_slot[var][r["slot"]]["hit"] += bool(hits)

        n = agg["n"] or 1
        per_variant[var] = {
            "description": TARGETS[var],
            "n": agg["n"],
            "detection_at_item": agg["flagged_any"] / n,
            "hit_rate_overlap": agg["hit_overlap"] / n,
            "hit_rate_iou50": agg["hit_iou50"] / n,
            "hit_rate_given_flagged": (agg["hit_overlap"] / agg["flagged_any"]
                                       if agg["flagged_any"] else None),
            "distractor_rate": agg["distractor_only"] / n,
            "span_precision": prec_num / prec_den if prec_den else None,
            "span_recall": agg["hit_overlap"] / rec_den if rec_den else None,
            "severity_when_hit": dict(sev),
            # xCOMET's own MQM labels. A `minor` on an honorific violation toward an elder is a
            # substantive misclassification, not a rounding error, so the share is reported
            # rather than left inside a raw count.
            "critical_share_when_hit": (sev["critical"] / sum(sev.values())) if sev else None,
            "major_or_critical_share_when_hit": (
                (sev["critical"] + sev["major"]) / sum(sev.values())) if sev else None,
            "mean_flagged_share_of_sentence": (agg["cov_num"] / agg["cov_den"]
                                               if agg["cov_den"] else None),
            "mean_spans_per_flagged_segment": (agg["nspans"] / agg["flagged_any"]
                                               if agg["flagged_any"] else None),
            "mean_gold_span_chars": (agg["gold_len"] / agg["flagged_any"]
                                     if agg["flagged_any"] else None),
            "skipped_no_gold_span": agg["skip_no_gold_span"],
        }
        p, rc = per_variant[var]["span_precision"], per_variant[var]["span_recall"]
        per_variant[var]["span_f1"] = (2 * p * rc / (p + rc)) if p and rc else None

    verdict = interpret(per_variant)
    out = {
        "status": "ok",
        "model": d.get("hf_model"),
        "span_schema_observed": {"|".join(k): v for k, v in schema.most_common()},
        "roundtrip": dict(roundtrip),
        "per_variant": per_variant,
        "per_slot": {v: {s: dict(c) for s, c in d2.items()} for v, d2 in per_slot.items()},
        "verdict": verdict,
    }
    (BASE / "localization.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"verdict": verdict,
                      "per_variant": {k: {kk: v[kk] for kk in
                                          ("n", "detection_at_item", "hit_rate_overlap",
                                           "hit_rate_iou50", "distractor_rate")}
                                      for k, v in per_variant.items()}},
                     ensure_ascii=False, indent=1))
    return 0


def interpret(pv: dict) -> dict:
    """The pre-registered decision rule, applied mechanically rather than by eye."""
    slot = pv.get("bad", {})
    base = pv.get("good", {})
    # A hit rate is only evidence of localisation if it beats what the model flags on a sentence
    # with NO injected error at all. If it does not, the model is flagging indiscriminately and
    # the whole localisation comparison is uninformative -- a distinct failure from "misses it".
    if base.get("n") and slot.get("n") and base["hit_rate_overlap"] >= slot["hit_rate_overlap"]:
        return {"call": "indiscriminate_flagging",
                "why": f"xCOMET flags the gold region on {base['hit_rate_overlap']:.1%} of "
                       f"UNPERTURBED correct sentences, at least as often as on the "
                       f"{slot['hit_rate_overlap']:.1%} of sentences that actually carry the "
                       f"error. Overlap-based localisation carries no information here; report "
                       f"the false-positive floor and the flagged share of the sentence "
                       f"({base.get('mean_flagged_share_of_sentence')}), not a hit rate.",
                "false_positive_hit_rate": base["hit_rate_overlap"],
                "slot_hit_rate": slot["hit_rate_overlap"]}
    ctrls = [pv.get(k, {}) for k in ("ctrl_num", "ctrl_ne")]
    ctrls = [c for c in ctrls if c.get("n")]
    if not slot.get("n"):
        return {"call": "not_evaluable", "why": "no scored slot-error variants"}
    if not ctrls:
        return {"call": "inconclusive",
                "why": "no positive control was scorable; per the pre-registration a "
                       "phenomenon-specific failure may not be claimed without one"}
    ctrl_hit = max(c["hit_rate_overlap"] for c in ctrls)
    slot_hit = slot["hit_rate_overlap"]
    out = {"control_hit_rate": ctrl_hit, "slot_hit_rate": slot_hit,
           "false_positive_hit_rate_on_correct": base.get("hit_rate_overlap"),
           "false_positive_flag_rate_on_correct": base.get("detection_at_item"),
           "slot_iou50": slot.get("hit_rate_iou50"),
           "control_iou50": max((c.get("hit_rate_iou50") or 0) for c in ctrls),
           "slot_flagged_share_of_sentence": slot.get("mean_flagged_share_of_sentence"),
           "slot_gold_span_chars": slot.get("mean_gold_span_chars"),
           "slot_critical_share": slot.get("critical_share_when_hit"),
           "control_critical_share": max((c.get("critical_share_when_hit") or 0) for c in ctrls)}
    if ctrl_hit < 0.20:
        return {**out, "call": "inconclusive_for_tamil",
                "why": f"xCOMET localises the positive controls at only {ctrl_hit:.1%}; it is not "
                       f"demonstrated to localise ANY error type in Tamil, so its "
                       f"{slot_hit:.1%} on the obligatory morpheme is not phenomenon-specific"}
    # Detection is not localisation. If the flagged region covers most of the sentence, an
    # overlap "hit" is close to automatic and the informative comparisons are IoU and severity.
    wide = (slot.get("mean_flagged_share_of_sentence") or 0) > 0.5
    if wide:
        sc, cc = out["slot_critical_share"] or 0, out["control_critical_share"] or 0
        return {**out, "call": "detects_but_does_not_localise",
                "why": (f"xCOMET flags the morpheme's region on {slot_hit:.1%} of items, but its "
                        f"spans cover {100 * (slot.get('mean_flagged_share_of_sentence') or 0):.0f}% "
                        f"of the sentence against a gold error span of "
                        f"{slot.get('mean_gold_span_chars'):.1f} characters, so IoU>=0.5 is only "
                        f"{slot.get('hit_rate_iou50'):.1%}. It also flags "
                        f"{base.get('detection_at_item', 0):.1%} of UNPERTURBED correct sentences. "
                        f"Detection is real; localisation in the advertised sense is not "
                        f"demonstrated. Severity is where the phenomenon shows: xCOMET calls the "
                        f"obligatory-slot violation `critical` {sc:.1%} of the time against "
                        f"{cc:.1%} for the numeral/named-entity controls on the same sentences.")}
    if slot_hit < ctrl_hit:
        return {**out, "call": "phenomenon_specific_failure",
                "why": f"xCOMET localises numeral/NE errors at {ctrl_hit:.1%} on the same "
                       f"sentences but the obligatory morpheme at only {slot_hit:.1%}"}
    return {**out, "call": "no_failure",
            "why": f"xCOMET localises the obligatory morpheme at {slot_hit:.1%}, at least as "
                   f"often as the positive controls ({ctrl_hit:.1%}). The prediction is falsified."}


if __name__ == "__main__":
    sys.exit(main())
