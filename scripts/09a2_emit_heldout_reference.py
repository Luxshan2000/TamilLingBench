#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The held-out reference configuration.

WHY THIS EXISTS. In the main configuration the `good` variant IS the reference, because that is
the DEMETR/ACES construction (perturb a reference to make the hypothesis). It has one property that
must not go unexamined: a reference-based metric scoring `good` is scoring an exact copy of its own
reference, which sits at the metric's ceiling. The measured delta is therefore an UPPER BOUND on
sensitivity -- the most favourable case the metric will ever see -- and it is not the situation any
deployed evaluation is in.

This configuration removes the identity. For each controlled record it uses, as the reference, a
DIFFERENT system's independently-produced output for the SAME item that the FST also verified as
carrying the gold slot value. Both `good` and `bad` are then genuine hypotheses that differ from
the reference in lexical choice and word order, and only `good` agrees with it on the obligatory
morpheme.

If a reference-based metric's penalty survives here, the metric really can see the distinction. If
it collapses, the main configuration's number was an artefact of the exact-match ceiling. Either
answer is reportable; not asking is not.

Reference-free metrics (CometKiwi, MetricX-QE) are unaffected by construction and are not re-run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs/metric_blindness"
OUT = BASE / "inputs_heldout"


def main() -> int:
    recs = [json.loads(l) for l in open(BASE / "scoring_set.jsonl")]
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r["arm"] == "controlled":
            by_item[r["item_id"]].append(r)

    rows, triples, uniq = [], [], {}
    n_no_alt = 0
    for r in recs:
        if r["arm"] != "controlled":
            continue
        good = r["variants"]["good"]
        alts = [o["variants"]["good"] for o in by_item[r["item_id"]]
                if o["system_id"] != r["system_id"] and o["variants"]["good"] != good]
        if not alts:
            n_no_alt += 1
            continue
        # Deterministic pick: the lexicographically first alternative, so a rerun is identical.
        ref = sorted(alts)[0]
        for var, mt in r["variants"].items():
            key = (r["source"], mt, ref)
            if key not in uniq:
                uniq[key] = len(uniq)
                triples.append(key)
            rows.append((uniq[key], r, var))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "segments.src.txt", "w") as fs, \
         open(OUT / "segments.mt.txt", "w") as fm, \
         open(OUT / "segments.ref.txt", "w") as fr, \
         open(OUT / "segments.metricx.jsonl", "w") as fx:
        for src, mt, ref in triples:
            fs.write(src.replace("\n", " ") + "\n")
            fm.write(mt.replace("\n", " ") + "\n")
            fr.write(ref.replace("\n", " ") + "\n")
            fx.write(json.dumps({"source": src, "hypothesis": mt, "reference": ref},
                                ensure_ascii=False) + "\n")
    with open(OUT / "index.tsv", "w") as fi:
        fi.write("line\titem_id\tsystem_id\tprompt_id\tarm\tslot\tvariant\n")
        for line, r, var in rows:
            fi.write(f"{line}\t{r['item_id']}\t{r['system_id']}\t{r['prompt_id']}\t"
                     f"heldout\t{r['slot']}\t{var}\n")
    digest = hashlib.sha256((OUT / "segments.mt.txt").read_bytes()
                            + (OUT / "segments.ref.txt").read_bytes()).hexdigest()
    (OUT / "inputs.sha256").write_text(digest + "\n")
    print(f"held-out reference set: {len(triples)} unique triples, {len(rows)} (record, variant) "
          f"pairs; {n_no_alt} controlled records had no independent alternative reference "
          f"and are excluded\nsha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
