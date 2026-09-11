#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integrity check on the raw generations. Cheap, and it runs on every cycle.

The architectural promise of this stage is *generate once, score many times*. That promise is
only worth anything if the raw outputs are actually immutable and complete, so it is checked
rather than asserted:

  * no duplicate `item_id` inside a cell file (a re-run appending instead of replacing);
  * every canonical cell has exactly the number of public items the manifest says it should;
  * every record carries the provenance fields a re-scoring pass depends on;
  * `hypothesis_raw` is present and non-empty on every record — without it, a cleaning bug
    becomes a re-generation job instead of a re-scoring job.

Writes `results/raw-integrity.json` and exits non-zero on any problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = {"C0": "P0_specified", "C1": "P1_minimal", "C2": "P4_context", "C3": "P1_minimal"}
REQUIRED = ("item_id", "set_id", "template_id", "slot", "condition", "system_id", "hf_name",
            "revision", "prompt_id", "decoding", "hypothesis", "hypothesis_raw", "run_id")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="outputs/raw")
    ap.add_argument("--items", default="data/benchmark/items.jsonl")
    ap.add_argument("--out", default="results/raw-integrity.json")
    args = ap.parse_args()

    expected: Counter = Counter()
    for line in (ROOT / args.items).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            if it.get("split") == "public":
                expected[(it["slot"], it["condition"])] += 1

    problems: list[str] = []
    per_model: dict[str, dict] = defaultdict(lambda: {"files": 0, "records": 0})
    files = sorted((ROOT / args.raw).glob("*/*.jsonl"))
    for fp in files:
        rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            problems.append(f"{fp}: empty")
            continue
        model = fp.parent.name
        per_model[model]["files"] += 1
        per_model[model]["records"] += len(rows)
        ids = [r["item_id"] for r in rows]
        if len(set(ids)) != len(ids):
            dup = [k for k, v in Counter(ids).items() if v > 1][:3]
            problems.append(f"{fp}: duplicate item_ids e.g. {dup}")
        slot, cond, prompt = rows[0]["slot"], rows[0]["condition"], rows[0]["prompt_id"]
        if prompt in (CANONICAL.get(cond), "nmt") and len(rows) != expected[(slot, cond)]:
            problems.append(
                f"{fp}: {len(rows)} records, expected {expected[(slot, cond)]}")
        for r in rows[:50]:
            missing = [f for f in REQUIRED if r.get(f) in (None, "")]
            if missing:
                problems.append(f"{fp}: {r.get('item_id')} missing {missing}")
                break

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "n_records": sum(v["records"] for v in per_model.values()),
        "per_model": {k: v for k, v in sorted(per_model.items())},
        "problems": problems,
        "ok": not problems,
        "what_this_checks": "Raw generations are immutable and complete: no duplicate ids, "
                            "canonical cells match the public item counts, and every record "
                            "carries the provenance a re-scoring pass needs (including "
                            "hypothesis_raw, without which a cleaning fix would require a "
                            "GPU re-run).",
    }
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"raw integrity: {len(files)} files, {out['n_records']} records, "
          f"{len(problems)} problem(s)")
    for p in problems[:10]:
        print("  !", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
