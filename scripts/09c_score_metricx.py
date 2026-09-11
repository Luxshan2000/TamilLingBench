#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MetricX-24-hybrid-XL and MetricX-23-Large. Runs in a separate venv for metricx.

Two models, two different jobs:

  metricx24xl     the metric we actually want to evaluate. It has NO entry in the Kocmi et al.
                  threshold table, so its delta is reported UNCALIBRATED and the gap is stated.
  metricx23large  the calibratable stand-in. It DOES have a ToShip key (`metricx-23-large`), so
                  it is the MetricX-family point that can be turned into a human-agreement
                  probability. It is not a substitute for 24 and its thresholds are never
                  borrowed for 24.

Both are 0-25 and LOWER IS BETTER; orientation is flipped once, centrally, in
`tamillingbench/metrics/blindness.py`, never here.

TWO UPSTREAM DEFECTS, both hit and both worked around here rather than papered over:

 1. The repo has no `setup.py`/`pyproject.toml` at HEAD (commit fc4978e, verified 2026-08-08), so
    `pip install git+...` fails with "does not appear to be a Python project". It is cloned and put
    on PYTHONPATH.
 2. `metricx24/predict.py` tokenizes with `padding=False` and then hands the dataset to
    `transformers.Trainer`, whose default collator cannot stack ragged tensors:
    `RuntimeError: stack expects each tensor to be equal size, but got [50] ... and [36]`.
    Rather than force `padding="max_length"` (which would pad every 6-word Tamil sentence to 1536
    tokens and waste ~40x the compute), this script runs the model directly with per-batch dynamic
    padding. The arithmetic is identical because the padded positions are masked out by
    `attention_mask`; the EOS-stripping and truncation are reproduced exactly from upstream.

 3. metricx23 and metricx24 build DIFFERENT input strings -- see `build_inputs`. This one is
    silent and produced a wrong-signed result before it was caught.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("MB_BASE",
                           Path(__file__).resolve().parent.parent / "outputs/metric_blindness"))
IN = BASE / "inputs"
RAW = BASE / "raw"
SRC = Path(os.environ.get("METRICX_SRC", "third_party/metricx"))   # a clone of google-research/metricx

MODELS = {
    # name              module      checkpoint                            max_len  qe
    "metricx24xl":     ("metricx24", "google/metricx-24-hybrid-xl-v2p6",  1536, False),
    "metricx24xlqe":   ("metricx24", "google/metricx-24-hybrid-xl-v2p6",  1536, True),
    "metricx23large":  ("metricx23", "google/metricx-23-large-v2p0",      1024, False),
}


def build_inputs(module: str, qe: bool) -> list[str]:
    """The two MetricX generations use DIFFERENT input strings and the difference is silent.

    Verified by reading `_make_input` in each module (metricx-src @ fc4978e):

        metricx24  qe : "source: {src} candidate: {hyp}"
        metricx24  ref: "source: {src} candidate: {hyp} reference: {ref}"
        metricx23  qe : "candidate: {hyp} source: {src}"
        metricx23  ref: "candidate: {hyp} reference: {ref}"          <- NO SOURCE, and hyp first

    Feeding MetricX-23 the MetricX-24 string produces plausible-looking numbers that are wrong:
    on the first pass it reported a large NEGATIVE slot signal (it appeared to prefer the
    ungrammatical form on 76-98% of pairs), which is what caught the bug. A metric wrapper that
    guesses the format is a metric wrapper that publishes an artefact.
    """
    src = (IN / "segments.src.txt").read_text().splitlines()
    mt = (IN / "segments.mt.txt").read_text().splitlines()
    ref = (IN / "segments.ref.txt").read_text().splitlines()
    if module == "metricx23":
        if qe:
            return [f"candidate: {h} source: {s}" for s, h in zip(src, mt)]
        return [f"candidate: {h} reference: {r}" for h, r in zip(mt, ref)]
    if qe:
        return [f"source: {s} candidate: {h}" for s, h in zip(src, mt)]
    return [f"source: {s} candidate: {h} reference: {r}" for s, h, r in zip(src, mt, ref)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["metricx24xl", "metricx23large"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    import torch
    import transformers

    in_hash = (IN / "inputs.sha256").read_text().strip()
    RAW.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False,
                                                     use_fast=True)

    for name in args.models:
        module, ckpt, maxlen, qe = MODELS[name]
        out = RAW / f"{name}.json"
        if out.exists() and json.loads(out.read_text()).get("input_sha256") == in_hash:
            print(f"{name}: up to date, skipping", flush=True)
            continue
        models = __import__(f"{module}.models", fromlist=["models"])
        print(f"=== {name} ({ckpt}) ===", flush=True)
        model = models.MT5ForRegression.from_pretrained(ckpt, torch_dtype="auto").to(device)
        model.eval()

        texts = build_inputs(module, qe)
        enc = tok(texts, max_length=maxlen, truncation=True, padding=False)["input_ids"]
        # Upstream drops the trailing EOS; reproduce that exactly.
        ids = [x[:-1] for x in enc]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))   # length-sorted batching
        scores = [0.0] * len(ids)
        pad = tok.pad_token_id or 0
        bs = args.batch_size
        with torch.no_grad():
            for start in range(0, len(order), bs):
                chunk = order[start:start + bs]
                L = max(len(ids[i]) for i in chunk)
                batch = torch.full((len(chunk), L), pad, dtype=torch.long)
                mask = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, i in enumerate(chunk):
                    batch[j, :len(ids[i])] = torch.tensor(ids[i])
                    mask[j, :len(ids[i])] = 1
                pred = model(input_ids=batch.to(device),
                             attention_mask=mask.to(device)).predictions
                for j, i in enumerate(chunk):
                    scores[i] = float(pred[j])
                if start % (bs * 100) == 0:
                    print(f"  {start}/{len(order)}", flush=True)
        out.write_text(json.dumps({
            "metric": name, "hf_model": ckpt, "module": module, "qe": qe,
            "scale": "0-25, LOWER IS BETTER", "input_sha256": in_hash,
            "n": len(scores), "batch_size": bs,
            "toship_key": "metricx-23-large" if name == "metricx23large" else None,
            "inference": "direct forward with per-batch dynamic padding; upstream predict.py "
                         "crashes on ragged collation (padding=False + default collator)",
            "input_format": build_inputs(module, qe)[0][:160],
            "scores": scores,
        }, ensure_ascii=False))
        print(f"{name}: {len(scores)} scores, mean {sum(scores) / len(scores):.4f} -> {out}",
              flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
