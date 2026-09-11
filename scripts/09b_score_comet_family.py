#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the COMET family. Runs on a GPU machine, in a separate venv for unbabel-comet.

Idempotent and resumable: a model whose output file already exists AND whose recorded input
hash matches the current one is skipped.

xCOMET is run through the PYTHON API rather than `comet-score --to_json`, because the whole of
Step 7 needs `metadata.error_spans`, and the CLI's JSON is flattened differently from what
`UnifiedMetric.predict` returns. The span payload is written out verbatim, unparsed, so the
localisation analysis can be re-run without another 25k forward passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import os

# On a GPU machine the tree may live outside the repo, so
# the base is overridable (MB_BASE). Defaults to the repo layout.
BASE = Path(os.environ.get("MB_BASE",
                           Path(__file__).resolve().parent.parent / "outputs/metric_blindness"))
IN = BASE / "inputs"
RAW = BASE / "raw"

MODELS = {
    "comet22":    dict(hf="Unbabel/wmt22-comet-da",       ref=True,  batch=32),
    "cometkiwi22": dict(hf="Unbabel/wmt22-cometkiwi-da",  ref=False, batch=32),
    "xcometxl":   dict(hf="Unbabel/XCOMET-XL",            ref=True,  batch=8),
}


def load_inputs():
    src = (IN / "segments.src.txt").read_text().splitlines()
    mt = (IN / "segments.mt.txt").read_text().splitlines()
    ref = (IN / "segments.ref.txt").read_text().splitlines()
    assert len(src) == len(mt) == len(ref), (len(src), len(mt), len(ref))
    return src, mt, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    from comet import download_model, load_from_checkpoint
    import torch

    src, mt, ref = load_inputs()
    in_hash = (IN / "inputs.sha256").read_text().strip()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"{len(src)} segments; input hash {in_hash[:16]}", flush=True)

    for name in args.models:
        spec = MODELS[name]
        out = RAW / f"{name}.json"
        if out.exists():
            try:
                if json.loads(out.read_text()).get("input_sha256") == in_hash:
                    print(f"{name}: up to date, skipping", flush=True)
                    continue
            except Exception:
                pass
        print(f"=== {name} ({spec['hf']}) ===", flush=True)
        model = load_from_checkpoint(download_model(spec["hf"]))
        model.eval()
        data = [{"src": s, "mt": m, **({"ref": r} if spec["ref"] else {})}
                for s, m, r in zip(src, mt, ref)]
        bs = args.batch_size or spec["batch"]
        res = model.predict(data, batch_size=bs, gpus=1, progress_bar=True)
        scores = [float(x) for x in res.scores]
        payload = {
            "metric": name, "hf_model": spec["hf"], "reference_based": spec["ref"],
            "input_sha256": in_hash, "n": len(scores), "batch_size": bs,
            "system_score": float(res.system_score),
            "scores": scores,
        }
        meta = getattr(res, "metadata", None)
        spans = getattr(meta, "error_spans", None) if meta is not None else None
        if spans is not None:
            payload["error_spans"] = spans
            print(f"{name}: kept error_spans for {len(spans)} segments", flush=True)
        # ⛔ `start` and `end` inside an xCOMET error span are 0-dim TORCH TENSORS, not ints:
        # `UnifiedMetric.decode` sets `span["offset"] = list(token_offset)` and `flatten_metadata`
        # only calls `.tolist()` on values that are themselves tensors -- a list of lists of dicts
        # takes the other branch and the tensors survive. A plain `json.dumps` therefore raises
        # `TypeError: Object of type Tensor is not JSON serializable` AT THE FINAL WRITE, after
        # the whole forward pass. It destroyed one complete 60-minute XCOMET-XL run before this
        # default was added. Never remove it.
        def _plain(o):
            if hasattr(o, "item"):
                return o.item()
            if hasattr(o, "tolist"):
                return o.tolist()
            raise TypeError(f"unserialisable {type(o)}")
        # Write to a temp path first so a serialisation failure cannot leave a truncated file
        # that the resume check would then treat as complete.
        tmp = out.with_suffix(".json.partial")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=_plain))
        tmp.replace(out)
        print(f"{name}: system={payload['system_score']:.4f} -> {out}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
