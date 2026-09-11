#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chrF and chrF++ with the full sacreBLEU signature. CPU only.

chrF is a character-n-gram metric on a language where the contrast IS one or two characters, so it
is the strongest possible prior for "at least this one should notice". That makes it a floor rather
than a strawman: if chrF cannot see the morpheme, no surface metric can.

Both word orders are reported. `--chrf-word-order 0` is chrF2 (the sacreBLEU default and the one
with a ToShip key); word order 2 is chrF++, carried as a secondary row.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sacrebleu.metrics import CHRF
import sacrebleu

BASE = Path(os.environ.get("MB_BASE",
                           Path(__file__).resolve().parent.parent / "outputs/metric_blindness"))
IN = BASE / os.environ.get("MB_IN", "inputs")
RAW = BASE / os.environ.get("MB_RAW", "raw")


def main() -> int:
    mt = (IN / "segments.mt.txt").read_text().splitlines()
    ref = (IN / "segments.ref.txt").read_text().splitlines()
    assert len(mt) == len(ref)
    in_hash = (IN / "inputs.sha256").read_text().strip()
    RAW.mkdir(parents=True, exist_ok=True)

    for name, wo in (("chrf", 0), ("chrfpp", 2)):
        m = CHRF(word_order=wo)
        # Sentence level. `sentence_score` on an empty hypothesis is defined and returns 0.0,
        # which is exactly the z denominator we want; it is not a missing value.
        scores = [float(m.sentence_score(h, [r]).score) for h, r in zip(mt, ref)]
        corpus = m.corpus_score(mt, [ref])
        payload = {
            "metric": name, "word_order": wo, "scale": "0-100, higher is better",
            "signature": str(m.get_signature()),
            "sacrebleu_version": sacrebleu.__version__,
            "toship_key": "chrf" if wo == 0 else None,
            "input_sha256": in_hash, "n": len(scores),
            "corpus_score": float(corpus.score),
            "scores": scores,
        }
        (RAW / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False))
        print(f"{name}: corpus={corpus.score:.4f}  {payload['signature']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
