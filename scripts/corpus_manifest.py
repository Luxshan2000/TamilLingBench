#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write `data/corpus/MANIFEST.json` — provenance, licence and release status.

Read by `scripts/check_release_licence.py` (the release-licence gate). Records, for every
corpus we touched: where it came from, what its licence says, whether its TEXT may ship,
whether its COUNTS may ship, and the sha256 of the local raw file so the build is
attributable to an exact byte sequence.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamillingbench.corpus.sources import SOURCES        # noqa: E402

RAW = ROOT / "data" / "corpus" / "raw"
INDEX = ROOT / "data" / "corpus" / "index"
OUT = ROOT / "data" / "corpus" / "MANIFEST.json"

RAW_FILES = {
    "tawiki": ["tawiki-latest-pages-articles.xml.bz2"],
    "opensubs": ["opensubtitles-v2024-ta.txt.gz"],
    "sangraha": ["sangraha-verified-tam-0.parquet", "sangraha-verified-tam-26.parquet"],
    "sangraha_speech": ["sangraha-verified-tam-0.parquet", "sangraha-verified-tam-26.parquet"],
    "cc100": ["cc100-ta.txt.xz"],
    "irumozhi": ["irumozhi-regdataset.csv"],
    "tatoeba": ["tatoeba-ta.txt.gz"],
}


def sha256(p: Path, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if n >= limit:
                return h.hexdigest() + f"(first {limit} bytes)"
    return h.hexdigest()


def main() -> None:
    build_log = {}
    p = INDEX / "build-log.json"
    if p.exists():
        build_log = {r["corpus"]: r for r in json.loads(p.read_text())}

    entries = []
    for key, s in SOURCES.items():
        files = []
        for name in RAW_FILES.get(key, []):
            f = RAW / name
            if f.exists():
                files.append({"file": name, "bytes": f.stat().st_size,
                              "sha256": sha256(f)})
        entries.append({
            "key": key, "name": s.name, "url": s.url,
            "licence": s.licence, "licence_checked": s.licence_checked,
            "register": s.register, "role": s.role,
            "obtained": s.obtained,
            "indexed": key in build_log,
            "release_text_ok": s.release_ok,
            "release_counts_ok": s.derived_ok,
            "size_note": s.size_note, "citation": s.citation, "notes": s.notes,
            "raw_files": files,
            "index_stats": build_log.get(key),
        })

    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release_gate": {
            "text_blocked": sorted(e["key"] for e in entries if not e["release_text_ok"]),
            "counts_blocked": sorted(e["key"] for e in entries
                                     if not e["release_counts_ok"]),
            "note": ("DECISIONS.md D-4. `text_blocked` corpora may contribute "
                     "aggregate counts to reports but no sentence may be quoted or "
                     "shipped. `counts_blocked` corpora contribute nothing at all — "
                     "UD_Tamil-TTB (CC BY-NC-SA 3.0) is deliberately not even indexed."),
        },
        "sources": entries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    print("text-blocked:", doc["release_gate"]["text_blocked"])
    print("counts-blocked:", doc["release_gate"]["counts_blocked"])


if __name__ == "__main__":
    main()
