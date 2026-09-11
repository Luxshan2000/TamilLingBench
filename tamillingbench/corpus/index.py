# -*- coding: utf-8 -*-
"""`FormIndex` — the form-frequency index and KWIC concordance.

One SQLite file per corpus under `data/corpus/index/`. The file answers three questions:

    idx.frequency("நீங்கள்")        -> Frequency(n=…, df=…, per_million=…)
    idx.kwic("நீங்கள்", limit=10)   -> attested sentences, with left/right context
    idx.doc_counts(["நாம்", …])     -> per-document counts, for the cluster bootstrap

**Counts are exact; concordance lines are capped.** `typ.n` is the true corpus token count
for the form. The postings list backing `kwic()` is truncated at `POSTING_CAP` sentences per
form — otherwise the postings for a function word would dominate the file — and the
truncation is recorded per form so nothing silently looks complete when it is not. Never
compute a frequency by counting KWIC hits.
"""
from __future__ import annotations

import json
import sqlite3
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import re

from .text import assert_shared_canonicalization, canonicalize, is_tamil_line, sentences, tokenize

#: Combining vowel signs U+0BBE–U+0BCC and base consonants U+0B95–U+0BB9. Their ratio is a
#: cheap, corpus-independent detector for a specific and common Tamil-text corruption:
#: legacy TSCII/Bamini → Unicode conversions that DROP the vowel signs (or turn them into
#: spaces), so `\u0b85\u0ba9\u0bc8\u0ba4\u0bcd\u0ba4\u0bc1` comes out as `\u0b85\u0ba9\u0ba4\u0bcd\u0ba4`.
#: MEASURED 2026-08-08: Tamil Wikipedia median 0.432 (5th percentile 0.38); Sangraha
#: web/pdf/speech medians 0.446/0.455/0.454; OPUS OpenSubtitles-ta median **0.014**.
#: The threshold sits far below every healthy corpus and far above the damaged one.
#: This matters more than ordinary hygiene: the corruption deletes exactly the vowel signs
#: that CARRY our slot contrasts (-\u0b86\u0ba9\u0bcd vs -\u0b86\u0bb3\u0bcd vs -\u0b86\u0bb0\u0bcd), so damaged text does not merely
#: add noise — it silently converts one slot value into another.
_VOWEL_SIGN = re.compile("[\u0bbe-\u0bcc]")
_CONSONANT = re.compile("[\u0b95-\u0bb9]")
MIN_SIGN_RATIO = 0.20


def vowel_sign_ratio(text: str) -> float:
    c = len(_CONSONANT.findall(text))
    return len(_VOWEL_SIGN.findall(text)) / c if c else 0.0

#: Max sentence ids retained per form. Exact counts live in `typ.n` and are unaffected.
POSTING_CAP = 200

SCHEMA = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS doc(id INTEGER PRIMARY KEY, ref TEXT, n_tokens INTEGER);
CREATE TABLE IF NOT EXISTS typ(form TEXT PRIMARY KEY, n INTEGER, df INTEGER);
CREATE TABLE IF NOT EXISTS sent(id INTEGER PRIMARY KEY, doc INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS post(form TEXT PRIMARY KEY, blob BLOB, n_post INTEGER,
                                truncated INTEGER);
CREATE TABLE IF NOT EXISTS docform(doc INTEGER, form TEXT, n INTEGER);
"""


@dataclass(frozen=True)
class Document:
    """A unit of clustering for the bootstrap and of provenance for KWIC."""
    ref: str          # human-readable source id: wiki page title, subtitle id, doc index
    text: str


@dataclass(frozen=True)
class Frequency:
    form: str
    n: int            # token count in this corpus
    df: int           # number of documents containing it
    n_tokens: int     # corpus size, so the caller can renormalise

    @property
    def per_million(self) -> float:
        return 1e6 * self.n / self.n_tokens if self.n_tokens else 0.0

    @property
    def attested(self) -> bool:
        return self.n > 0


@dataclass(frozen=True)
class KwicHit:
    form: str
    left: str
    match: str
    right: str
    sentence: str
    doc_ref: str
    corpus: str


# --------------------------------------------------------------------------- varint codec

def _encode(ids: Sequence[int]) -> bytes:
    """Delta + LEB128. Sentence ids arrive ascending, so deltas are small."""
    out = bytearray()
    prev = 0
    for i in ids:
        d = i - prev
        prev = i
        while d >= 0x80:
            out.append((d & 0x7F) | 0x80)
            d >>= 7
        out.append(d)
    return bytes(out)


def _decode(blob: bytes) -> list[int]:
    out, cur, shift, prev = [], 0, 0, 0
    for b in blob:
        cur |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            prev += cur
            out.append(prev)
            cur, shift = 0, 0
    return out


# --------------------------------------------------------------------------- builder

class IndexBuilder:
    """Single-pass builder. Memory is held in parallel arrays keyed by an interned form id.

    A dict-of-lists keyed by form string costs ~3× this and Tamil Wikipedia has millions of
    types, so the id-array layout is not premature optimisation — it is what makes the build
    fit in 16 GB alongside everything else.
    """

    def __init__(self, path: str | Path, corpus: str, *,
                 store_sentences: bool = True,
                 full_types: bool = True,
                 candidate: "callable | None" = None,
                 min_doc_tokens: int = 20,
                 max_doc_tokens: int = 100_000,
                 tamil_threshold: float = 0.8,
                 min_sign_ratio: float = MIN_SIGN_RATIO,
                 dedup: bool = True) -> None:
        self.path = Path(path)
        self.corpus = corpus
        self.store_sentences = store_sentences
        self.full_types = full_types
        self.candidate = candidate
        self.min_doc_tokens = min_doc_tokens
        self.max_doc_tokens = max_doc_tokens          # per-document cap
        self.tamil_threshold = tamil_threshold
        self.min_sign_ratio = min_sign_ratio
        self.dedup = dedup

        self._id: dict[str, int] = {}
        self._forms: list[str] = []
        self._n = array("q")
        self._df = array("q")
        self._last = array("q")
        self._post: list[array] = []

        self.n_docs = 0
        self.n_tokens = 0
        self.n_tokens_raw = 0        # before the per-document cap
        self.n_sents = 0
        self.n_docs_dropped_short = 0
        self.n_docs_dropped_lang = 0
        self.n_docs_dropped_quality = 0
        self.n_lines_dedup = 0
        self._seen_lines: set[int] = set()
        self._docform: list[tuple[int, str, int]] = []
        self._doc_rows: list = []
        self._sent_rows: list = []
        self._db: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ internals
    def _fid(self, form: str) -> int:
        i = self._id.get(form)
        if i is None:
            i = len(self._forms)
            self._id[form] = i
            self._forms.append(form)
            self._n.append(0)
            self._df.append(0)
            self._last.append(-1)
            self._post.append(array("i"))
        return i

    def _keep(self, form: str) -> bool:
        return self.full_types or (self.candidate is not None and self.candidate(form))

    # ------------------------------------------------------------------ public
    def add(self, doc: Document) -> None:
        keep_lines: list[str] = []
        for line in doc.text.split("\n"):
            if not line.strip():
                continue
            if not is_tamil_line(line, self.tamil_threshold):
                continue
            if self.dedup:
                from .text import line_hash
                h = line_hash(line)
                if h in self._seen_lines:
                    self.n_lines_dedup += 1
                    continue
                self._seen_lines.add(h)
            keep_lines.append(line)
        if not keep_lines:
            self.n_docs_dropped_lang += 1
            return

        text = "\n".join(keep_lines)
        sents = list(sentences(text))
        toks_per_sent = [tokenize(s) for s in sents]
        total = sum(len(t) for t in toks_per_sent)
        if total < self.min_doc_tokens:
            self.n_docs_dropped_short += 1
            return
        if self.min_sign_ratio > 0:
            joined = " ".join(t for ts in toks_per_sent for t in ts)
            if vowel_sign_ratio(joined) < self.min_sign_ratio:
                self.n_docs_dropped_quality += 1
                return

        self.n_tokens_raw += total
        doc_id = self.n_docs
        self.n_docs += 1

        budget = self.max_doc_tokens
        counts: Counter[str] = Counter()
        used = 0
        for sent, toks in zip(sents, toks_per_sent):
            if budget <= 0:
                break
            toks = toks[:budget]
            budget -= len(toks)
            used += len(toks)
            sid = self.n_sents
            if self.store_sentences and toks:
                self._sent_rows.append((sid, doc_id, canonicalize(sent)))
                self.n_sents += 1
            counts.update(toks)
            if self.store_sentences:
                for form in set(toks):
                    if not self._keep(form):
                        continue
                    fi = self._fid(form)
                    p = self._post[fi]
                    if len(p) < POSTING_CAP:
                        p.append(sid)
        self.n_tokens += used

        for form, c in counts.items():
            if not self._keep(form):
                continue
            fi = self._fid(form)
            self._n[fi] += c
            if self._last[fi] != doc_id:
                self._last[fi] = doc_id
                self._df[fi] += 1
            if self.candidate is not None and self.candidate(form):
                self._docform.append((doc_id, form, c))

        self._doc_rows.append((doc_id, doc.ref, used))
        if len(self._doc_rows) >= 5_000 or len(self._sent_rows) >= 200_000 \
                or len(self._docform) >= 500_000:
            self._flush()

    def _flush(self) -> None:
        cur = self._db
        if self._doc_rows:
            cur.executemany("INSERT INTO doc VALUES(?,?,?)", self._doc_rows)
            self._doc_rows.clear()
        if self._sent_rows:
            cur.executemany("INSERT INTO sent VALUES(?,?,?)", self._sent_rows)
            self._sent_rows.clear()
        if self._docform:
            cur.executemany("INSERT INTO docform VALUES(?,?,?)", self._docform)
            self._docform.clear()
        cur.commit()

    def open(self) -> "IndexBuilder":
        assert_shared_canonicalization()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._db = sqlite3.connect(self.path)
        self._db.executescript(SCHEMA)
        return self

    def close(self, meta: dict | None = None) -> None:
        self._flush()
        cur = self._db
        cur.executemany(
            "INSERT INTO typ VALUES(?,?,?)",
            ((self._forms[i], self._n[i], self._df[i]) for i in range(len(self._forms))))
        if self.store_sentences:
            cur.executemany(
                "INSERT INTO post VALUES(?,?,?,?)",
                ((self._forms[i], _encode(self._post[i]), len(self._post[i]),
                  int(len(self._post[i]) >= POSTING_CAP))
                 for i in range(len(self._forms)) if len(self._post[i])))
        m = {
            "corpus": self.corpus,
            "n_docs": self.n_docs,
            "n_tokens": self.n_tokens,
            "n_tokens_uncapped": self.n_tokens_raw,
            "n_types": len(self._forms),
            "n_sents": self.n_sents,
            "n_docs_dropped_short": self.n_docs_dropped_short,
            "n_docs_dropped_lang": self.n_docs_dropped_lang,
            "n_docs_dropped_quality": self.n_docs_dropped_quality,
            "min_sign_ratio": self.min_sign_ratio,
            "n_lines_deduped": self.n_lines_dedup,
            "full_types": int(self.full_types),
            "store_sentences": int(self.store_sentences),
            "posting_cap": POSTING_CAP,
            "min_doc_tokens": self.min_doc_tokens,
            "max_doc_tokens": self.max_doc_tokens,
        }
        m.update(meta or {})
        cur.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)",
                        ((k, json.dumps(v, ensure_ascii=False)) for k, v in m.items()))
        cur.execute("CREATE INDEX IF NOT EXISTS ix_docform ON docform(form)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_sentdoc ON sent(doc)")
        cur.commit()
        cur.close()


# --------------------------------------------------------------------------- reader

class FormIndex:
    """Read-only query interface. Cheap to open; safe to keep around."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.meta = {k: json.loads(v) for k, v in
                     self._db.execute("SELECT key,value FROM meta")}
        self.corpus = self.meta.get("corpus", self.path.stem)
        self.n_tokens = int(self.meta.get("n_tokens", 0))
        self.n_docs = int(self.meta.get("n_docs", 0))

    # ------------------------------------------------------------------ frequency
    def frequency(self, form: str) -> Frequency:
        f = canonicalize(form)
        row = self._db.execute("SELECT n,df FROM typ WHERE form=?", (f,)).fetchone()
        n, df = row if row else (0, 0)
        return Frequency(form=f, n=n, df=df, n_tokens=self.n_tokens)

    def frequencies(self, forms: Iterable[str]) -> dict[str, Frequency]:
        return {f: self.frequency(f) for f in forms}

    def attested(self, form: str) -> bool:
        return self.frequency(form).n > 0

    def top(self, limit: int = 50) -> list[Frequency]:
        return [Frequency(f, n, df, self.n_tokens) for f, n, df in
                self._db.execute("SELECT form,n,df FROM typ ORDER BY n DESC LIMIT ?",
                                 (limit,))]

    def forms_with_suffix(self, suffix: str, min_n: int = 1) -> list[Frequency]:
        s = canonicalize(suffix)
        return [Frequency(f, n, df, self.n_tokens) for f, n, df in
                self._db.execute(
                    "SELECT form,n,df FROM typ WHERE form LIKE ? AND n>=? ORDER BY n DESC",
                    ("%" + s, min_n))]

    def forms_with_prefix(self, prefix: str, min_n: int = 1) -> list[Frequency]:
        """Every attested type beginning with `prefix` — the competitor enumeration.

        `scripts/gold_attestation_gate.py` uses this to answer the only question that makes
        a zero corpus count *diagnostic* rather than merely suggestive: is there an attested
        alternative for the same paradigm cell? `ஓடினன` has n=0 and so does every rare
        inflection of a rare verb; what separates them is that `ஓடின` (311) and `ஓடியது`
        (740) sit right next to it in the same prefix family and `நீந்தினன`'s neighbours do
        not. Runs on the `typ` PRIMARY KEY, so it is an index range scan.
        """
        p = canonicalize(prefix)
        return [Frequency(f, n, df, self.n_tokens) for f, n, df in
                self._db.execute(
                    "SELECT form,n,df FROM typ WHERE form>=? AND form<? AND n>=? "
                    "ORDER BY n DESC", (p, p + "￿", min_n))]

    def iter_types(self, min_n: int = 1) -> Iterator[tuple[str, int, int]]:
        yield from self._db.execute("SELECT form,n,df FROM typ WHERE n>=?", (min_n,))

    # ------------------------------------------------------------------ concordance
    def kwic(self, form: str, limit: int = 20, window: int = 8) -> list[KwicHit]:
        """Attested sentences containing `form`, as key-word-in-context lines.

        This is what curation verification consumes and what naturalness
        grounding needs — Bawden et al.'s convention is items *inspired by attested corpus
        sentences*, which requires the sentences to be in hand, with provenance.
        """
        f = canonicalize(form)
        row = self._db.execute("SELECT blob FROM post WHERE form=?", (f,)).fetchone()
        if not row:
            return []
        sids = _decode(row[0])[:limit]
        if not sids:
            return []
        q = ",".join("?" * len(sids))
        rows = self._db.execute(
            f"SELECT s.id, s.text, d.ref FROM sent s JOIN doc d ON d.id=s.doc "
            f"WHERE s.id IN ({q})", sids).fetchall()
        by_id = {r[0]: (r[1], r[2]) for r in rows}
        hits: list[KwicHit] = []
        for sid in sids:
            if sid not in by_id:
                continue
            text, ref = by_id[sid]
            toks = tokenize(text)
            try:
                i = toks.index(f)
            except ValueError:
                continue
            hits.append(KwicHit(
                form=f,
                left=" ".join(toks[max(0, i - window):i]),
                match=toks[i],
                right=" ".join(toks[i + 1:i + 1 + window]),
                sentence=text, doc_ref=ref, corpus=self.corpus))
        return hits

    def concordance_truncated(self, form: str) -> bool:
        f = canonicalize(form)
        row = self._db.execute("SELECT truncated FROM post WHERE form=?", (f,)).fetchone()
        return bool(row and row[0])

    # ------------------------------------------------------------------ bootstrap input
    def doc_counts(self, forms: Sequence[str]) -> dict[str, dict[int, int]]:
        """form -> {doc_id: count}, restricted to documents that contain the form.

        Documents are the clustering unit for the bootstrap: one narrator
        uses நீ throughout, so an i.i.d. token bootstrap is badly overconfident.
        """
        out: dict[str, dict[int, int]] = {canonicalize(f): {} for f in forms}
        keys = list(out)
        for i in range(0, len(keys), 400):
            chunk = keys[i:i + 400]
            q = ",".join("?" * len(chunk))
            for doc, form, n in self._db.execute(
                    f"SELECT doc,form,n FROM docform WHERE form IN ({q})", chunk):
                out[form][doc] = out[form].get(doc, 0) + n
        return out

    def doc_token_counts(self) -> dict[int, int]:
        return dict(self._db.execute("SELECT id,n_tokens FROM doc"))

    def close(self) -> None:
        self._db.close()
