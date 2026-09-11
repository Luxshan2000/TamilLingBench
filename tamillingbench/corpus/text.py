# -*- coding: utf-8 -*-
"""Canonicalization, tokenization and line hygiene for corpus text.

⚠ The single most dangerous silent bug in this project (a normalization mismatch between
the hypothesis path and the corpus path): if the corpus counts
`கொ` decomposed (க + ெ + ா) and the model-output path counts it composed (க + ொ), the two
strings are visually identical, compare unequal, and every DFR number is quietly wrong.

So there is exactly ONE canonicalization function in the project, it lives in
`tamillingbench.morph.normalize`, and this module wraps it rather than reimplementing it.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterator

from ..morph.normalize import normalize as _morph_normalize
from ..morph.normalize import strip_zero_width

#: Tamil block U+0B80–U+0BFF. `TAMIL_CHAR` includes the combining vowel signs and pulli,
#: so a "Tamil word" is a maximal run of characters from this block.
TAMIL_CHAR = re.compile(r"[஀-௿]")
TAMIL_WORD = re.compile(r"[஀-௿]+")
_NON_TAMIL_KEEP = re.compile(r"[\s​-‍﻿­]")

#: Sentence terminators. Tamil uses ASCII '.' '?' '!' plus the danda U+0964/U+0965 (which
#: `TamilNormalizer` maps the Tamil poorna virama U+0BE4/5 onto) and the ellipsis.
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥…])[\s\"'“”‘’)\]]*\s+")


def canonicalize(s: str) -> str:
    """THE canonicalization entry point for corpus text.

    Identical to `morph.normalize.normalize(s, reject_latin=False)`. Latin rejection is
    switched off because corpus lines legitimately contain Latin — we *filter* such lines
    (`is_tamil_line`) and *drop* such tokens (`tokenize`) rather than raising.

    Do not add steps here. Any transformation applied to corpus text but not to model
    output re-opens exactly the mismatch this module exists to close.
    """
    return _morph_normalize(s, reject_latin=False)


def assert_shared_canonicalization() -> None:
    """Hard assertion that the corpus path and the checker path agree.

    Called at the top of every index build. Cheap, and it fails loudly the day someone
    'improves' one of the two normalizers.
    """
    probes = [
        "\u0b95\u0bc6\u0bbe",   # \u0b95\u0bca DECOMPOSED  \u0b95 + \u0bc6 + \u0bbe
        "\u0b95\u0bca",         # \u0b95\u0bca COMPOSED
        "\u0b95\u0bc7\u0bbe",   # \u0b95\u0bcb decomposed
        "\u0b95\u0bcb",         # \u0b95\u0bcb composed
        "\u0ba8\u0bc0\u0b99\u0bcd\u0b95\u0bb3\u0bcd",
        "\u0ba8\u0bbe\u0b99\u0bcd\u0b95\u0bb3\u0bcd",
        "\u0bb5\u0ba8\u0bcd\u0ba4\u0bbe\u0ba9\u0bcd",
        "\u0bb5\u0ba8\u0bcd\u0ba4\u0bbe\u0bb3\u0bcd",
    ]
    if canonicalize("\u0b95\u0bc6\u0bbe") != canonicalize("\u0b95\u0bca"):
        raise AssertionError("two-part vowel sign \u0bc6+\u0bbe -> \u0bca is NOT being composed")
    for p in probes:
        mine = canonicalize(p)
        theirs = _morph_normalize(p, reject_latin=True)
        if mine != theirs:
            raise AssertionError(
                f"canonicalization drift on {p!r}: corpus={mine!r} checker={theirs!r}")
        if unicodedata.normalize("NFC", mine) != mine:
            raise AssertionError(f"canonicalize() left {p!r} un-composed: {mine!r}")


def tamil_ratio(line: str) -> float:
    """Share of non-space characters that are Tamil. The line-level language-ID filter."""
    chars = [c for c in line if not _NON_TAMIL_KEEP.match(c)]
    if not chars:
        return 0.0
    return sum(bool(TAMIL_CHAR.match(c)) for c in chars) / len(chars)


def is_tamil_line(line: str, threshold: float = 0.8) -> bool:
    """Line hygiene: drop lines with <80% characters in U+0B80–U+0BFF.

    Punctuation and digits count against the line, which is the point — a line that is
    mostly a table of numbers is not Tamil text and must not contribute to the prior.
    """
    return tamil_ratio(line) >= threshold


def tokenize(text: str) -> list[str]:
    """Canonicalize, then split into maximal Tamil-script words.

    Non-Tamil material (Latin, digits, punctuation) is a token boundary and is dropped.
    Canonicalization runs on the whole string BEFORE splitting so that a two-part vowel
    sign spanning the boundary of a naive split cannot survive.
    """
    return TAMIL_WORD.findall(canonicalize(text))


def sentences(text: str) -> Iterator[str]:
    """Split a document into sentences. Cheap and deliberately conservative.

    Tamil has no abbreviation-heavy convention comparable to English, so a terminator +
    whitespace rule is adequate; over-splitting a KWIC line costs context, not correctness.
    """
    for para in strip_zero_width(text).split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                yield s


def line_hash(s: str) -> int:
    """Stable 64-bit hash of a canonicalized line, for exact dedup."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(canonicalize(s).encode("utf-8"),
                                          digest_size=8).digest(), "big")
