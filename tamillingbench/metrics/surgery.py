# -*- coding: utf-8 -*-
"""Minimal-pair surgery on Tamil sentences.

The whole metric-blindness argument turns on one property of the scored pairs: the two
strings a metric compares must differ in **exactly one grapheme-cluster span**, and that span
must be the obligatory-slot morpheme. If they differ anywhere else, a small metric delta is
uninterpretable -- it could be the other difference.

The benchmark's `contrast_targets` are NOT minimal pairs. Measured on the shipped items:

    gender   gold 'அகநகை வந்தாள்.'  contrast 'அகரன் வந்தான்.'      <- the NAME changed too
    rational gold 'நாய்கள் வந்தன.'   contrast 'மாணவர்கள் வந்தார்கள்.' <- the NOUN changed too

They are the paired *item* from the same set, not a minimal edit. So the flipped reference is
constructed here by splicing the recorded `morpheme_char_span`, and every result is gated on a
grapheme-cluster diff assertion. Sandhi means the splice is not always clean; those cases are
rejected and counted, never silently repaired.

Grapheme clusters come from `tamil.utf8.get_letters` (open-tamil), which is the project's
sanctioned segmenter -- a Tamil "letter" is consonant+vowel-sign, so a naive codepoint diff
would report two changes where a reader sees one.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import tamil.utf8 as tutf8
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_INDIC_NORM = IndicNormalizerFactory().get_normalizer("ta")

ZERO_WIDTH = "‌‍﻿­"


def normalize_sentence(s: str) -> str:
    """Canonical form for a whole sentence.

    Deliberately NOT `morph.normalize`: that strips edge punctuation, which is correct for a
    single token entering the FST and wrong for a sentence entering COMET. Composition of the
    two-part vowel signs is the part that matters, because an uncomposed 'கொ' would shift every
    downstream character offset and turn a span hit into a spurious miss.
    """
    s = s.translate({ord(c): None for c in ZERO_WIDTH})
    s = unicodedata.normalize("NFC", s)
    s = _INDIC_NORM.normalize(s)
    return unicodedata.normalize("NFC", s)


def letters(s: str) -> list[str]:
    """Grapheme clusters, open-tamil's definition. Non-Tamil chars come back one per item."""
    return tutf8.get_letters(s)


def n_diff_spans(a: str, b: str) -> int:
    """Number of maximal differing runs between two grapheme-cluster sequences.

    A single contiguous edit -- substitution, insertion or deletion -- is one span. Two edits
    in different words are two. This is the assertion that makes a metric delta attributable.
    """
    la, lb = letters(a), letters(b)
    # longest common prefix / suffix over grapheme clusters
    i = 0
    while i < len(la) and i < len(lb) and la[i] == lb[i]:
        i += 1
    j = 0
    while (j < len(la) - i) and (j < len(lb) - i) and la[len(la) - 1 - j] == lb[len(lb) - 1 - j]:
        j += 1
    mid_a, mid_b = la[i:len(la) - j], lb[i:len(lb) - j]
    if not mid_a and not mid_b:
        return 0
    # One contiguous run on each side => exactly one span. If the middles still share
    # material the edit was not contiguous; fall back to a full LCS-based run count.
    if not (set(mid_a) & set(mid_b)):
        return 1
    return _diff_runs(mid_a, mid_b) or 1


def _diff_runs(a: list[str], b: list[str]) -> int:
    """Count differing runs via a standard LCS backtrace. Only called on the rare non-clean case."""
    n, m = len(a), len(b)
    if n * m > 250_000:            # pathological; treat as many-span and reject upstream
        return 99
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for x in range(n - 1, -1, -1):
        for y in range(m - 1, -1, -1):
            dp[x][y] = dp[x + 1][y + 1] + 1 if a[x] == b[y] else max(dp[x + 1][y], dp[x][y + 1])
    x = y = 0
    runs, in_run = 0, False
    while x < n and y < m:
        if a[x] == b[y]:
            in_run = False
            x, y = x + 1, y + 1
        else:
            if not in_run:
                runs += 1
                in_run = True
            if dp[x + 1][y] >= dp[x][y + 1]:
                x += 1
            else:
                y += 1
    if x < n or y < m:
        if not in_run:
            runs += 1
    return runs


@dataclass(frozen=True)
class Splice:
    text: str
    ok: bool
    reason: str = ""


def splice(text: str, span: tuple[int, int], replacement: str) -> Splice:
    """Replace `text[span[0]:span[1]]` with `replacement` on already-normalised text.

    Returns ok=False rather than raising, because a bad span is a data condition to be counted
    (`rejected_repairs.jsonl`), not an exception to be swallowed at a call site.
    """
    a, b = span
    if not (0 <= a <= b <= len(text)):
        return Splice(text, False, f"span {span} out of range for len {len(text)}")
    out = normalize_sentence(text[:a] + replacement + text[b:])
    return Splice(out, True)


def build_flipped(ref_gold: str,
                  gold_span: tuple[int, int],
                  contrast_surface: str,
                  contrast_full: str | None) -> Splice:
    """The `ref_flipped` variant: gold reference, obligatory morpheme set to a WRONG value.

    Two candidates are tried and the first that yields exactly one differing grapheme-cluster
    span wins:

      A. the benchmark's own `contrast_targets[i].tamil` -- attested and FST-verified, but only
         minimal when the contrast item shares the gold's lexical material (true for the
         imperative honorificity family, false for gender and rationality);
      B. a splice of `gold_span` with the contrast's surface morpheme -- always lexically
         minimal, but has to be re-verified by the FST because sandhi can make the join
         illegal.

    A is preferred when it qualifies because it needs no FST re-verification.
    """
    if contrast_full:
        cand_a = normalize_sentence(contrast_full)
        if n_diff_spans(ref_gold, cand_a) == 1:
            return Splice(cand_a, True, "contrast_target (already minimal)")
    sp = splice(ref_gold, gold_span, contrast_surface)
    if not sp.ok:
        return sp
    nd = n_diff_spans(ref_gold, sp.text)
    if nd != 1:
        return Splice(sp.text, False, f"splice produced {nd} differing spans, expected 1")
    return Splice(sp.text, True, "spliced morpheme span")
