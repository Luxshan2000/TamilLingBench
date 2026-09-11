# -*- coding: utf-8 -*-
"""Tamil normalization.

MUST run before anything touches the FST. Without it, visually identical strings compare
unequal because the two-part vowel signs have both a decomposed and a composed encoding:

    கொ  decomposed = U+0B95 U+0BC6 U+0BBE   (க + ெ + ா)
    கொ  composed   = U+0B95 U+0BCA          (க + ொ)

Model output, UD files and hand-written templates will not agree on which one they use.
"""
from __future__ import annotations

import re
import unicodedata

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# TamilNormalizer: composes ெ+ா→ொ, ே+ா→ோ, ெ+ௗ→ௌ, ஒ+ௗ→ஔ; maps poorna virama
# U+0BE4/5 → U+0964/5 and ':' after a Tamil char → visarga U+0B83.
_NORM = IndicNormalizerFactory().get_normalizer("ta")

ZERO_WIDTH = "‌‍﻿­"

# Punctuation and quotation marks stripped from token edges.
_EDGE_PUNCT = "\"'“”‘’«»`´.,;:!?()[]{}<>…—–-*/\\|_+=~^&%$#@¶§௦"

_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9०-९]")

# Tamil block plus the visarga/aytham. Used only for the artifact guard.
_TAMIL = re.compile(r"[஀-௿]")

# Consonants that carry an explicit pulli (virama) U+0BCD.
PULLI = "்"


class NonTamilToken(ValueError):
    """Raised for a token containing Latin characters or digits.

    Rejected loudly rather than silently passed: a Latin-containing token in a Tamil
    benchmark item is a generation bug, and swallowing it hides the bug.
    """


def strip_zero_width(s: str) -> str:
    return s.translate({ord(c): None for c in ZERO_WIDTH})


def strip_edge_punct(s: str) -> str:
    return s.strip(_EDGE_PUNCT).strip()


def normalize(surface: str, *, reject_latin: bool = True) -> str:
    """Canonical normalization for every string that enters or leaves the checker.

    Order matters: zero-width removal first (a ZWNJ between a consonant and a vowel sign
    would otherwise block composition), then edge punctuation, then TamilNormalizer.

    Deliberately NOT done: case folding, pulli stripping, trailing-consonant
    trimming. The last of these would silently "repair" TTB's tokenization artifacts
    (`கூறிய்`) into real words and inflate coverage.
    """
    s = strip_zero_width(surface)
    s = strip_edge_punct(s)
    if reject_latin and _LATIN_OR_DIGIT.search(s):
        raise NonTamilToken(f"token contains Latin/digit characters: {s!r}")
    return _NORM.normalize(s)


def is_tamil(s: str) -> bool:
    return bool(_TAMIL.search(s))


# The consonants TTB's clitic splitting strands on the *left* token. Derived from data,
# not assumed: over TTB-test's 95 unanalysable VERB/AUX forms the pulli-final tails are
# ர்16 ல்11 ம்9 வ்9 ய்7 ட்5 த்4 ற்2 ப்1 ன்1, and only the வ்/ய் tails (9+7 = 16) are forms
# whose vowel-restored counterpart is a real word. The ர்/ல்/ம் tails are genuine analyser
# gaps (`என்றனர்`, `கேட்டனர்` are real -அனர் rational plurals; `செல்லல்`, `வரல்` are verbal
# nouns; `விடில்` is a conditional) and must NOT be excused as artifacts.
TTB_SPLIT_CONSONANTS = "வய"


def looks_like_split_artifact(s: str) -> bool:
    """R10 shape test — TTB tokenization artifact guard.

    TTB's clitic splitting emits non-words by moving a token-final sandhi consonant off the
    following token onto this one: `கூறிய்` = `கூறி` + a stranded ய். These are not analyser
    failures and must be excluded from coverage denominators, not counted as misses.

    This is only the cheap *shape* test. `MorphChecker.is_artifact_token` is the definitive
    one: shape test AND the FST fails AND the vowel-restored form analyses. Real words that
    end in ய் (`செய்`, `வாய்`) are analysed by the FST and therefore never reach the guard.
    """
    return len(s) >= 3 and s.endswith(PULLI) and s[-2] in TTB_SPLIT_CONSONANTS


def nfc(s: str) -> str:
    """Python's own NFC. Kept only for the smoke test's comparison against TamilNormalizer."""
    return unicodedata.normalize("NFC", s)
