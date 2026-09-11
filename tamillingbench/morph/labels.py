# -*- coding: utf-8 -*-
"""Label tables (see Trap B).

Two independent jobs live here:

1. `LABEL_TO_SURFACE` — the fix for Trap B. ThamizhiMorph writes bound suffixes with
   INDEPENDENT vowel letters (ஆ U+0B86) while surfaces use COMBINING vowel signs
   (ா U+0BBE), so `surface.endswith(binding.suffix)` is always False for that class.
   Measured over 3,909 unique UD tokens: 623 bindings have an independent-vowel-initial
   suffix and `surface.endswith(raw_suffix)` is true for **exactly 0** of them.

2. The FST-tag → linguistic-slot mapping. Every entry is grounded in a tag actually emitted
   by the shipped nets (inventory dumped over TTB train+test and MWTT test), not in the
   ThamizhiMorph paper.
"""
from __future__ import annotations

from enum import Enum

# --------------------------------------------------------------------------- Trap B

#: Independent vowel letter → the combining vowel sign that realises it word-medially.
#: அ U+0B85 maps to the EMPTY string: short /a/ is Tamil's inherent vowel, so it is realised
#: by *removing* the preceding consonant's pulli rather than by adding a sign. That is why
#: `+3pln=அன` surfaces as ...தன in வந்தன (த் + அன → தன).
LABEL_TO_SURFACE: dict[str, str] = {
    "அ": "",     # U+0B85 inherent /a/  → no sign
    "ஆ": "ா",    # U+0B86 → U+0BBE
    "இ": "ி",    # U+0B87 → U+0BBF
    "ஈ": "ீ",    # U+0B88 → U+0BC0
    "உ": "ு",    # U+0B89 → U+0BC1
    "ஊ": "ூ",    # U+0B8A → U+0BC2
    "எ": "ெ",    # U+0B8E → U+0BC6
    "ஏ": "ே",    # U+0B8F → U+0BC7
    "ஐ": "ை",    # U+0B90 → U+0BC8
    "ஒ": "ொ",    # U+0B92 → U+0BCA
    "ஓ": "ோ",    # U+0B93 → U+0BCB
    "ஔ": "ௌ",   # U+0B94 → U+0BCC
}

#: The FST writes a zero morph as U+2205 EMPTY SET.
ZERO_MORPH = "∅"


def label_suffix_to_surface(suffix: str) -> str:
    """Convert a bound suffix label string to the shape it takes on the surface.

    Only the FIRST character can be an independent vowel — everything after it is already
    written in surface form. This is why the table is applied to `suffix[0]` only.

        'ஆன்'   -> 'ான்'      (வந்தான் ends with it)
        'அன'    -> 'ன'        (வந்தன ends with it; the அ removed த்'s pulli)
        'இல்லை' -> 'ில்லை'    (வரவில்லை ends with it)
        'த்'    -> 'த்'       (consonant-initial: unchanged)
        '∅'     -> ''
    """
    if not suffix or suffix == ZERO_MORPH:
        return ""
    return LABEL_TO_SURFACE.get(suffix[0], suffix[0]) + suffix[1:]


# --------------------------------------------------------------------------- slots

class Slot(str, Enum):
    HONORIFICITY = "honorificity"   # FAMILIAR | POLITE  (DEFERENTIAL is a SURFACE label for
                                    # தாங்கள் only; D-6.1 groups it under POLITE at scoring)
    CLUSIVITY = "clusivity"         # INCL | EXCL
    RATIONALITY = "rationality"     # UYARTHINAI | AHRINAI
    GENDER = "gender"               # MASC | FEM | EPICENE | NEUT
    NUMBER = "number"               # SG | PL            (the control slot)
    POLARITY = "polarity"           # POS | NEG


SG, PL = "SG", "PL"
MASC, FEM, EPICENE, NEUT = "MASC", "FEM", "EPICENE", "NEUT"
FAMILIAR, POLITE, DEFERENTIAL = "FAMILIAR", "POLITE", "DEFERENTIAL"
UYARTHINAI, AHRINAI = "UYARTHINAI", "AHRINAI"
INCL, EXCL = "INCL", "EXCL"
POS, NEG = "POS", "NEG"

#: Person-number-gender tag → feature bundle. One entry per PNG tag the nets actually emit.
#:
#: A tag may map to MORE THAN ONE bundle when it is genuinely syncretic in the language
#: rather than merely ambiguous in the analyser. `2plh` (நீங்கள்) is the case that matters:
#: it is honorific-singular OR plain-plural, and *nothing in the word form* decides which.
#: Each bundle becomes a separate feature reading, so D-2's context filter can drop the one
#: the item's declared referent number rules out. This is exactly why the design mandates
#: `confound_checks.number` on every honorificity item.
#:
#: `3sghe` vs `3ple` on வந்தார்கள் is NOT handled here — the FST already returns them as two
#: separate analyses, so the ambiguity arrives pre-split.
PNG_FEATURES: dict[str, tuple[dict[Slot, str], ...]] = {
    # --- 1st person
    "1sg": ({Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI},),
    "1pl": ({Slot.NUMBER: PL, Slot.RATIONALITY: UYARTHINAI},),
    # --- 2nd person
    "2sg": ({Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI, Slot.HONORIFICITY: FAMILIAR},),
    "2sgn": ({Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI, Slot.HONORIFICITY: FAMILIAR},),
    "2sgm": ({Slot.NUMBER: SG, Slot.GENDER: MASC, Slot.RATIONALITY: UYARTHINAI,
              Slot.HONORIFICITY: FAMILIAR},),
    "2sgf": ({Slot.NUMBER: SG, Slot.GENDER: FEM, Slot.RATIONALITY: UYARTHINAI,
              Slot.HONORIFICITY: FAMILIAR},),
    "2sgh": ({Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI, Slot.HONORIFICITY: POLITE},),
    "2pl": ({Slot.NUMBER: PL, Slot.RATIONALITY: UYARTHINAI, Slot.HONORIFICITY: FAMILIAR},),
    # நீங்கள் — syncretic, two readings. THE case D-2's context filter exists for.
    "2plh": (
        {Slot.NUMBER: SG, Slot.RATIONALITY: UYARTHINAI, Slot.HONORIFICITY: POLITE},
        {Slot.NUMBER: PL, Slot.RATIONALITY: UYARTHINAI, Slot.HONORIFICITY: FAMILIAR},
    ),
    # --- 3rd person
    "3sgm": ({Slot.NUMBER: SG, Slot.GENDER: MASC, Slot.RATIONALITY: UYARTHINAI,
              Slot.HONORIFICITY: FAMILIAR},),
    "3sgf": ({Slot.NUMBER: SG, Slot.GENDER: FEM, Slot.RATIONALITY: UYARTHINAI,
              Slot.HONORIFICITY: FAMILIAR},),
    "3sgn": ({Slot.NUMBER: SG, Slot.GENDER: NEUT, Slot.RATIONALITY: AHRINAI},),
    "3pln": ({Slot.NUMBER: PL, Slot.GENDER: NEUT, Slot.RATIONALITY: AHRINAI},),
    # honorific singular: வந்தார் / அவர். 'he' = honorific epicene.
    "3sgh": ({Slot.NUMBER: SG, Slot.GENDER: EPICENE, Slot.RATIONALITY: UYARTHINAI,
              Slot.HONORIFICITY: POLITE},),
    "3sghe": ({Slot.NUMBER: SG, Slot.GENDER: EPICENE, Slot.RATIONALITY: UYARTHINAI,
               Slot.HONORIFICITY: POLITE},),
    "3sge": ({Slot.NUMBER: SG, Slot.GENDER: EPICENE, Slot.RATIONALITY: UYARTHINAI},),
    "3ple": ({Slot.NUMBER: PL, Slot.GENDER: EPICENE, Slot.RATIONALITY: UYARTHINAI,
              Slot.HONORIFICITY: FAMILIAR},),
    "3pl": ({Slot.NUMBER: PL, Slot.RATIONALITY: UYARTHINAI},),
    "3sg": ({Slot.NUMBER: SG},),
    "pl": ({Slot.NUMBER: PL},),
}

#: SUFFIX-CONDITIONED SYNCRETISM — a documented repair of a ThamizhiMorph inconsistency.
#:
#: The finite suffix -ஆர் is 3rd-person HONORIFIC SINGULAR *or* 3rd-person RATIONAL PLURAL.
#: Both readings are always available; nothing in the word form decides. The FST does not
#: apply this uniformly — measured on the shipped nets:
#:     செய்தார்   -> ...+3sghe=ஆர்     (honorific singular only)
#:     பிறந்தார்  -> ...+3ple=ஆர்      (plural only)
#: Left alone, that inconsistency silently decides honorificity items by which verb happens
#: to be in the lexicon, which is exactly the failure mode where
#: "false positives go unnoticed because coverage looks fine". So whenever -ஆர்/-ஆர்கள் is
#: bound to a 3rd-person label, BOTH readings are licensed and D-2's context filter decides.
#:
#: ⚠ Grammatical claim. Flagged for native-speaker audit.
#:
#: ⛔ EXTENDED 2026-08-08 during generation. Building the honorificity slot surfaced
#: the SAME defect one person lower in the paradigm: the 2nd-person suffixes -ஈர்கள்
#: (வந்தீர்கள்) and -உங்கள் (வாருங்கள்) are syncretic between 2nd-PLURAL and
#: 2nd-HONORIFIC-SINGULAR — that syncretism is the entire reason the design makes
#: `confound_checks.number` mandatory on every honorificity item — but ThamizhiMorph tags
#: both suffixes bare `+2pl`. The `2plh` fan-out in PNG_FEATURES therefore only ever fired on
#: the PRONOUN நீங்கள் and never on the verb, so `வந்தீர்கள்` with a declared singular
#: addressee was scored as *number-incompatible* and failed. Measured on our own generated
#: gold before the fix: 152 of 360 honorificity items rejected, all of them correct Tamil.
SUFFIX_SYNCRETISM: dict[str, tuple[str, ...]] = {
    "ஆர்": ("3sghe", "3ple"),
    "ஆர்கள்": ("3sghe", "3ple"),
    "ஈர்கள்": ("2pl", "2plh"),
    "உங்கள்": ("2pl", "2plh"),
}

#: Non-PNG tags that contribute a slot value on their own.
TAG_FEATURES: dict[str, dict[Slot, str]] = {
    "incl": {Slot.CLUSIVITY: INCL},
    "excl": {Slot.CLUSIVITY: EXCL},
}

#: Labels whose presence (bound or bare) marks negative polarity.
NEGATION_LABELS = frozenset({"neg", "negpart", "negcop", "negfut", "negdeb"})

#: POS tags for which absence of a negation label implies POS polarity. A noun or particle
#: analysis makes no polarity claim, so it must not be defaulted.
POLARITY_BEARING_POS = frozenset({"verb", "aux"})
