# -*- coding: utf-8 -*-
"""Paradigm repairs — the cells where ThamizhiMorph's *generation* is wrong about Tamil.

`screens.py` removes over-generation that is detectable from the string or from the
transducer's own inconsistency. This file handles the harder case: a cell where the
transducer generates **exactly one** form, that form round-trips, passes every orthographic
and stem screen, and is **still not the Tamil word**. Nothing internal to the transducer can
detect that. Only the corpus can.

MEASURED 2026-08-08 over `data/corpus/index/*.sqlite` (~105 M tokens, 7 corpora). For the
`-இன்` past class ThamizhiMorph builds the அஃறிணை (irrational) cells by concatenating the
past stem with the PNG suffix, which is right for `-த்`/`-ட்`/`-ன்ற்`/`-த்த்` and wrong here:

    bundle             FST form     n   n(final)   real form    n    n(final)
    past.3pl.irrat     ஓடினன        0        0     ஓடின        311      262
    past.3sg.neut      ஓடினது       1        1     ஓடியது      740      648

Summed over the **20** `-இன்`-class lemmas in `data/lexicon/verbs.yaml`, counting only
sentence-final (i.e. finite, not adjectival-participle) tokens:

    past.3pl.irrat     FST 4      vs   repaired 3 585      20/20 lemmas favour the repair
    past.3sg.neut      FST 18     vs   repaired 15 379     20/20 lemmas favour the repair

The sentence-final restriction is load-bearing and was not obvious: `எழுதின` has n=485 but
only 21 % of its tokens are sentence-final, because colloquial Tamil also uses `-இன` as the
ADJECTIVAL PARTICIPLE (`அவரு எழுதின கட்டுரை`) where literary Tamil uses `-இய`. Comparing raw
counts would have credited the repair for tokens that are not the cell under repair.

Two different verdicts follow, and the difference between them is the point of this file:

* **`past.3sg.neut` is REPAIRED.** `-இனது` → `-இயது`. The அஃறிணை singular of this class is
  built on the past adjectival participle `-இய` plus `து`, not on the past stem plus `-அது`.
  The repaired surface is verified three ways — orthographic screen B, a **corpus floor**,
  and, replacing the round trip, **the validated checker must return the intended feature
  bundle**. `ஓடியது` → `{NUMBER: SG, GENDER: NEUT, RATIONALITY: AHRINAI}` (via the checker's
  fallback rule R13, `-வது`/`-யது` → 3sgn). That last check is strictly stronger than the
  round trip: it is run against the transducer configuration that SCORES the models, not
  against the one that generated the form.

* **`past.3pl.irrat` is REPAIRED**, and why it could not be repaired at first is itself a
  tooling defect. The corpus was always unambiguous
  that the form is `ஓடின`; what blocked the repair was that the **checker could not read
  it**:

      ஓடின  ->  ஓடு+verb+nonfin+sim+past=இன்+adjpart=அ   ->  {POLARITY: POS}

  no NUMBER, no RATIONALITY. `MorphChecker.check` counts a reading that leaves the slot
  undefined as non-matching (its docstring says so), so a model that produced the CORRECT
  Tamil `நாய்கள் ஓடின` was scored WRONG — and would still have been scored wrong if we had
  simply swapped the gold string. Repairing the gold without repairing the analyser would
  have moved the error rather than removed it, so the cell was withdrawn until the
  analyser was fixed.

  **DECISIONS.md D-7.1** (native-speaker ruling) then settled it directly:
  *"நாய்கள் ஓடின. is a complete sentence"* — the ANALYSER is wrong, not the Tamil. The
  analyser repair is `morph/fallback.rule_in_class_finite` (R14), which adds the finite
  `3pl-n` reading and keeps the participle analysis. With R14 in place the checker returns
  `{NUMBER: PL, GENDER: NEUT, RATIONALITY: AHRINAI}` for `ஓடின`, so the same two verification
  tests that licensed the singular repair now pass for the plural, and the cell is repaired
  the same way: `-இனன` → the bare past stem `-இன`. `NATIVE-CHECK-RAT-5` is closed;
  `NATIVE-CHECK-RAT-6` (that the participle reading genuinely co-exists) is opened.

  ⚠ The repaired plural has **no separable PNG exponent**: the அஃறிணை-plural marker is fused
  into the past morph `-இன`, which is exactly why the transducer's concatenative `-இனன` is
  not Tamil. It is therefore recorded as a ZERO exponent rather than pointing at the final
  `ன`, which belongs to `-இன்`. `imp.2sg.fam` already ships as a zero-exponent cell, so this
  is an existing code path, not a new one.

Both entries are grammatical claims that we make and the transducer does not. Each is filed
as an open native-speaker check (`NATIVE-CHECK-RAT-4`, `NATIVE-CHECK-RAT-5`,
`NATIVE-CHECK-RAT-6`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

#: The past-tense allomorph tag, as it appears in a ThamizhiMorph analysis string.
IN_CLASS = "+past=இன்"

#: Surface shape of the `-இன்` past stem immediately before a vowel-initial PNG suffix:
#: `ஓடு` -> `ஓடின`(+து). U+0BBF is the vowel sign ி, U+0BA9 is ன.
_IN_STEM_TAIL = "ின"


def _in_class_neuter_singular(stem_surface: str) -> str:
    """`ஓடின` -> `ஓடியது`. The cell is built on the adjectival participle `-இய`, plus `து`."""
    if not stem_surface.endswith(_IN_STEM_TAIL):
        raise ValueError(f"not an -இன் past stem: {stem_surface!r}")
    return stem_surface[:-1] + "யது"          # ...ி + ய + து


def _in_class_irrational_plural(stem_surface: str) -> str:
    """`ஓடின`(+`ன`) -> `ஓடின`. The அஃறிணை PLURAL of this class IS the bare past stem.

    The transducer builds it by concatenating the `-அன` PNG suffix onto the past stem, which
    is right for `-த்`/`-ட்`/`-ன்ற்`/`-த்த்` (`வந்த`+`ன` = `வந்தன`) and wrong here: `-இன்`
    already carries the exponent. Hence `ஓடினன` n=4 against `ஓடின` n=3 585, sentence-final,
    summed over all 20 lemmas of the class.
    """
    if not stem_surface.endswith(_IN_STEM_TAIL):
        raise ValueError(f"not an -இன் past stem: {stem_surface!r}")
    return stem_surface


@dataclass(frozen=True)
class Repair:
    """One (past class, bundle) cell that the transducer gets wrong.

    `rewrite is None` means WITHDRAW — the cell is deleted from the paradigm and no
    replacement is admitted, because the corpus-attested form exists but the checker cannot
    assign it the intended features.
    """
    past_class: str                       # analysis substring identifying the class
    bundle: str
    rewrite: Callable[[str], str] | None  # past-stem surface -> repaired surface
    why: str
    native_check: str
    #: (FST form, its finite corpus count, repaired form, its finite corpus count) summed
    #: over every lemma of the class in `data/lexicon/verbs.yaml`, measured 2026-08-08.
    evidence: str
    #: True when the repaired form has NO separable PNG exponent — the marker is fused into
    #: the tense morph. The span is then recorded as zero-width at the end of the word rather
    #: than pointing at a character that belongs to another morpheme.
    zero_exponent: bool = False


REPAIRS: tuple[Repair, ...] = (
    Repair(
        past_class=IN_CLASS, bundle="past.3sg.neut",
        rewrite=_in_class_neuter_singular,
        why="the -இன் past class builds its அஃறிணை SINGULAR on the past adjectival "
            "participle -இய plus து (ஓடியது), not on the past stem plus -அது (ஓடினது)",
        native_check="NATIVE-CHECK-RAT-4",
        evidence="20/20 lemmas favour the repair; sentence-final tokens 18 (FST) vs "
                 "15379 (repaired) over 105M tokens"),
    Repair(
        past_class=IN_CLASS, bundle="past.3pl.irrat",
        rewrite=_in_class_irrational_plural,
        why="the அஃறிணை PLURAL of this class is the bare past stem (ஓடின, not ஓடினன): -இன் "
            "already carries the exponent, so the transducer's concatenation of -அன onto it "
            "over-generates. ThamizhiMorph analyses the corpus form only as +nonfin+adjpart; "
            "DECISIONS.md D-7.1 rules the form finite and morph/fallback.py's R14 supplies "
            "the missing 3pl-n reading",
        native_check="NATIVE-CHECK-RAT-5",
        evidence="20/20 lemmas favour the repair; sentence-final tokens 4 (FST) vs 3585 "
                 "(repaired) over 105M tokens; with R14 the checker returns "
                 "{NUMBER: PL, GENDER: NEUT, RATIONALITY: AHRINAI} for the repaired form",
        zero_exponent=True),
)


def repair_for(analysis: str, bundle: str) -> Repair | None:
    """The repair that applies to this (analysis, bundle), or None."""
    for r in REPAIRS:
        if r.bundle == bundle and r.past_class in analysis:
            return r
    return None
