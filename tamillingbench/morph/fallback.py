# -*- coding: utf-8 -*-
"""Fallback and override rules R1–R10.

Two jobs, and conflating them is a known source of bugs:

  * FALLBACK (R1–R4, R7) fills gaps — forms the FST returns `+?` for.
  * OVERRIDE (R5, R6, R9) corrects forms the FST analyses *wrongly*. A layer that only
    fills `+?` never sees these, because the FST did return something.

Every rule carries a provenance string that lands in `check()`'s `why`.

⚠ EVERY RULE ENCODES A GRAMMATICAL CLAIM. R1's split point, R5's treatment of
அன்று as copular, and R9's honorific reading of தாங்கள் are judgments a Tamil linguist must
sign off on before the benchmark is trusted. This is not a mechanical table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .labels import NEG, PNG_FEATURES, Slot, ZERO_MORPH, label_suffix_to_surface

OVERRIDES_TSV = Path(__file__).with_name("overrides.tsv")


# --------------------------------------------------------------------------- PNG table

#: Person-number-gender suffixes, as LABEL strings (independent vowels, FST convention).
#: A label may license more than one PNG tag — `ஆர்கள்` is the honorific-singular /
#: plural-epicene syncretism that D-2 exists to handle.
#: ORDER IS LOAD-BEARING: longest first, so ஆர்கள் wins over ஆர் and அனர் over அன.
PNG_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ஆர்கள்", ("3sghe", "3ple")),
    ("ஈர்கள்", ("2plh",)),
    ("அனர்", ("3ple",)),
    ("ஆன்", ("3sgm",)),
    ("ஆள்", ("3sgf",)),
    ("ஆர்", ("3sghe",)),
    ("ஏன்", ("1sg",)),
    ("ஓம்", ("1pl",)),
    ("ஆய்", ("2sg",)),
    ("ஆது", ("3sgn",)),
    ("அது", ("3sgn",)),
    ("அன", ("3pln",)),
    ("ஆ", ("3sgn",)),
)

#: R7 — irregular past stems the FST has no entry for. Key is the stem AS IT SURFACES;
#: value is (lemma, tense-binding label, tense-binding suffix).
#: Sourced from the measured miss lists, not invented: உள்ளன, உள்ளனர்,
#: உள்ளார், என்றார், என்றனர், ஆனான், சொன்னான், போனான், கொடுத்தார், தந்தார் are all in the
#: "genuine analyser gap" bucket and all high-frequency.
#: ORDER IS LOAD-BEARING: longest stem first (கொடுத்த before தந்த would be fine, but
#: செய்த vs த must not shadow).
IRREGULAR_STEMS: tuple[tuple[str, str, str, str], ...] = (
    ("சொன்ன", "சொல்", "past", "ன்ன"),
    ("கொடுத்த", "கொடு", "past", "த்த"),
    ("சாப்பிட்ட", "சாப்பிடு", "past", "ட்ட"),
    ("காப்பாற்றின", "காப்பாற்று", "past", "இன"),
    ("சந்தோஷப்பட்ட", "சந்தோஷப்படு", "past", "ட்ட"),
    ("கூப்பிட்ட", "கூப்பிடு", "past", "ட்ட"),
    ("தூங்கின", "தூங்கு", "past", "இன"),
    ("கேட்கிற", "கேள்", "pres", "கிற்"),
    ("கொடுப்ப", "கொடு", "fut", "ப்ப்"),
    ("உள்ள", "உள்", "pres", ""),
    ("வந்த", "வா", "past", "த்"),   # ThamizhiMorph HAS வந்தான்/வந்தார்கள் but NOT வந்தார்
    ("என்ற", "என்", "past", "ன்ற"),
    ("நின்ற", "நில்", "past", "ன்ற"),
    ("இருந்த", "இரு", "past", "ந்த"),
    ("விட்ட", "விடு", "past", "ட்ட"),
    ("தந்த", "தா", "past", "ந்த"),
    ("போன", "போ", "past", "ன"),
    ("ஆன", "ஆகு", "past", "ன"),
)

#: R11 — enclitics that attach after a finite verb and block FST lookup. Stripping one
#: restores the host word's final pulli: வந்தானா → வந்தான் + ஆ(interrogative).
#: Value is the clitic's label. Five of the seven unanalysed MWTT masculine forms were
#: clitic-suffixed (வந்தானா, வந்தானோ, வந்தானே, ஜெயிக்கிறானோ, வருவானோ), so this is not a
#: long-tail rule — it is a systematic gap in ThamizhiMorph's finite-verb nets.
CLITICS: dict[str, str] = {
    "ா": "q",        # interrogative -ஆ
    "ோ": "dub",      # dubitative / disjunctive -ஓ
    "ே": "emph",     # emphatic -ஏ
    "ும்": "incl",   # concessive/additive -உம்
}

#: R12 — auxiliary stems that carry the PNG in a compound verb (போய்விட்டான் = போய் + விட்டான்).
#: Restricted to a closed list on purpose: a general "does any suffix substring analyse?"
#: split would manufacture false positives, and precision is the acceptance metric here.
COMPOUND_AUXILIARIES: tuple[str, ...] = (
    "விட்", "விடு", "கொண்", "கொள்", "போட்", "இரு", "வை", "ஆகு", "ஆயி", "முடி", "தள்ளு",
)

#: R1 — the மாட்ட- negative future infix. Split at the LAST occurrence; the material before
#: it is the infinitive stem (போகமாட்டான் → போக + மாட்ட + ான்).
NEGFUT_INFIX = "மாட்ட"

# ------------------------------------------------- R14: the -இன் class அஃறிணை plural (D-7.1)
#
# ⛔ THE EIGHTH TOOLING DEFECT, and the third where the tool rather than the data was wrong.
#
# DECISIONS.md **D-7.1** (native-speaker ruling): *"நாய்கள் ஓடின. is a complete
# sentence"* — `ஓடின` IS a finite verb, the அஃறிணை PLURAL of the `-இன்` past class. The
# corpus agrees overwhelmingly: summed over the 20 `-இன்`-class lemmas in the lexicon and
# counting only SENTENCE-FINAL tokens, the transducer's concatenated `ஓடினன` has **4** and
# `ஓடின` has **3 585**.
#
# ThamizhiMorph returns exactly one analysis for it:
#
#     ஓடின  ->  ஓடு+verb+nonfin+sim+past=இன்+adjpart=அ   ->  {POLARITY: POS}
#
# no NUMBER, no RATIONALITY. `MorphChecker.check` counts a reading that leaves the slot
# undefined as non-matching — its documented and correct policy — so **a model that produces
# the correct Tamil `நாய்கள் ஓடின` is scored WRONG**. Repairing the gold string without
# repairing the analyser would move the error, not remove it; this rule repairs the
# analyser.
#
# THE CLAIM THIS RULE MAKES, stated so a native speaker can reject it:
#
#   In the `-இன்` past class, and ONLY in that class, the past adjectival participle and the
#   அஃறிணை-plural finite form are HOMOPHONOUS. `ஓடின` is genuinely ambiguous between
#   `ஓடு+nonfin+adjpart` ("that ran", modifying a following noun) and
#   `ஓடு+fin+past+3pln` ("(they, non-human) ran"). The rule ADDS the finite reading and
#   KEEPS the participle analysis.
#
# WHY IT CANNOT OVER-FIRE ONTO OTHER CLASSES. The homophony is a property of this class
# alone, because every other past class realises 3pl-n with an overt `-அன` that the
# participle lacks: `வந்த`(adjpart) vs `வந்தன`(3pl-n), `நடந்த` vs `நடந்தன`, `சென்ற` vs
# `சென்றன`. Measured on the shipped nets, `வந்த`/`நடந்த`/`சென்ற`/`பார்த்த`/`செய்த` carry
# `+past=த்` / `+past=ற்` and are untouched by the guard below.
#
# ⚠ THE COST, stated once and measured rather than asserted away. `_assemble` applies the
# feature bundle through `_forced`, which merges into EVERY reading of the surface — exactly
# as R13 does for `-வது`/`-யது`. So at the FEATURE layer the participle reading is subsumed:
# `ஓடின` reports {PL, NEUT, AHRINAI} whatever its position. That is right where the checker
# is actually consulted — `Builder._verify` reads the SENTENCE-FINAL deciding word and
# `eval.extract` restricts the verb-final slots to the RIGHTMOST committed token, and a
# Tamil adjectival participle can never be either, since it must be followed by the noun it
# modifies. It is wrong for a mid-sentence `அவரு எழுதின கட்டுரை`, and `எழுதின` is only 21 %
# sentence-final (§000.2), so the exposure is real and is reported in the ablation rather
# than hidden. `MorphChecker(repair_in_class=False)` reproduces the unrepaired behaviour.
#
# ⚠ GRAMMATICAL CLAIM — `NATIVE-CHECK-RAT-6`. D-7.1 rules the finite reading licit; that the
# participle reading survives alongside it is our own claim about the ambiguity.

#: The `-இன்` past allomorph, as it appears in a ThamizhiMorph analysis string.
IN_CLASS_PAST = "+past=இன்"

#: The adjectival-participle tag, as it appears bound in an analysis string.
ADJPART = "+adjpart="

#: The bundle the added reading commits to. `3pln` is what `labels.PNG_FEATURES` maps to
#: {NUMBER: PL, GENDER: NEUT, RATIONALITY: AHRINAI}; the table itself is NOT touched.
IN_CLASS_FINITE_PNG = "3pln"


def rule_in_class_finite(surface: str, raw_analyses: Sequence[str]) -> RuleHit | None:
    """R14 — add the finite அஃறிணை-plural reading of an `-இன்`-class `-இன` form (D-7.1).

    Unlike R1–R7 this is an AUGMENT, not a fallback: the FST *did* answer, and answered
    with a reading that is correct but incomplete. It therefore takes the raw analyses and
    is called from `MorphChecker._assemble`, next to R13, rather than from
    `apply_fallback` (which only ever runs when the FST returned nothing).

    Fires only when EVERY analysis of the surface is an `-இன்`-class past adjectival
    participle. If any analysis already carries a PNG label the form is not the ambiguous
    one and the rule stands down, so it can never overturn a reading the FST made.
    """
    if not raw_analyses:
        return None
    if not all(IN_CLASS_PAST in a and ADJPART in a for a in raw_analyses):
        return None
    if any(f"+{png}" in a for a in raw_analyses for png in PNG_FEATURES):
        return None
    lemma = raw_analyses[0].split("+", 1)[0]
    return RuleHit(
        lemma=lemma, pos="verb", tags=("fin", "sim", "past"),
        # The அஃறிணை-plural exponent of this class is FUSED into the past marker `-இன`
        # (historically `-இன்` + `-அ(ன)`), so there is no separable PNG string to point at.
        # It is recorded as a zero exponent rather than being made up.
        bindings=(("past", "இன்"), (IN_CLASS_FINITE_PNG, ZERO_MORPH)),
        features=dict(PNG_FEATURES[IN_CLASS_FINITE_PNG][0]),
        rule="R14",
        why=f"R14 (D-7.1, NATIVE-CHECK-RAT-6): {surface!r} is the -இன் class's அஃறிணை "
            f"PLURAL finite form as well as its past adjectival participle; the transducer "
            f"returns only the participle and so assigns no number or rationality")


# --------------------------------------------------------------------------- types

@dataclass(frozen=True)
class RuleHit:
    """A synthetic analysis produced by a rule, in the same shape the FST parser emits."""
    lemma: str
    pos: str
    tags: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]   # (label, raw suffix in FST convention)
    features: dict[Slot, str]
    rule: str                               # 'R1' … 'R10'
    why: str


# --------------------------------------------------------------------------- overrides

def _parse_features(s: str) -> dict[Slot, str]:
    out: dict[Slot, str] = {}
    for part in filter(None, s.split(";")):
        k, v = part.split("=", 1)
        out[Slot(k)] = v
    return out


def _parse_bindings(s: str) -> tuple[tuple[str, str], ...]:
    out = []
    for part in filter(None, s.split(";")):
        label, _, suf = part.partition("=")
        out.append((label, suf))
    return tuple(out)


def load_overrides(path: Path = OVERRIDES_TSV) -> dict[str, list[dict]]:
    """Read overrides.tsv. Kept as data, not code, so native speakers can audit it as a table."""
    rows: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                raise ValueError(f"malformed overrides.tsv row ({len(f)} fields): {line!r}")
            surface, rule, mode, lemma, pos, tags, bindings, features, note = f[:9]
            rows.setdefault(surface, []).append({
                "rule": rule, "mode": mode, "lemma": lemma, "pos": pos,
                "tags": tuple(t for t in tags.split("+") if t),
                "bindings": _parse_bindings(bindings),
                "features": _parse_features(features),
                "note": note,
            })
    return rows


# --------------------------------------------------------------------------- rules

def rule_negfut(surface: str) -> list[RuleHit]:
    """R1 — மாட்ட- negative future. போகமாட்டான் → போக+neg+fut+3sgm=ஆன், polarity=NEG."""
    idx = surface.rfind(NEGFUT_INFIX)
    if idx < 0:
        return []
    tail = surface[idx + len(NEGFUT_INFIX):]
    stem = surface[:idx]
    for label, tags in PNG_SUFFIXES:
        ss = label_suffix_to_surface(label)
        if ss and tail == ss:
            return [
                RuleHit(lemma=stem or "மாட்டு", pos="verb",
                        tags=("fin", "sim", "neg", "fut"),
                        bindings=(("negfut", NEGFUT_INFIX), (t, label)),
                        features={Slot.POLARITY: NEG},
                        rule="R1",
                        why=f"R1 negfut: {stem!r} + மாட்ட + {ss!r} -> +{t}")
                for t in tags
            ]
    return []


def rule_illai(surface: str) -> list[RuleHit]:
    """R2 — standalone இல்லை (and spoken variants)."""
    if surface in ("இல்லை", "இல்ல", "இல்லைன்னு"):
        return [RuleHit("இல்", "verb", ("fin", "negpart"), (("negpart", "இல்லை"),),
                        {Slot.POLARITY: NEG}, "R2", "R2 standalone negative particle இல்லை")]
    return []


def rule_illaamal(surface: str) -> list[RuleHit]:
    """R3 — இல்லாமல் / இல்லாது / இன்றி, the negative verbal participle."""
    table = {"இல்லாமல்": (("neg", "இல்லா"), ("vpart", "மல்")),
             "இல்லாது": (("neg", "இல்லா"), ("vpart", "அது")),
             "இன்றி": (("neg", "இன்று"), ("vpart", "இ"))}
    if surface in table:
        return [RuleHit("இல்", "verb", ("nonfin", "neg"), table[surface],
                        {Slot.POLARITY: NEG}, "R3",
                        f"R3 negative verbal participle {surface}")]
    return []


def rule_vendaam(surface: str) -> list[RuleHit]:
    """R4 — வேண்டாம் / வேண்டாத / வேண்டா, prohibitive/debitive negation."""
    if surface in ("வேண்டாம்", "வேண்டாத", "வேண்டா"):
        return [RuleHit("வேண்டு", "verb", ("fin", "neg", "deb"), (("neg", "ஆ"),),
                        {Slot.POLARITY: NEG}, "R4", f"R4 prohibitive/debitive {surface}")]
    return []


def rule_irregular_stem(surface: str) -> list[RuleHit]:
    """R7 — irregular past/present stems + a regular PNG suffix."""
    hits: list[RuleHit] = []
    for stem, lemma, tense, tsuf in IRREGULAR_STEMS:
        if not surface.startswith(stem):
            continue
        tail = surface[len(stem):]
        for label, tags in PNG_SUFFIXES:
            ss = label_suffix_to_surface(label)
            if tail != ss:
                continue
            for t in tags:
                hits.append(RuleHit(
                    lemma=lemma, pos="verb", tags=("fin", "sim", tense),
                    bindings=((tense, tsuf), (t, label)) if tsuf else ((t, label),),
                    features={}, rule="R7",
                    why=f"R7 irregular stem {stem!r} (lemma {lemma}) + {ss!r} -> +{t}"))
        if hits:
            return hits
    return hits


PULLI = "்"


def strip_clitic(surface: str) -> list[tuple[str, str, str]]:
    """R11 — return [(host, clitic_label, clitic_surface)] candidates, longest clitic first.

    Stripping a vowel-sign clitic restores the host's final pulli, because the clitic's
    vowel displaced it: வந்தான் + ஆ → வந்தானா, so வந்தானா[:-1] + '்' → வந்தான்.
    """
    out = []
    for cl, label in sorted(CLITICS.items(), key=lambda kv: -len(kv[0])):
        if not surface.endswith(cl) or len(surface) <= len(cl) + 1:
            continue
        host = surface[: -len(cl)]
        if host.endswith(PULLI):
            out.append((host, label, cl))          # clitic after a pulli consonant
        else:
            out.append((host + PULLI, label, cl))  # clitic displaced the host's pulli
            # Tamil inserts an epenthetic glide வ்/ய் between a vowel-final host and a
            # vowel-initial clitic: வர + ஆ -> வரவா. Peel the glide too, else the host is
            # recovered as the non-word 'வரவ்'.
            if host and host[-1] in "வய":
                out.append((host[:-1], label, host[-1] + cl))
    return out


def split_compound(surface: str) -> list[tuple[str, str]]:
    """R12 — return [(main_verb_part, auxiliary_part)] at the LAST auxiliary boundary."""
    best = None
    for aux in COMPOUND_AUXILIARIES:
        i = surface.rfind(aux)
        if i > 0 and (best is None or i > best[0]):
            best = (i, aux)
    if best is None:
        return []
    i, _aux = best
    return [(surface[:i], surface[i:])]


#: Ordered; first rule that fires wins.
FALLBACK_RULES = (
    ("R1", rule_negfut),
    ("R2", rule_illai),
    ("R3", rule_illaamal),
    ("R4", rule_vendaam),
    ("R7", rule_irregular_stem),
)


def apply_fallback(surface: str) -> list[RuleHit]:
    """Run the ordered fallback rules. Only called when the FST returned nothing.

    R5/R6/R9 are NOT here — they are overrides, applied before the FST result is accepted.
    R8 (rationality from -அன vs -ஆர்கள்) is subsumed by `labels.PNG_FEATURES`, which maps
    3pln→AHRINAI and 3ple→UYARTHINAI directly; a separate rule would be a second, divergent
    source of truth for the same fact.
    R10 (TTB artifact guard) lives in `normalize.looks_like_split_artifact` +
    `MorphChecker.is_artifact_token`, because it must run before analysis, not after.
    """
    for _rid, fn in FALLBACK_RULES:
        hits = fn(surface)
        if hits:
            return hits
    return []
