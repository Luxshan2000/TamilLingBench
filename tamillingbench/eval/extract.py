# -*- coding: utf-8 -*-
"""Hypothesis → slot outcome, using the EXISTING validated checker (`tamillingbench.morph`).

This module does not re-implement morphology. It does three things the checker deliberately
does not do, because the checker's contract is one *word form*:

  1. **Locus selection.** Decide which token in a whole sentence bears the slot. This is not
     cosmetic: `Slot.HONORIFICITY` is defined on 3rd-person tags too (`3sgm → FAMILIAR`), so a
     naive "any token with an honorificity feature" reader would score an honorificity item off
     an incidental 3rd-person verb. Candidates are therefore filtered by GRAMMATICAL PERSON,
     derived from the same PNG label the checker uses.
  2. **The outcome taxonomy**, first-class and non-collapsing:
     `CORRECT / WRONG_<value> / AVOIDANT_DROP / AVOIDANT_NEUTRAL / AVOIDANT_NEG / UNPARSED`
     (+ `UNDECIDABLE`, which D-2 requires be reported with its own denominator).
     **Avoidance is never folded into "incorrect".**
  3. **D-2 applied to output**: context filter → universal over survivors (headline),
     existential (robustness column), undecidable reported separately.

⛔ Two structural findings are encoded here rather than papered over:

  * **Honorificity is BINARY and a gold value is a GROUP of forms** (DECISIONS.md D-6.1,
    native-speaker ruling). {நீ, நீர்} realises T; {நீங்கள், தாங்கள்} realises V.
    `GOLD_TO_CHECKER` therefore maps a gold value to a *set* of checker values and `judge()`
    tests set membership, so a தாங்கள் output and a நீங்கள் output score identically.
    தாங்கள் still analyses as `+pron+3pl+refl` and still reaches the checker label DEFERENTIAL
    only through the project's own R9 override, which carries `number=PL` — that override is a
    true statement about the surface form and the validated checker is deliberately left
    untouched; the collapse into the polite group happens here, at the scoring layer.
    `VV_EXEMPT_NUMBER_FILTER` remains, because the `+3pl` on தாங்கள் is a reflexive-pronoun
    artifact rather than a claim about the addressee, and every honorificity item declares
    `referent_number=SG`.
    ⚠ The binary is the INDICATIVE only. **D-7.3 rules the IMPERATIVE ternary**:
    வா < வாருங்கள் < வருக, the optative a genuinely higher level. So honorificity holds two
    paradigms of unequal cardinality and **chance is per-family, not per-slot** — HON-V and
    HON-P are k=2 at 50%, HON-I is k=3 at 33.3%. `judge()` takes `slot_family` for exactly
    this reason. Of the 152 shipped VV-gold items, HON-P's 56 are RETRACTED (a second
    spelling of POLITE) while HON-I's 96 are REAL but unscoreable by the checker, which
    licenses no honorificity reading for `opt.def`. Both are `provisional`, for different
    reasons, and the reasons are recorded separately.
  * **AVOIDANT_DROP vs UNPARSED.** These are different facts and the analyser's ~64-75% form
    coverage means they are easy to confuse. A hypothesis is only called AVOIDANT_DROP when the
    parse is good enough to license the claim "the exponent is genuinely absent"; otherwise it
    is UNPARSED. `unparsed_reason` records which.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from ..morph.checker import MorphChecker
from ..morph.labels import PNG_FEATURES, SUFFIX_SYNCRETISM, Slot

TAMIL_RE = re.compile(r"[஀-௿]")
PUNCT = "".join(chr(c) for c in range(0x20, 0x7F) if not chr(c).isalnum()) + "।॥“”‘’—–…"

# --- benchmark gold_value vocabulary → the SET of checker values that realise it ------------
#
# A gold value maps to a *group* of surface realisations, not to a single one. DECISIONS.md
# D-6.1 is the reason: a native speaker ruled that Tamil second-person
# address is BINARY — {நீ, நீர்} familiar against {நீங்கள், தாங்கள்} polite — and that scoring
# may accept EITHER member of a group as realising that value. தாங்கள் still analyses as
# `+pron+3pl+refl` and still reaches the checker label DEFERENTIAL through the R9 override;
# that label is a true fact about the surface form and the validated checker is left alone.
# The collapse happens HERE, at the scoring layer, where it belongs.
GOLD_TO_CHECKER: dict[str, dict[str, frozenset[str]]] = {
    "clusivity":    {"INCL": frozenset({"INCL"}), "EXCL": frozenset({"EXCL"})},
    "gender":       {"MASC": frozenset({"MASC"}), "FEM": frozenset({"FEM"})},
    "honorificity": {
        "T": frozenset({"FAMILIAR"}),                 # நீ, நீர்
        "V": frozenset({"POLITE", "DEFERENTIAL"}),    # நீங்கள், தாங்கள் — one value, two forms
        # `VV` means two different things depending on the family, so the family has to be
        # passed in for it to be judged correctly:
        #   HON-P  — RETRACTED (D-6.1). தாங்கள் is a second spelling of POLITE, so the item
        #            asked for a choice between two ways of writing one answer. Scored as
        #            the polite group and stamped provisional; excluded from headlines.
        #   HON-I  — REAL (D-7.3). வருக is a genuinely higher level than வாருங்கள். The
        #            checker still licenses no honorificity reading for the optative
        #            (`opt.def` carries an empty feature bundle), so it stays provisional —
        #            but as a TOOLING gap, not as a retracted claim. The two must not be
        #            reported as the same kind of caveat.
        "VV": frozenset({"POLITE", "DEFERENTIAL"}),
    },
    "number":       {"SG": frozenset({"SG"}), "PL": frozenset({"PL"})},
    # D-6.2: both -அன (Sri Lankan) and -அது (Indian) are licit அஃறிணை plural agreement.
    # Both already reach AHRINAI in the checker, so no group is needed here — but the
    # variety split this used to hedge on is CLOSED, and the `provisional` flag below is
    # removed accordingly.
    "rationality":  {"UYAR": frozenset({"UYARTHINAI"}), "AHRI": frozenset({"AHRINAI"})},
}

#: Reverse map, for reporting what a hypothesis actually committed to. Where several checker
#: values collapse onto one gold value (honorificity), each maps back to that gold value, so
#: a தாங்கள் output and a நீங்கள் output report as the same commitment — which is the point.
CHECKER_TO_GOLD: dict[str, dict[str, str]] = {
    slot: {cv: gv for gv, cvs in m.items() for cv in cvs
           # `VV` must not shadow `V` in the reverse direction: a DEFERENTIAL surface form
           # reports as the surviving value V, never as the retracted one.
           if not (slot == "honorificity" and gv == "VV")}
    for slot, m in GOLD_TO_CHECKER.items()
}

SLOT_ENUM = {"clusivity": Slot.CLUSIVITY, "gender": Slot.GENDER,
             "honorificity": Slot.HONORIFICITY, "number": Slot.NUMBER,
             "rationality": Slot.RATIONALITY}

#: Which grammatical person may carry the slot. Filtering on this is what stops an incidental
#: 3rd-person verb from being read as an answer to a 2nd-person honorificity item.
SLOT_PERSON: dict[str, tuple[int, ...]] = {
    "clusivity": (1,), "honorificity": (2,),
    "gender": (3,), "rationality": (3,), "number": (3,),
}

#: Values that exist in the checker's space for this slot but are NOT a commitment to a licit
#: gold value — emitting one is avoidance, not an error.
NEUTRALIZING: dict[str, tuple[str, ...]] = {
    "gender": ("EPICENE", "NEUT"),
    "clusivity": (), "honorificity": (), "number": (), "rationality": (),
}

#: Lemmas whose DEFERENTIAL reading survives the referent-number filter. See module docstring.
VV_EXEMPT_LEMMAS = ("தாங்கள்", "தாம்", "தங்கள்")

#: Slots whose exponent is borne by the FINITE VERB. Tamil is verb-final, so for these the
#: matrix verb is the RIGHTMOST token carrying a committed reading, and only that token votes.
#:
#: Measured reason this is not optional: ThamizhiMorph analyses the noun மாணவர் ("student-HON")
#: as `மாண+verb+fin+sim+weak+fut+3sge`, a spurious finite-verb reading with `number=SG`. In
#: `மாணவர் வெளியே பேசினார்` that noun then votes against the actual matrix verb பேசினார் and
#: turns a clean WRONG into a spurious UNDECIDABLE. Clusivity (pronoun-borne) and honorificity
#: (pronoun OR verb, which must agree) are deliberately excluded.
VERB_FINAL_SLOTS = ("gender", "rationality", "number")

#: ⚠ DOCUMENTED CHECKER DEFECT, repaired at the SCORING layer and reported both ways.
#:
#: ThamizhiMorph tags the 3rd-person suffix -அனர் (surface ...னர்: குளித்தனர், விளையாடினர்,
#: பாடினர்) as `3sge`, which `labels.PNG_FEATURES` maps to NUMBER=SG. But -அனர் is the
#: UNAMBIGUOUSLY PLURAL rational form — DECISIONS.md D-5 relies on exactly that property when
#: it defines the `strict_anar` prior variant as "uyartiṇai counted only from the
#: unambiguously-plural -அனர்". Left alone, every plural rendered with -அனர் scores WRONG on
#: the NUMBER control and the control inverts.
#:
#: Measured: a -னர் form appears in 2,744 / 15,724 hypotheses (17.5%).
#:
#: The repair lives HERE, not in `tamillingbench/morph/labels.py`, because that checker is
#: validated and must not be silently altered. `SlotExtractor(repair_anar=False)` reproduces
#: the unrepaired numbers, and both are reported so the size of the effect is visible.
#: ⚠ SECOND DOCUMENTED CHECKER GAP, also repaired at the scoring layer.
#:
#: Tamil sandhi doubles the initial consonant of the following word onto the preceding one:
#: எங்களை + பார்த்தார் -> `எங்களைப் பார்த்தார்`. ThamizhiMorph analyses `எங்களை`
#: (`+pron+1pl+excl+acc`) but on `எங்களைப்` it returns a SPURIOUS VERB reading with no
#: clusivity — so the pronoun is present in the output and invisible to the checker, and the
#: item is scored AVOIDANT_DROP.
#:
#: This is not cosmetic: the clusivity pronoun-drop rate is the headline of the panel-wide
#: gate, and measured before this repair, 135 / 2,489 clusivity AVOIDANT_DROP verdicts
#: contained a 1PL pronoun (எங்களைச் 33, எங்களைப் 28, எங்களைக் 12, நம்மைத் 10, …).
#:
#: Deliberately narrow: the stripped form is consulted ONLY when the original yields no
#: pronoun reading, and its analysis is accepted ONLY if it is a pronoun. It can therefore add
#: a pronoun reading that was invisible; it can never overturn one the checker already made.
SANDHI_FINALS = ("க்", "ச்", "த்", "ப்")
ANAR_SURFACE = "னர்"
ANAR_LABELS = ("3sge", "3sg", "3sgh", "3sghe")
ANAR_REPAIRED_FEATURES = {"number": "PL", "rationality": "UYARTHINAI", "gender": "EPICENE"}

#: `COMMITTED` exists because C3 has NO GOLD BY DESIGN. Labelling a C3 emission "CORRECT"
#: would be a category error — there is nothing for it to be correct about. C3 records the
#: value the system committed to and nothing more.
OUTCOMES = ("CORRECT", "WRONG", "COMMITTED", "UNDECIDABLE", "AVOIDANT_DROP",
            "AVOIDANT_NEUTRAL", "AVOIDANT_NEG", "UNPARSED")
COMMITTED_OUTCOMES = ("CORRECT", "WRONG", "COMMITTED", "UNDECIDABLE")
AVOIDANT_OUTCOMES = ("AVOIDANT_DROP", "AVOIDANT_NEUTRAL", "AVOIDANT_NEG")


# --------------------------------------------------------------------------- helpers

def tokenize(text: str) -> list[tuple[str, int, int]]:
    """(token, char_start, char_end) over the hypothesis as written to the raw JSONL, so
    `hypothesis[start:end] == token` holds and the metric study can slice the string it already has."""
    out = []
    for m in re.finditer(r"\S+", text):
        tok, a, b = m.group(), m.start(), m.end()
        while tok and tok[-1] in PUNCT:
            tok, b = tok[:-1], b - 1
        while tok and tok[0] in PUNCT:
            tok, a = tok[1:], a + 1
        if tok:
            out.append((tok, a, b))
    return out


def tamil_share(s: str) -> float:
    nw = [c for c in s if not c.isspace()]
    return sum(bool(TAMIL_RE.match(c)) for c in nw) / len(nw) if nw else 0.0


def png_labels(analysis) -> tuple[str, ...]:
    """The PNG label(s) this analysis carries, resolved exactly as the checker resolves them
    (suffix-conditioned syncretism first, then bare tags). Duplicated deliberately: the checker
    discards the label when it flattens to feature bundles, and person is what we need."""
    for b in analysis.bindings:
        if b.label in PNG_FEATURES and b.suffix in SUFFIX_SYNCRETISM:
            return SUFFIX_SYNCRETISM[b.suffix]
    for l in [b.label for b in analysis.bindings] + list(analysis.tags):
        if l in PNG_FEATURES:
            return (l,)
    return ()


def person_of(label: str) -> int | None:
    return int(label[0]) if label and label[0] in "123" else None


@dataclass
class Reading:
    token: str
    span: tuple[int, int]
    analysis_raw: str
    source: str
    png: str | None
    person: int | None
    features: dict[str, str]
    surface_morpheme: str | None
    pos: str = ""
    tags: tuple[str, ...] = ()

    @property
    def is_anchor(self) -> bool:
        """A FINITE VERB or a PRONOUN — the only two loci any slot in this benchmark rides on.

        This is the discriminator between `AVOIDANT_DROP` ("we located the predicate and the
        exponent genuinely is not there") and `UNPARSED` ("we never located the predicate, so
        we cannot say"). It replaces an earlier rule based on the share of analysable tokens,
        which was wrong: ThamizhiMorph is a verb+pronoun net, so ordinary nouns (நாய்கள்,
        பொழுதில்) never analyse and that share is essentially never 1.0.
        """
        if self.pos == "pron":
            return True
        if self.pos in ("verb", "aux"):
            return "fin" in self.tags or (self.source in ("fallback-rule", "override")
                                          and self.png is not None)
        return False


@dataclass
class Judgment:
    outcome: str                      # one of OUTCOMES
    emitted_value: str | None         # gold-vocabulary value the system committed to
    #: ⛔ None, NOT False, whenever the item has no gold (all of C3). Accuracy is UNDEFINED
    #: without a gold; a False here would silently become "incorrect" in any aggregate, and a
    #: True would invert the C1→C2→C3 gradient — the exact opposite of the finding.
    universal: bool | None
    existential: bool | None
    undecidable: bool | None
    n_candidate_readings: int
    n_surviving: int
    n_matching: int
    surface_morpheme: str | None
    morpheme_char_span: tuple[int, int] | None
    decision_source: str              # 'fst-lexicon' | 'fst-guesser' | 'fallback-rule' | 'override' | 'abstain'
    why: str
    analysable_rate: float
    tamil_char_share: float
    unparsed_reason: str | None = None
    provisional: bool = False
    provisional_reason: str | None = None
    all_candidate_values: list[str] = field(default_factory=list)
    #: For AVOIDANT_NEUTRAL only: WHICH neutralising form was used. Gender's escape hatch is
    #: not one thing — EPICENE (-ஆர், the honorific/epicene form) and NEUT (3sgn, treating a
    #: human referent as inanimate) are different behaviours with different implications, and
    #: collapsing them would hide the more interesting of the two.
    neutralising_value: str | None = None


# --------------------------------------------------------------------------- extractor

class SlotExtractor:
    """One long-lived `MorphChecker` (persistent flookup) + a per-token reading cache."""

    def __init__(self, checker: MorphChecker | None = None, *,
                 repair_anar: bool = True, repair_sandhi: bool = True) -> None:
        self.checker = checker or MorphChecker()
        self.repair_anar = repair_anar
        self.repair_sandhi = repair_sandhi
        self.repairs: dict[str, int] = {}
        self._cache: dict[str, list[Reading] | None] = {}

    def close(self) -> None:
        self.checker.close()

    # ------------------------------------------------------------------ readings
    def _token_readings(self, tok: str) -> list[Reading]:
        """All (analysis × PNG-syncretism) readings of one token, person-tagged."""
        if tok in self._cache:
            cached = self._cache[tok]
            return list(cached) if cached else []
        try:
            res = self.checker.analyse(tok)
        except Exception:
            self._cache[tok] = []
            return []
        forced = self.checker._forced.get(res.surface, {})
        out: list[Reading] = []
        seen: set = set()
        for a in res.analyses:
            pls = png_labels(a) or (None,)
            base: dict[str, str] = {}
            for f in a.features():
                base = {k.value: v for k, v in f.items()}
                break
            morph = None
            for b in a.bindings:
                if b.label in PNG_FEATURES and not b.is_zero:
                    morph = b.surface_suffix
                    break
            for pl in pls:
                feats = dict(base)
                if pl:
                    # Re-expand this specific label so `person` and `features` agree.
                    for bundle in PNG_FEATURES[pl]:
                        f2 = dict(base)
                        f2.update({k.value: v for k, v in bundle.items()})
                        f2.update({k.value if hasattr(k, "value") else k: v
                                   for k, v in forced.items()})
                        key = (a.raw, pl, tuple(sorted(f2.items())))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(Reading(tok, (0, 0), a.raw, a.source, pl, person_of(pl),
                                           f2, morph, a.pos, a.tags))
                else:
                    feats.update({k.value if hasattr(k, "value") else k: v
                                  for k, v in forced.items()})
                    key = (a.raw, None, tuple(sorted(feats.items())))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Reading(tok, (0, 0), a.raw, a.source, None, None, feats, morph,
                                       a.pos, a.tags))
        if (self.repair_sandhi and not any(r.pos == "pron" for r in out)
                and tok.endswith(SANDHI_FINALS)):
            stripped = tok[:-2]
            if len(stripped) >= 3:
                for r in self._token_readings(stripped):
                    if r.pos == "pron":
                        out.append(Reading(tok, (0, 0), r.analysis_raw + "|SANDHI_REPAIR",
                                           r.source, r.png, r.person, r.features,
                                           r.surface_morpheme, r.pos, r.tags))
                        self.repairs["sandhi_final_consonant"] = \
                            self.repairs.get("sandhi_final_consonant", 0) + 1
        if self.repair_anar and tok.endswith(ANAR_SURFACE):
            fixed = []
            for r in out:
                if r.pos in ("verb", "aux") and (r.png in ANAR_LABELS
                                                 or "fin" in r.tags) and \
                        r.features.get("number") == "SG":
                    f = dict(r.features)
                    f.update(ANAR_REPAIRED_FEATURES)
                    fixed.append(Reading(r.token, r.span, r.analysis_raw + "|ANAR_REPAIR",
                                         r.source, r.png, r.person, f, r.surface_morpheme,
                                         r.pos, r.tags))
                    self.repairs["anar_3sge_to_3pl"] = \
                        self.repairs.get("anar_3sge_to_3pl", 0) + 1
                else:
                    fixed.append(r)
            out = fixed
        self._cache[tok] = out
        return list(out)

    def readings(self, hypothesis: str) -> tuple[list[Reading], float]:
        toks = tokenize(hypothesis)
        tamil_toks = [t for t in toks if TAMIL_RE.search(t[0])]
        out: list[Reading] = []
        n_ok = 0
        for tok, a, b in tamil_toks:
            rs = self._token_readings(tok)
            if rs:
                n_ok += 1
            for r in rs:
                out.append(Reading(tok, (a, b), r.analysis_raw, r.source, r.png, r.person,
                                   r.features, r.surface_morpheme, r.pos, r.tags))
        rate = n_ok / len(tamil_toks) if tamil_toks else 0.0
        return out, rate

    # ------------------------------------------------------------------ judgment
    def judge(self, hypothesis: str, slot: str, gold_value: str | None, *,
              referent_number: str | None = None,
              referent_honorificity: str | None = None,
              third_degree: dict | None = None,
              slot_family: str | None = None) -> Judgment:
        senum = SLOT_ENUM[slot].value
        persons = SLOT_PERSON[slot]
        neutral_vals = NEUTRALIZING[slot]
        licit = {v for vs in GOLD_TO_CHECKER[slot].values() for v in vs}
        ts = tamil_share(hypothesis)
        readings, rate = self.readings(hypothesis)

        # D-6 closed the two standing provisional flags. Honorificity T/V is now a settled
        # binary and rationality's variety split is settled in favour of accepting both
        # realisations, so neither slot is provisional as a whole any more. What remains
        # provisional is narrower and is stamped per item, not per slot.
        prov, prov_why = False, None
        if slot == "honorificity" and gold_value == "VV":
            if slot_family == "HON-I":
                prov, prov_why = True, (
                    "TOOLING GAP, not a retraction: D-7.3 ruled the optative (வருக) a genuine "
                    "third imperative degree, but ThamizhiMorph gives `opt.def` an empty "
                    "feature bundle and licenses no honorificity reading for it, so the "
                    "checker cannot confirm the degree it realises")
            else:
                prov, prov_why = True, (
                    "RETRACTED item family (D-6.1): the indicative is binary, so தாங்கள் is a "
                    "second spelling of POLITE rather than a third value, and this item asked "
                    "for a choice between two ways of writing one answer. Scored as the polite "
                    "group so it does not crash; must not enter headline results")

        has_gold = gold_value is not None
        U = (lambda v: v if has_gold else None)   # accuracy-valued fields are None w/o gold

        if not readings:
            reason = ("no_tamil" if ts < 0.5 else
                      "empty" if not hypothesis.strip() else "no_analysis")
            return Judgment("UNPARSED", None, U(False), U(False), U(False), 0, 0, 0, None, None,
                            "abstain", f"no analysable Tamil token (tamil_share={ts:.2f})",
                            rate, ts, reason, prov, prov_why)

        # --- locus selection: readings of the admissible person that define the slot --------
        def admissible(r: Reading) -> bool:
            if r.person in persons:
                return True
            # Narrow exemption: தாங்கள்/தங்கள் is FST-tagged 3pl+refl but is the deferential
            # 2nd-person address form under the project's own R9 override.
            return (slot == "honorificity"
                    and r.features.get(senum) == "DEFERENTIAL"
                    and any(l in r.analysis_raw for l in VV_EXEMPT_LEMMAS))

        pool = [r for r in readings if admissible(r)]
        committed = [r for r in pool if r.features.get(senum) in licit]
        neutral = [r for r in pool if r.features.get(senum) in neutral_vals]
        if slot in VERB_FINAL_SLOTS and (committed or neutral):
            # Tamil is verb-final: only the RIGHTMOST committed/neutral token votes, so a
            # spuriously verb-analysed subject noun cannot outvote the matrix verb.
            last = max(r.span[0] for r in (committed or neutral))
            committed = [r for r in committed if r.span[0] == last]
            neutral = [r for r in neutral if r.span[0] == last]

        if not committed:
            # order matters: a neutralising form is a DIFFERENT fact from a dropped exponent
            if neutral:
                r = neutral[0]
                return Judgment("AVOIDANT_NEUTRAL", None, U(False), U(False), U(False),
                                len(pool), 0, 0, r.surface_morpheme,
                                r.span if r.surface_morpheme else None, r.source,
                                f"neutralising form {r.token!r} → {senum}="
                                f"{r.features.get(senum)}", rate, ts, None, prov, prov_why,
                                [], r.features.get(senum))
            anchors = [r for r in readings if r.is_anchor]
            if not anchors:
                # We never located a finite verb or a pronoun, so we cannot distinguish an
                # absent exponent from an analyser miss. Abstain rather than guess.
                return Judgment("UNPARSED", None, U(False), U(False), U(False), len(pool), 0, 0, None,
                                None, "abstain",
                                f"no finite verb or pronoun located "
                                f"(analysable_rate={rate:.2f}); cannot distinguish an absent "
                                f"exponent from an analyser miss", rate, ts, "no_anchor",
                                prov, prov_why)
            pos_anchor = [r for r in anchors if r.features.get("polarity") != "NEG"
                          and r.pos in ("verb", "aux")]
            neg_anchor = [r for r in anchors if r.features.get("polarity") == "NEG"
                          and r.pos in ("verb", "aux")]
            if neg_anchor and not pos_anchor:
                r = neg_anchor[0]
                return Judgment("AVOIDANT_NEG", None, U(False), U(False), U(False), len(pool), 0, 0,
                                None, None, r.source,
                                f"negative rendering {r.token!r} ({r.analysis_raw}); Tamil "
                                f"negation strips the PNG exponent", rate, ts, None,
                                prov, prov_why)
            # A finite verb or pronoun IS present and carries no value for this slot:
            # pro-drop, nominalisation, or a form that simply does not encode the feature.
            r = anchors[0]
            return Judgment("AVOIDANT_DROP", None, U(False), U(False), U(False), len(pool), 0, 0,
                            None, None, r.source,
                            f"anchor {r.token!r} ({r.analysis_raw}) present but no {senum} "
                            f"exponent on any person-{persons} form", rate, ts, None,
                            prov, prov_why)

        # --- D-2: context filter, then universal over survivors ----------------------------
        survivors = committed
        applied = []
        if referent_number:
            def keep_num(r: Reading) -> bool:
                if (slot == "honorificity" and r.features.get(senum) == "DEFERENTIAL"
                        and any(l in r.analysis_raw for l in VV_EXEMPT_LEMMAS)):
                    return True          # VV exemption, documented in the module docstring
                return r.features.get("number") in (None, referent_number)
            survivors = [r for r in survivors if keep_num(r)]
            applied.append(f"referent_number={referent_number}")
        if referent_honorificity:
            survivors = [r for r in survivors
                         if r.features.get("honorificity") in (None, referent_honorificity)]
            applied.append(f"referent_honorificity={referent_honorificity}")
        if not survivors:                # filter deleted everything → fall back, and say so
            survivors = committed
            applied.append("FILTER_VACUOUS_reverted")

        vals = [r.features.get(senum) for r in survivors]
        gold_vals = sorted({CHECKER_TO_GOLD[slot][v] for v in vals if v in CHECKER_TO_GOLD[slot]})
        r0 = survivors[0]

        if gold_value is None:                      # C3 — no gold by design
            emitted = gold_vals[0] if len(set(gold_vals)) == 1 else None
            # No gold by design: record WHAT was committed to and nothing more. universal /
            # existential / undecidable are None because they are undefined here.
            return Judgment("COMMITTED" if emitted else "UNDECIDABLE", emitted,
                            None, None, None, len(committed),
                            len(survivors), len(survivors), r0.surface_morpheme,
                            r0.span if r0.surface_morpheme else None, r0.source,
                            f"C3 (no gold); {', '.join(applied) or 'no filter'}; "
                            f"emitted={gold_vals}", rate, ts, None, prov, prov_why,
                            gold_vals)

        # D-6.1: a gold value is realised by ANY member of its group, so this is a set
        # membership test, not an equality test. நீங்கள் and தாங்கள் both realise V.
        wants = GOLD_TO_CHECKER[slot][gold_value]
        matching = [r for r in survivors if r.features.get(senum) in wants]
        universal = len(matching) == len(survivors)
        existential = len(matching) > 0
        undecidable = existential and not universal
        outcome = ("CORRECT" if universal else "UNDECIDABLE" if undecidable else "WRONG")
        emitted = (gold_value if universal
                   else (gold_vals[0] if len(set(gold_vals)) == 1 else None))
        pick = (matching or survivors)[0]
        return Judgment(outcome, emitted, universal, existential, undecidable,
                        len(committed), len(survivors), len(matching),
                        pick.surface_morpheme, pick.span if pick.surface_morpheme else None,
                        pick.source,
                        f"{outcome}: {senum}∈{{{','.join(sorted(wants))}}}; "
                        f"{', '.join(applied) or 'no filter'}; "
                        f"{len(matching)}/{len(survivors)} surviving readings match; "
                        f"emitted={gold_vals}", rate, ts, None, prov, prov_why, gold_vals)


def context_filters(item: dict) -> tuple[str | None, str | None]:
    """Which pinned cell is available as the D-2 context filter.

    Honorificity/clusivity filter on the declared referent NUMBER. Number/gender/rationality
    cannot — filtering a number item on number is circular — so they filter on the
    honorificity the design pins on every one of their templates (`hon: minus`). This is
    exactly the split `MorphChecker.check`'s docstring describes.
    """
    slot = item["slot"]
    if slot in ("honorificity", "clusivity"):
        return item.get("referent_number"), None
    hon = ((item.get("holds_constant") or {}).get("hon")
           or (item.get("gold_features") or {}).get("hon"))
    return None, {"minus": "FAMILIAR", "plus": "POLITE"}.get(hon)
