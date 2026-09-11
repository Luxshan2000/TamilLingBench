# -*- coding: utf-8 -*-
"""Source ↔ gold feature-consistency audit.

The construction defect described in the paper was not a typo. It was a *class* of bug:

    the Tamil gold encodes a grammatical feature that the English input never specifies.

The paper's first release had two instances. Gender C2/C3 golds are 3**sg** while their source says
*they*, so a plural translation is licit and `AVOIDANT_NEUTRAL` overstates avoidance. The
number C2 SG gold is 3sg **masculine** while its source says "One student … They spoke", so
it requires singular-*they* AND invents a gender. Both are the same failure: a feature in the
key with no exponent in the input.

This module turns that into a check that runs over every (slot × condition) cell on every
build. It is deliberately **lexical and explicit** rather than clever: the marker inventories
below are readable, and a linguist can dispute a row by disputing a word. Nothing here parses
English; it asks whether the input contains any exponent at all of the feature the gold
commits to, which is the weakest question that would still have caught both defects.

Severities:

* ``FAIL`` — the gold commits to a feature value the input never expresses, or expresses only
  in contradiction. The item is not answerable from its input.
* ``WARN`` — answerable, but the input carries a competing exponent (an opposite-gender word
  elsewhere in the sentence) or is otherwise degraded. Reported, not blocking.

Both are returned; the caller decides which are blocking. The released benchmark has zero
FAILs (see `data/benchmark/manifest.json`).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- inventories

#: English 3pl pronouns. Singular-`they` exists, but it is gender-neutral BY DESIGN, which is
#: precisely why it cannot carry a gender gold, and it is number-plural in form, which is
#: precisely why it cannot carry a singular-number gold.
PLURAL_PRONOUN = re.compile(r"\b(they|them|their|theirs|themselves)\b", re.I)

MASC_WORDS = ("he", "him", "his", "himself", "man", "men", "boy", "boys", "brother",
              "brothers", "father", "son", "sons", "uncle", "grandfather", "nephew",
              "schoolboy", "schoolboys", "king", "kings", "sir", "gentleman", "male",
              "husband", "lad", "headman", "watchman", "widower")
FEM_WORDS = ("she", "her", "hers", "herself", "woman", "women", "girl", "girls", "sister",
             "sisters", "mother", "daughter", "daughters", "aunt", "grandmother", "niece",
             "schoolgirl", "schoolgirls", "queen", "madam", "lady", "female", "wife",
             "widow", "lass", "lassie")

#: Singular / plural exponents for the number control. Closed lists: every one of them is a
#: word that actually occurs in a number item's context or source, so an addition to the
#: template inventory that this audit cannot see is itself caught (as a missing exponent).
#: person-denoting number exponents — the only ones that can antecede a number item's
#: referent, and therefore the only ones whose presence in a C3 distractor is a leak.
PERSON_NUMBER_WORDS = ("man", "men", "student", "students", "soldier", "soldiers",
                       "teacher", "teachers", "doctor", "doctors", "officer", "officers",
                       "king", "kings", "leader", "leaders", "warrior", "warriors",
                       "guard", "guards", "youth", "youths", "schoolboy", "schoolboys",
                       "he", "him", "his", "boy", "boys", "girl", "girls")

#: Number exponents for a non-human (அஃறிணை) referent, whose `-அது` / `-அன` agreement avoids
#: the `-ஆர்` syncretism: an animal or object gloss, and the copula. `was` / `were` earn their
#: place for a number-INVARIANT NP ("the aircraft"), where English marks the number only on
#: the verb of the CONTEXT sentence; without these two words the audit would report a
#: correctly-cued item as unanswerable.
#:
#: The glosses are DERIVED from the noun lexicon rather than hand-written, because a
#: hand-written mirror of a generated file drifts silently. They are merely EXTENDED by hand
#: with glosses that come from context sentences rather than from a lexicon row.
def _number_glosses(key: str) -> tuple[str, ...]:
    rows: list[dict] = []
    for f in sorted((ROOT / "data" / "lexicon").glob("nouns*.yaml")):
        rows += yaml.safe_load(f.read_text(encoding="utf-8"))
    return tuple(sorted({n["en"][key] for n in rows if n["tinai"] == "akrinai"
                         if n["en"][key]}))


_AHRI_SG_GLOSSES = _number_glosses("sg") + ("bird",)
_AHRI_PL_GLOSSES = _number_glosses("pl") + ("birds",)

SG_WORDS = ("one", "single", "lone", "sole", "man", "student", "soldier", "teacher",
            "doctor", "officer", "king", "leader", "warrior", "guard", "youth", "schoolboy",
            "he", "him", "his", "was") + _AHRI_SG_GLOSSES
PL_WORDS = ("men", "students", "soldiers", "teachers", "doctors", "officers", "kings",
            "leaders", "warriors", "guards", "youths", "schoolboys", "two", "three", "four",
            "five", "six", "seven", "eight", "nine", "ten", "twelve", "twenty", "dozen",
            "several", "many", "few", "both", "all", "some", "others",
            "were") + _AHRI_PL_GLOSSES

#: Rationality: உயர்திணை needs a human exponent, அஃறிணை a non-human one. The pinned plural
#: means "the dogs" / "the students" is the whole cue.
#: ⛔ 2026-08-08. These two lists used to be hand-maintained, and were **silently
#: incomplete**: `father`, `grandmother`, `madam`, `sir`, `younger brother` and `younger
#: sister` are all `tinai: uyartinai` in `data/lexicon/nouns.yaml` and none of them was
#: listed. Nothing failed only because the filler sampler had not reached those six nouns;
#: once it did, 54 perfectly good items were reported as
#: `gold_rationality_unsupported`. A hand-written mirror of a generated file drifts, and the
#: drift only shows up when something else moves. So the lists are now DERIVED from the
#: lexicon's own `tinai` annotation and merely *extended* by hand with the glosses that come
#: from context sentences and cue tables rather than from `nouns.yaml`.
#: `tinai: contested` nouns (child / team / company / squad) are deliberately in NEITHER:
#: they are the contested stratum, and an item built on one is not supposed to be decidable
#: from the English.
#: Every `nouns*.yaml` inventory is read, not one fixed filename, so an added lexicon file
#: cannot reintroduce the drift.
def _glosses(tinai: str) -> tuple[str, ...]:
    rows: list[dict] = []
    for f in sorted((ROOT / "data" / "lexicon").glob("nouns*.yaml")):
        rows += yaml.safe_load(f.read_text(encoding="utf-8"))
    return tuple(sorted({w for n in rows if n["tinai"] == tinai
                         for w in (n["en"]["sg"], n["en"]["pl"]) if w}))


_UYAR_EXTRA = ("teacher", "teachers", "officer", "officers", "man", "men", "woman", "women",
               "boy", "boys", "girl", "girls", "people", "villager", "villagers")
_AHRI_EXTRA = ("cattle", "sheep", "buffalo", "livestock", "bird", "birds")

UYAR_WORDS = tuple(sorted(set(_glosses("uyartinai")) | set(_UYAR_EXTRA)))
AHRI_WORDS = tuple(sorted(set(_glosses("akrinai")) | set(_AHRI_EXTRA)))

FIRST_PL = re.compile(r"\b(we|us|our|ours|ourselves)\b", re.I)
#: the clusivity cue table realises 1PL periphrastically — "my brother and me", "two of
#: my friends and me" — so a bare 1sg pronoun beside another participant counts.
FIRST_SG = re.compile(r"\b(I|me|myself)\b")
SECOND = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.I)


def _has(text: str, words: Iterable[str]) -> list[str]:
    low = text.lower()
    return [w for w in words if re.search(rf"\b{re.escape(w)}\b", low)]


# --------------------------------------------------------------------------- findings

@dataclass(frozen=True)
class Finding:
    item_id: str
    slot: str
    condition: str
    check: str
    severity: str            # FAIL | WARN
    detail: str

    def key(self) -> tuple[str, str, str, str]:
        return (self.slot, self.condition, self.check, self.severity)


def _input_text(item: dict) -> str:
    """Everything the system is shown, minus the instruction.

    C0 states the required value in the *instruction*, so its instruction is
    a legitimate exponent and is included; for C1/C2/C3 there is none.
    """
    return " ".join(x for x in (item.get("source_context"),
                                (item.get("fillers") or {}).get("c0_instruction"),
                                item.get("source")) if x)


# --------------------------------------------------------------------------- checks

def _check_plural_pronoun(item: dict) -> list[Finding]:
    """A 3pl pronoun in the sentence being translated, with a singular-committed gold.

    This is the head of the first-release defect described in the paper. `referent_number` is the item's own declaration
    (DECISIONS.md D-2); for the number slot it is null on purpose, so the gold value is used
    instead, and a C3 item is judged against the singular member of its contrast set —
    a C3 whose outcome space contains a singular form is not cue-*absent* if the source
    rules the singular out.
    """
    src = item.get("source") or ""
    if not PLURAL_PRONOUN.search(src):
        return []
    if item["slot"] == "number":
        vals = ([item["gold_value"]] if item.get("gold_value")
                else [t.get("value") for t in item.get("contrast_targets") or []])
        if "SG" not in vals:
            return []
        why = f"gold_value={item['gold_value']}" if item.get("gold_value") \
            else "C3 outcome space contains SG"
    else:
        if item.get("referent_number") != "SG":
            return []
        why = "referent_number=SG"
    return [Finding(item["item_id"], item["slot"], item["condition"],
                    "plural_pronoun_singular_referent", "FAIL",
                    f"source realises the referent as a 3pl pronoun "
                    f"({PLURAL_PRONOUN.search(src).group(0)!r}) but {why}; a plural "
                    f"translation is a licit reading of the source")]


def _gold_encodes_gender(item: dict) -> str | None:
    """MASC / FEM when the reference Tamil form commits to a gender, else None.

    Tamil marks gender only in the 3sg உயர்திணை non-honorific verb, so:
    the gender slot always, the number slot on its SG arm (`-ஆன்`), and nowhere else —
    `-ஆர்கள்`, `-அன`, `-ஓம்`, `-ஆய்`, `-ஈர்கள்` are all genderless.
    """
    if item["slot"] == "gender":
        return item.get("gold_value")
    if item["slot"] == "number" and item.get("gold_value") == "SG":
        g = (item.get("holds_constant") or {}).get("gender")
        return {"masc": "MASC", "fem": "FEM"}.get(g)
    return None


def _c0_states_the_value(item: dict) -> bool:
    """C0 names the required value literally in the prompt instruction,
    so the feature IS specified — by the instruction rather than by the sentence."""
    instr = (item.get("fillers") or {}).get("c0_instruction") or ""
    return bool(item.get("gold_value")) and item["gold_value"] in instr


def _check_gender(item: dict) -> list[Finding]:
    want = _gold_encodes_gender(item)
    if want is None or _c0_states_the_value(item):
        return []
    text = _input_text(item)
    mine = _has(text, MASC_WORDS if want == "MASC" else FEM_WORDS)
    other = _has(text, FEM_WORDS if want == "MASC" else MASC_WORDS)
    # the C1 cue for the gender slot is a NAME, whose gender is carried by the name itself
    named = _cue_name(item)
    out = []
    if not mine and not named:
        # FAIL when gender is the SCORED slot — the item is then unanswerable. WARN when
        # gender is a held-constant nuisance dimension (the number control pins MASC only
        # because Tamil 3sg -HON has no gender-free member): the checker scores Slot.NUMBER
        # and never looks at gender, so the item is answerable and it is the REFERENCE
        # STRING, not the verdict, that overcommits.
        sev = "FAIL" if item["slot"] == "gender" else "WARN"
        out.append(Finding(
            item["item_id"], item["slot"], item["condition"], "gold_gender_unsupported",
            sev, f"gold commits to {want} but the input contains no {want} exponent "
                 f"(searched context + instruction + source)"
                 + ("" if sev == "FAIL" else "; gender is not the scored feature here, so "
                    "this is reference over-commitment, not unanswerability")))
    elif other:
        out.append(Finding(
            item["item_id"], item["slot"], item["condition"],
            "gender_cue_has_competing_exponent", "WARN",
            f"gold is {want}; the input also contains {sorted(other)}"))
    return out


def _cue_name(item: dict) -> str | None:
    """The romanised personal name the item's own C1 cue uses, when it uses one."""
    f = item.get("fillers") or {}
    return f.get("name_id") if item.get("cue_location") == "in_sentence" else None


def _check_number(item: dict) -> list[Finding]:
    if item["slot"] != "number" or not item.get("gold_value") \
            or _c0_states_the_value(item):
        return []
    want = item["gold_value"]
    text = _input_text(item)
    mine = _has(text, SG_WORDS if want == "SG" else PL_WORDS)
    if not mine:
        return [Finding(item["item_id"], item["slot"], item["condition"],
                        "gold_number_unsupported", "FAIL",
                        f"gold commits to {want} but the input contains no {want} exponent")]
    return []


def _check_rationality(item: dict) -> list[Finding]:
    if item["slot"] != "rationality" or not item.get("gold_value") \
            or _c0_states_the_value(item):
        return []
    want = item["gold_value"]
    text = _input_text(item)
    mine = _has(text, UYAR_WORDS if want == "UYAR" else AHRI_WORDS)
    if not mine:
        return [Finding(item["item_id"], item["slot"], item["condition"],
                        "gold_rationality_unsupported", "FAIL",
                        f"gold commits to {want} but the input names no "
                        f"{'human' if want == 'UYAR' else 'non-human'} referent")]
    return []


def _check_person(item: dict) -> list[Finding]:
    hc = item.get("holds_constant") or {}
    text = _input_text(item)
    p = hc.get("person")
    if p == 1 and not FIRST_PL.search(text) and not FIRST_SG.search(text):
        return [Finding(item["item_id"], item["slot"], item["condition"],
                        "gold_person_unsupported", "FAIL",
                        "gold is 1pl but the input contains no 1st-person plural exponent")]
    if p == 2 and not SECOND.search(text) and item.get("slot_family") != "HON-I":
        # HON-I is exempt: an English imperative has no overt addressee, which is the point
        # of the family.
        return [Finding(item["item_id"], item["slot"], item["condition"],
                        "gold_person_unsupported", "FAIL",
                        "gold is 2nd person but the input contains no addressee exponent")]
    return []


def _check_gold_name(item: dict, names: dict[str, tuple[str, str]]) -> list[Finding]:
    """Every personal name spelled in the Tamil reference must occur in the English input.

    A gold that names a subject the source never names is the same failure as a gold that
    marks a feature the source never marks: the reference is not a translation of its own
    source. This is what catches the `cue_table` gender families, whose C1 source says
    "the young nurse, a man" while their Tamil reference spells a proper name.

    `names` maps name_id -> (Tamil surface, romanisation). The Tamil side is matched against
    the reference and the romanisation against the English — comparing either one against
    both was the first version of this check and it silently never fired.
    """
    out = []
    for role in ("name_id", "anaphor_id"):
        nid = (item.get("fillers") or {}).get(role)
        if not nid or nid not in names:
            continue
        tamil, rom = names[nid]
        # whole-token match, not substring: the name அஞ்சி is a prefix of the verb
        # அஞ்சினான், and a substring test reports the reference as naming a subject it
        # does not contain.
        used = any(tamil in (tgt.get("tamil") or "").rstrip(".!?").split()
                   for tgt in (item.get("gold_targets") or [])
                   + (item.get("contrast_targets") or []))
        if not used:
            continue
        english = (item.get("source") or "") + " " + (item.get("source_context") or "")
        if rom not in english:
            out.append(Finding(
                item["item_id"], item["slot"], item["condition"],
                "gold_names_a_referent_the_source_does_not", "FAIL",
                f"reference target spells the subject {tamil!r} ({nid}, romanised {rom!r}) "
                f"but neither the source sentence nor its context names it"))
    return out


def _check_c3_cue_leak(item: dict) -> list[Finding]:
    """The C3 distractor must carry no exponent of the slot."""
    if item["condition"] != "C3":
        return []
    ctx = item.get("source_context") or ""
    if item["slot"] == "gender":
        leak = _has(ctx, MASC_WORDS) + _has(ctx, FEM_WORDS)
    elif item["slot"] == "rationality":
        leak = _has(ctx, UYAR_WORDS) + _has(ctx, AHRI_WORDS)
    elif item["slot"] == "number":
        # Only exponents that could ANTECEDE this item's referent: "all night" and "a single
        # wooden bench" quantify something that is not the referent and cannot bind to it.
        # Which words those are depends on the referent's திணை — person-denoting for an
        # உயர்திணை referent, non-human for an அஃறிணை one.
        leak = _has(ctx, AHRI_WORDS if (item.get("holds_constant") or {}).get("tinai") == "ahri"
                    else PERSON_NUMBER_WORDS)
    else:
        return []
    if leak:
        return [Finding(item["item_id"], item["slot"], item["condition"],
                        "c3_distractor_leaks_a_cue", "FAIL",
                        f"C3 context contains slot exponents {sorted(set(leak))}")]
    return []


_MALFORMED = re.compile(r"^[\s,.]|\s{2,}|\bThe [a-z]*ed\b(?! )|^The (rose|came|spoke) ")


def _check_malformed_source(item: dict) -> list[Finding]:
    """Purely mechanical English well-formedness, aimed at cue-stripping accidents.

    C0 builds its source by substituting the empty string for `{CUE}`, which
    leaves ` came at noon.` for a name-cued frame and `The rose outside the hall.` for an
    NP-cued one. Neither is English, so a C0 failure cannot be read as a capability ceiling.
    """
    src = item.get("source") or ""
    bad = []
    if src[:1] in (" ", ",", "."):
        bad.append("starts with punctuation or whitespace")
    if "  " in src:
        bad.append("double space (an empty substitution)")
    if re.match(r"^The (rose|came|spoke|did|arrived|left|ran)\b", src):
        bad.append("determiner with no head noun")
    if not bad:
        return []
    return [Finding(item["item_id"], item["slot"], item["condition"], "malformed_source",
                    "FAIL", f"{src!r}: " + "; ".join(bad))]


# --------------------------------------------------------------------------- entry point

def audit(items: Iterable[dict],
          names: dict[str, tuple[str, str]] | None = None) -> list[Finding]:
    """Run every check. `names` maps name_id -> (Tamil surface, romanisation)."""
    names = names or {}
    out: list[Finding] = []
    for it in items:
        out += _check_plural_pronoun(it)
        out += _check_gender(it)
        out += _check_number(it)
        out += _check_rationality(it)
        out += _check_person(it)
        out += _check_gold_name(it, names)
        out += _check_c3_cue_leak(it)
        out += _check_malformed_source(it)
    return out


def summarise(findings: Iterable[Finding]) -> dict[tuple[str, str, str, str], int]:
    from collections import Counter
    return dict(Counter(f.key() for f in findings))


def name_romanisations(
        names_rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """`{name_id: (Tamil, 'Akaran')}`, via the generator's own romanisation path."""
    return {n["id"]: (n["tamil"], _ascii_fold(_iso(n["tamil"])).capitalize())
            for n in names_rows}


def _iso(s: str) -> str:
    """ISO 15919 romanisation (the same path the item builder used)."""
    from aksharamukha import transliterate as tr
    return tr.process("Tamil", "ISO", s)


def _ascii_fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("ḵ", "k")
