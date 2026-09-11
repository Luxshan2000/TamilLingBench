# -*- coding: utf-8 -*-
"""The counting rule — how a corpus token becomes a slot value.

This is the part reviewers will attack, so every choice below is written out.

### Why the value is read off the SURFACE SUFFIX and not off the FST's PNG label

ThamizhiMorph's PNG labels for the -ஆர் family are **demonstrably inconsistent**. Measured
on the shipped nets on 2026-08-08:

    செய்தார்   -> +3sghe=ஆர்      (honorific singular)
    பிறந்தார்  -> +3ple=ஆர்       (rational plural)
    கூறினர்    -> +3sge=அர்       (labelled SINGULAR for a plural suffix)
    வந்தனர்    -> +3ple=அனர்      (via fallback R7)
    வந்தார்    -> no PNG at all from the FST; supplied by fallback R7

The label an item receives therefore depends on which verb happens to be in the lexicon,
not on the grammar (the checker validation records the same finding). If the prior were counted by
label, the rationality and gender rates would be an artifact of FST lexicon coverage.

So the rule is a **two-key** rule:

  * the **FST decides verb-hood and finiteness** — `pos ∈ {verb, aux}` and `fin ∈ tags`.
    This is what stops `நாள்` ('day') and `மகான்` from being counted as verbs;
  * the **surface suffix decides the value**, gated by a coarse label class (2nd vs 3rd
    person, neuter vs epicene) that ThamizhiMorph *is* consistent about.

### Which analyses count — and why `endswith` is not enough

A type counts for a value only when some analysis **binds a PNG label to the matching
surface suffix**: `+3sgm=ஆன்` on a form ending in -ஆன், not merely "the FST called this a
finite verb and the string ends in -ஆன்". Only `fst-lexicon`, `fallback-rule` and
`override` emit bindings; the guesser emits bare labels (Trap A).

⛔ MEASURED CONSEQUENCE, and it is why the loose rule was abandoned. Counting
"finite-verb-by-any-source + `endswith`" over Tamil Wikipedia produced, as the top MASC
`-ஆன்` types: **பாக்கித்தான்** (Pakistan, 4,024), **சுல்தான்** (sultan, 3,160),
**இராசத்தான்** (Rajasthan), **யோவான்** (John) — the guesser hallucinates a finite-verb
reading for any noun of the right shape. The top FAMILIAR `-ஆய்` types were **மலாய்**,
**வாய்** ('mouth'), **கால்வாய்** ('canal'), **செவ்வாய்** ('Tuesday'). The binding rule
removes all of these, because the FST binds `+imp=∅+2sg=∅` (zero morphs) to பாய், never
`+2sg=ஆய்`, and has no verb analysis at all for பாக்கித்தான்.

Zero-morph bindings are therefore excluded explicitly: an empty suffix is a suffix of every
string and would re-admit exactly the forms this rule exists to reject.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..morph.checker import MorphChecker
from ..morph.labels import PNG_FEATURES, Slot

#: PNG labels the analyser can emit. Used to filter bindings down to person-number-gender.
PNG_LABELS = frozenset(PNG_FEATURES)

#: ThamizhiMorph's noun lexicon, used as a HOMOGRAPH VETO. A type the noun net recognises
#: is not counted as a finite verb, even when the verb net also analyses it.
#: MEASURED on Tamil Wikipedia (2026-08-08): the veto removes கால்வாய் ('canal', 1,201
#: tokens wrongly counted as `+fut+2sg=ஆய்`), ஆண்டாள் (429, a proper name), நாள் and வாய்,
#: and removes ZERO of வந்தான்/வந்தாள்/வந்தார்/வந்தாய்/வந்தன/வந்தனர்/உள்ளனர்/பெற்றார்.
#: Its coverage is partial — மானிப்பாய் (a place name) and எழுவாய் ('grammatical subject')
#: are not in the noun lexicon and survive — so it reduces the noun contamination without
#: eliminating it. The residue is exposed by the top-forms diagnostic in the report rather
#: than papered over.
DEFAULT_NOUN_NET = "build/thamizhi-nouns.bin"
from .text import canonicalize

#: Analysis sources that bind a feature to a real surface string.
SUFFIX_BOUND_SOURCES = ("fst-lexicon", "fallback-rule", "override")

VERB_POS = ("verb", "aux")

# --------------------------------------------------------------------------- slot values

# honorificity, pronoun-borne (HON-P). Ternary.
HON_FAMILIAR = "FAMILIAR"          # நீ
HON_POLITE = "POLITE"              # நீங்கள்
HON_DEFERENTIAL = "DEFERENTIAL"    # தாங்கள்

# honorificity, verb-borne (HON-V). Binary — this is the only verb-fused honorific contrast
# (நீங்கள்/தாங்கள் is pronoun-only in the indicative).
HONV_FAMILIAR = "FAMILIAR"         # -ஆய்
HONV_POLITE = "POLITE"             # -ஈர்கள் / -ஈர்

INCL, EXCL = "INCL", "EXCL"
UYARTHINAI, AHRINAI = "UYARTHINAI", "AHRINAI"
MASC, FEM, EPICENE = "MASC", "FEM", "EPICENE"


@dataclass(frozen=True)
class SuffixRule:
    """(coarse label class, surface suffix) -> slot value."""
    slot: str
    value: str
    suffix: str                  # SURFACE form, e.g. 'ான்' not 'ஆன்'
    labels: tuple[str, ...]      # PNG labels the FST is consistent about at this coarseness
    ambiguous: str = ""          # non-empty ⇒ the count is an UPPER BOUND on `value`


#: Ordered longest-suffix-first. `ன` must come last: it is the surface of `-அன` (`வந்தன`)
#: and is only reachable when no longer suffix matched. Note `-ஆன்` ends in `ன்` (ன + pulli),
#: so it can never collide with the bare-`ன` rule.
SUFFIX_RULES: tuple[SuffixRule, ...] = (
    # --- rationality: 3PL uyartiṇai vs aḵriṇai
    SuffixRule("rationality", UYARTHINAI, "ார்கள்", ("3sghe", "3ple", "3pl"),
               ambiguous="D-2: -ஆர்கள் is honorific-SINGULAR or rational-PLURAL; the "
                         "count is an UPPER bound on the plural reading"),
    SuffixRule("rationality", UYARTHINAI, "னர்", ("3ple", "3sge", "3pl", "3sghe"),
               ambiguous=""),   # -அனர்/-இனர் is unambiguously plural
    SuffixRule("rationality", AHRINAI, "ன", ("3pln",)),
    # --- gender in the 3SG verb
    SuffixRule("gender", MASC, "ான்", ("3sgm",)),
    SuffixRule("gender", FEM, "ாள்", ("3sgf",)),
    SuffixRule("gender", EPICENE, "ார்", ("3sghe", "3ple", "3sge", "3sgh"),
               ambiguous="D-2: -ஆர் is honorific-SINGULAR (epicene) or rational-PLURAL"),
    # --- honorificity, verb-borne (2nd person)
    SuffixRule("honorificity_verb", HONV_FAMILIAR, "ாய்", ("2sg", "2sgn", "2sgm", "2sgf")),
    SuffixRule("honorificity_verb", HONV_POLITE, "ீர்கள்", ("2pl", "2plh"),
               ambiguous="-ஈர்கள் is honorific-SINGULAR addressee or plain PLURAL addressee"),
    SuffixRule("honorificity_verb", HONV_POLITE, "ீர்", ("2sgh", "2pl", "2plh"),
               ambiguous="literary -ஈர்; low frequency, reported separately"),
)


# --------------------------------------------------------------------------- pronouns
#
# Every form below was confirmed to analyse against ThamizhiMorph's `pronoun.fst` on
# 2026-08-08 unless marked. Case forms are listed explicitly rather than generated, because
# a generated list would silently include non-words and inflate the count.

CLUSIVITY_FORMS: dict[str, dict[str, list[str]]] = {
    INCL: {
        "nominative": ["நாம்"],
        "oblique": ["நம்", "நம்மை", "நமக்கு", "நமது", "நம்முடைய", "நம்மால்", "நம்மில்",
                    "நம்மிடம்", "நம்முடன்", "நம்மோடு", "நம்மிடையே"],
    },
    EXCL: {
        "nominative": ["நாங்கள்"],
        "oblique": ["எங்கள்", "எங்களை", "எங்களுக்கு", "எங்களது", "எங்களுடைய", "எங்களால்",
                    "எங்களில்", "எங்களிடம்", "எங்களுடன்", "எங்களோடு", "எங்களிடையே"],
    },
}

HONORIFICITY_FORMS: dict[str, dict[str, list[str]]] = {
    HON_FAMILIAR: {
        "nominative": ["நீ"],
        "oblique": ["உன்", "உன்னை", "உனக்கு", "உனது", "உன்னுடைய", "உன்னால்", "உன்னிடம்",
                    "உன்னுடன்", "உன்னோடு", "உன்னில்"],
    },
    HON_POLITE: {
        "nominative": ["நீங்கள்"],
        "oblique": ["உங்கள்", "உங்களை", "உங்களுக்கு", "உங்களது", "உங்களுடைய", "உங்களால்",
                    "உங்களிடம்", "உங்களுடன்", "உங்களோடு", "உங்களில்"],
    },
    HON_DEFERENTIAL: {
        "nominative": ["தாங்கள்"],
        "oblique": ["தங்கள்", "தங்களை", "தங்களுக்கு", "தங்களது", "தங்களுடைய", "தங்களால்",
                    "தங்களிடம்", "தங்களுடன்", "தங்களோடு", "தங்களில்"],
    },
}

#: Cues that a தாங்கள்/தங்கள் token is 2nd-person ADDRESS rather than 3rd-person reflexive.
#: ThamizhiMorph analyses தாங்கள் ONLY as `+pron+3pl+refl` — the deferential 2nd-person
#: reading is absent from the analyser, so this heuristic, not the FST, is what brackets it.
SECOND_PERSON_CUES = ("நீங்கள்", "நீ", "உங்கள்", "உங்களை", "உங்களுக்கு", "தாங்கள்")
SECOND_PERSON_VERB_SUFFIXES = ("ீர்கள்", "ீர்", "ாய்", "ுங்கள்")


@dataclass
class FormInventory:
    """FST-derived: which surface types realise which slot value.

    Built ONCE from the union of every corpus's type inventory, then applied identically to
    every corpus. Building it per corpus would make the corpora incomparable, which is the
    one thing the cross-corpus agreement analysis must not allow.
    """
    #: slot -> value -> {form: 'bound'|'guesser'}
    verb: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    #: slot -> value -> [form] for EVERY candidate type matching the suffix, with NO FST
    #: gate at all. Backs the `surface_only` variant, which bounds how much the FST gate
    #: itself moves the rate. It over-counts (`நாள்` 'day' ends in -ஆள்), so it is a
    #: diagnostic, never a headline.
    surface: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    ambiguity: dict[str, dict[str, str]] = field(default_factory=dict)
    n_types_analysed: int = 0
    n_types_finite: int = 0
    n_types_noun_vetoed: int = 0
    noun_veto: bool = False

    def forms(self, slot: str, value: str, include_guesser: bool = False) -> list[str]:
        d = self.verb.get(slot, {}).get(value, {})
        return [f for f, src in d.items() if include_guesser or src == "bound"]

    def all_forms(self, include_guesser: bool = False) -> list[str]:
        out: set[str] = set()
        for slot, vals in self.verb.items():
            for value in vals:
                out.update(self.forms(slot, value, include_guesser))
        return sorted(out)

    def to_json(self) -> dict:
        return {
            "verb": self.verb,
            "surface": self.surface,
            "ambiguity": self.ambiguity,
            "n_types_analysed": self.n_types_analysed,
            "n_types_finite": self.n_types_finite,
            "n_types_noun_vetoed": self.n_types_noun_vetoed,
            "noun_veto": self.noun_veto,
            "pronouns": {"clusivity": CLUSIVITY_FORMS, "honorificity": HONORIFICITY_FORMS},
            "suffix_rules": [
                {"slot": r.slot, "value": r.value, "suffix": r.suffix,
                 "labels": list(r.labels), "ambiguous": r.ambiguous}
                for r in SUFFIX_RULES],
        }

    @classmethod
    def from_json(cls, d: dict) -> "FormInventory":
        return cls(verb=d["verb"], surface=d.get("surface", {}),
                   ambiguity=d.get("ambiguity", {}),
                   n_types_analysed=d.get("n_types_analysed", 0),
                   n_types_finite=d.get("n_types_finite", 0),
                   n_types_noun_vetoed=d.get("n_types_noun_vetoed", 0),
                   noun_veto=d.get("noun_veto", False))


def candidate(form: str) -> bool:
    """Cheap superset test — is this type worth sending to the FST?

    A superset by construction: every suffix in `SUFFIX_RULES` plus every listed pronoun.
    Anything this rejects can carry no slot value, so rejecting it costs nothing and it
    keeps the FST pass over a Wikipedia-sized type inventory to a few hundred thousand
    lookups instead of several million.
    """
    if form in _PRONOUN_SET:
        return True
    return form.endswith(_CANDIDATE_SUFFIXES)


_CANDIDATE_SUFFIXES = tuple(r.suffix for r in SUFFIX_RULES)
_PRONOUN_SET = frozenset(
    canonicalize(f)
    for table in (CLUSIVITY_FORMS, HONORIFICITY_FORMS)
    for value in table.values()
    for lst in value.values()
    for f in lst)


def _match_rule(form: str, bindings: "set[tuple[str, str]]") -> SuffixRule | None:
    """`bindings` is {(label, surface_suffix)} from finite verb analyses, zero morphs removed.

    A rule fires when the form ends in the rule's surface suffix AND some binding carries a
    label in the rule's class AND that binding's own surface string is a suffix of the
    rule's suffix. The last conjunct is what lets `கூறினர்` (`+3sge=அர்`, surface `ர்`)
    match the `-அனர்` rule while `பாய்` (`+2sg=∅`) matches nothing.
    """
    for r in SUFFIX_RULES:
        if not form.endswith(r.suffix):
            continue
        for label, suf in bindings:
            if label in r.labels and suf and r.suffix.endswith(suf):
                return r
    return None


def _match_suffix(form: str) -> SuffixRule | None:
    """Suffix match with NO label gate — the `surface_only` diagnostic."""
    for r in SUFFIX_RULES:
        if form.endswith(r.suffix):
            return r
    return None


#: ⚠ MUST stay small. `MorphChecker._flookup` writes an entire batch to `flookup`'s stdin
#: and only then starts reading its stdout. Both pipes are 64 KB on macOS, so a large batch
#: DEADLOCKS: flookup blocks writing analyses nobody is reading, we block writing words
#: nobody is consuming. Measured: batch=20,000 hangs indefinitely (killed after 3.5 min at
#: 0% CPU). 250 words in ≈ 5 KB and out ≈ 30 KB, comfortably inside one buffer.
FLOOKUP_BATCH = 250


def build_inventory(types: Iterable[str], checker: MorphChecker | None = None,
                    batch: int = FLOOKUP_BATCH,
                    noun_net: str | None = DEFAULT_NOUN_NET) -> FormInventory:
    """Run the FST over candidate types and assign each to a slot value."""
    import os

    checker = checker or MorphChecker()
    inv = FormInventory()
    use_nouns = bool(noun_net) and os.path.exists(noun_net)
    inv.noun_veto = use_nouns
    buf: list[str] = []

    def flush(chunk: list[str]) -> None:
        nouns = ({w for w, a in checker._flookup(chunk, noun_net).items() if a}
                 if use_nouns else set())
        for res in checker.analyse_batch(chunk):
            inv.n_types_analysed += 1
            sr = _match_suffix(res.surface)
            if sr is not None:
                inv.surface.setdefault(sr.slot, {}).setdefault(sr.value, []).append(
                    res.surface)
            bindings: set[tuple[str, str]] = set()
            finite = False
            for a in res.analyses:
                if a.pos not in VERB_POS or "fin" not in a.tags:
                    continue
                finite = True
                if a.source not in SUFFIX_BOUND_SOURCES:
                    continue
                for b in a.bindings:
                    if b.label in PNG_LABELS and not b.is_zero:
                        bindings.add((b.label, b.surface_suffix))
            if not finite:
                continue
            inv.n_types_finite += 1
            if res.surface in nouns:
                inv.n_types_noun_vetoed += 1
                continue
            rule = _match_rule(res.surface, bindings)
            if rule is None:
                continue
            inv.verb.setdefault(rule.slot, {}).setdefault(rule.value, {})[res.surface] = "bound"
            if rule.ambiguous:
                inv.ambiguity.setdefault(rule.slot, {})[rule.value] = rule.ambiguous

    for t in types:
        if not candidate(t):
            continue
        buf.append(t)
        if len(buf) >= batch:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    return inv


# --------------------------------------------------------------------------- slot specs

@dataclass(frozen=True)
class SlotSpec:
    """One prior table: a slot, its licit values, and the forms realising each."""
    slot: str
    values: tuple[str, ...]
    forms: dict[str, list[str]]
    kind: str                      # 'pronoun' | 'verb'
    variant: str                   # 'nominative' | 'all_forms' | 'suffix_bound' | …
    caveats: list[str] = field(default_factory=list)


def slot_specs(inv: FormInventory) -> list[SlotSpec]:
    """The prior tables the analysis needs, each with its variants."""
    specs: list[SlotSpec] = []

    # ---- clusivity (pronoun-only: the verb ending -ஓம் is clusivity-neutral)
    for variant, keys in (("nominative", ("nominative",)),
                          ("all_forms", ("nominative", "oblique"))):
        specs.append(SlotSpec(
            slot="clusivity", values=(INCL, EXCL), kind="pronoun", variant=variant,
            forms={v: [canonicalize(f) for k in keys for f in CLUSIVITY_FORMS[v][k]]
                   for v in (INCL, EXCL)},
            caveats=["நாம் also has a generic/impersonal 'one' use; not separated here.",
                     "The 1PL verb ending -ஓம் does not mark clusivity, so there is no "
                     "verb-borne variant of this slot."]
            + (["Oblique நம் is homographous with the possessive 'our(incl)'."]
               if variant == "all_forms" else [])))

    # ---- honorificity, pronoun-borne (ternary)
    for variant, keys in (("nominative", ("nominative",)),
                          ("all_forms", ("nominative", "oblique"))):
        specs.append(SlotSpec(
            slot="honorificity_pronoun",
            values=(HON_FAMILIAR, HON_POLITE, HON_DEFERENTIAL),
            kind="pronoun", variant=variant,
            forms={v: [canonicalize(f) for k in keys for f in HONORIFICITY_FORMS[v][k]]
                   for v in (HON_FAMILIAR, HON_POLITE, HON_DEFERENTIAL)},
            caveats=[
                "நீங்கள் is syncretic: honorific-SINGULAR or plain PLURAL addressee. The "
                "POLITE count is an UPPER bound on the honorific-singular reading "
                "(DECISIONS D-2).",
                "தாங்கள்/தங்கள் is syncretic with the 3rd-person REFLEXIVE plural ('they "
                "themselves'), and ThamizhiMorph analyses it ONLY as +pron+3pl+refl. The "
                "DEFERENTIAL count is therefore a gross upper bound before the "
                "`thangal_second_person_rate` correction is applied.",
            ] + (["Oblique உங்கள்/தங்கள் are also plain genitives, which inflates both "
                  "non-familiar values relative to the nominative variant."]
                 if variant == "all_forms" else [])))

    # ---- verb-borne slots, from the FST inventory.
    #
    # THREE variants, and the spread between them is the honest measure of how much the
    # analyser is deciding the answer:
    #   `suffix_bound`  — HEADLINE. Some analysis binds a PNG label to the matching surface
    #                     suffix. Precise; its residual risk is LEXICAL homography
    #                     (`வருவாய்` is both `வரு+fut+2sg=ஆய்` and the noun 'revenue'),
    #                     which no morphology can resolve — hence the top-forms diagnostic.
    #   `surface_only`  — no FST gate at all. Over-counts badly (`நாள்` 'day' ends in -ஆள்,
    #                     `பாக்கித்தான்` in -ஆன்); it exists to show how much work the gate
    #                     is doing, and the gap between the two variants IS that number.
    for slot, values in (("honorificity_verb", (HONV_FAMILIAR, HONV_POLITE)),
                         ("rationality", (UYARTHINAI, AHRINAI)),
                         ("gender", (MASC, FEM, EPICENE))):
        specs.append(SlotSpec(
            slot=slot, values=values, kind="verb", variant="suffix_bound",
            forms={v: inv.forms(slot, v) for v in values},
            caveats=[inv.ambiguity.get(slot, {}).get(v, "") for v in values]))
        specs.append(SlotSpec(
            slot=slot, values=values, kind="verb", variant="surface_only",
            forms={v: sorted(set(inv.surface.get(slot, {}).get(v, []))) for v in values},
            caveats=[inv.ambiguity.get(slot, {}).get(v, "") for v in values]
            + ["No FST gate: any type ending in the suffix is counted, so nouns such as "
               "நாள் ('day', -ஆள்) and மகான் (-ஆன்) are included. Diagnostic upper bound "
               "on the effect of the analyser gate; never a headline."]))

    # ---- rationality, strict: -அனர் only, which carries no honorific-singular reading
    specs.append(SlotSpec(
        slot="rationality", values=(UYARTHINAI, AHRINAI), kind="verb",
        variant="strict_anar",
        forms={UYARTHINAI: [f for f in inv.forms("rationality", UYARTHINAI)
                            if f.endswith("னர்")],
               AHRINAI: inv.forms("rationality", AHRINAI)},
        caveats=["UYARTHINAI restricted to -அனர்/-இனர், which is unambiguously plural. "
                 "This is the lower bound on the uyartiṇai rate; the `suffix_bound` "
                 "variant, which also counts -ஆர்கள், is the upper bound."]))

    for s in specs:
        s.caveats[:] = [c for c in s.caveats if c]
    return specs
