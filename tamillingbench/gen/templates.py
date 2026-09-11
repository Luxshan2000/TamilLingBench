# -*- coding: utf-8 -*-
"""Template loading and validation.

The rule this module exists to enforce, and the highest-leverage design decision:

    **A template file may not contain a Tamil surface string outside `provenance`.**

Templates carry *feature bundles*; `fstgen` turns bundles into characters. If a template were
allowed to name a Tamil form, the benchmark's morphological correctness would rest on whoever
typed it, and the whole point of running ThamizhiMorph backwards would be lost. `load_all`
raises `TamilInTemplate` on any violation, and `tests/test_templates.py` asserts it fires on a
constructed negative rather than trusting that it would.

The loader also enforces the per-slot obligations:

* gender templates pin `holds_constant.hon: minus` — `+HON` neutralises gender (§2.1), so a
  gender item whose frame licenses `-ஆர்` has no gold;
* rationality templates pin `number: PL` — the singular contrast is not a minimal pair (§1.3);
* number templates pin `tinai: uyar`, `gender: masc` and exclude the future — the அஃறிணை
  future collapses sg and pl into `வரும்` (§1.4);
* every honorificity template declares `confound_checks.number` with
  `addressee_cardinality: sg`, because நீங்கள் is honorific-sg / plain-pl syncretic and
  without this we would be scoring a *number* distinction as an honorific one;
* every clusivity template declares an `overt_environment`, because Tamil is pro-drop and the
  verb carries no clusivity exponent, so a dropped pronoun leaves the slot unrealised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "templates"

#: Tamil block U+0B80–U+0BFF plus the Tamil supplement.
TAMIL_RE = re.compile(r"[஀-௿]")

#: Keys whose subtree is exempt: `provenance` records a corpus sentence id and may carry a
#: transliterated note; `_gloss` fields are documentation for the human reader.
EXEMPT_KEYS = {"provenance", "_gloss", "_note"}

VALID_SLOTS = {"honorificity", "clusivity", "rationality", "gender", "number"}
OVERT_ENVIRONMENTS = {"nom", "acc", "dat", "gen", "nom_focus"}


class TamilInTemplate(ValueError):
    """A template file contains a Tamil surface string outside `provenance`."""


class TemplateError(ValueError):
    """A template violates the feasible-cell table or a per-slot obligation."""


@dataclass(frozen=True)
class Template:
    id: str
    version: str
    slot: str
    slot_family: str | None
    k: int
    values: tuple[str, ...]
    register: str
    stratum: str
    holds_constant: dict[str, Any]
    fillers: dict[str, Any]
    source: dict[str, str]
    context_slot: str
    target: dict[str, dict[str, Any]]
    target_frame: dict[str, str]
    decides_on: str
    cue_source: str
    #: which key of `cues.yaml` the cue table lives under. Defaults to `slot`;
    #: a dataset version that re-authors its cue inventory names a new key
    #: rather than editing the old one, so earlier versions stay rebuildable.
    cue_slot: str | None
    referent_number: str | None
    confound_checks: dict[str, Any]
    avoidance: dict[str, Any]
    interp_eligible: bool
    provenance: dict[str, Any]
    flags: tuple[str, ...] = ()
    third_degree: dict[str, Any] | None = None
    variety_variants: dict[str, list[dict]] | None = None
    c0_probe: bool = False
    c3_only: bool = False
    path: str = ""

    @property
    def chance_rate(self) -> float:
        return 1.0 / self.k


def _scan_tamil(node: Any, path: str) -> None:
    if isinstance(node, dict):
        for key, val in node.items():
            if key in EXEMPT_KEYS:
                continue
            _scan_tamil(val, f"{path}.{key}")
    elif isinstance(node, list):
        for i, val in enumerate(node):
            _scan_tamil(val, f"{path}[{i}]")
    elif isinstance(node, str) and TAMIL_RE.search(node):
        raise TamilInTemplate(
            f"{path} = {node!r} contains Tamil. Templates carry feature bundles only; "
            f"surface forms come from ThamizhiMorph generation.")


def load_template(path: Path) -> Template:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _scan_tamil(raw, path.name)

    slot = raw["slot"]
    if slot not in VALID_SLOTS:
        raise TemplateError(f"{path.name}: unknown slot {slot!r}")
    hc = raw.get("holds_constant") or {}
    cc = raw.get("confound_checks") or {}
    av = raw.get("avoidance") or {}

    # ---- feasible-cell table ---------------------------------------------------------
    if slot == "gender":
        if hc.get("hon") != "minus":
            raise TemplateError(f"{path.name}: gender templates must pin holds_constant.hon "
                                f"= 'minus'; got {hc.get('hon')!r}")
        if av.get("neutral_form_possible") is not True:
            raise TemplateError(f"{path.name}: gender templates must declare "
                                f"avoidance.neutral_form_possible: true (the -ஆர் escape)")
        if not cc.get("stereotype"):
            raise TemplateError(f"{path.name}: gender templates require "
                                f"confound_checks.stereotype")
    elif av.get("neutral_form_possible"):
        raise TemplateError(f"{path.name}: neutral_form_possible is a gender-only property")

    if slot == "rationality" and hc.get("number") != "PL":
        raise TemplateError(f"{path.name}: rationality templates must pin number: PL"
                            f"; got {hc.get('number')!r}")
    if slot == "number":
        # Two licensed referent types, and the choice determines every other pin.
        #   uyar — 3sg −HON is -ஆன்/-ஆள் with no gender-free member, so gender must be
        #     pinned MASC and supplied by the input (the released number items use this);
        #   ahri — -அது / -அன, which carry neither gender nor honorificity, so both are
        #     pinned null. See `_assert_number_not_syncretic` for why this type exists.
        tinai = hc.get("tinai")
        if tinai == "uyar":
            if hc.get("gender") != "masc":
                raise TemplateError(f"{path.name}: an உயர்திணை number template pins "
                                    f"gender: masc")
        elif tinai == "ahri":
            if hc.get("gender") is not None or hc.get("hon") is not None:
                raise TemplateError(
                    f"{path.name}: an அஃறிணை number template pins gender: null and "
                    f"hon: null. அஃறிணை agreement expresses neither, and declaring either "
                    f"would let the reference over-commit to a feature the source never "
                    f"supplies (NUMBER-C1-GENDER-OVERCOMMIT).")
        else:
            raise TemplateError(f"{path.name}: number templates pin tinai ∈ "
                                f"{{'uyar', 'ahri'}}; got {tinai!r}")
        if any(b.startswith("fut") for v in raw["target"].values()
               for b in _bundles_of(v)):
            raise TemplateError(f"{path.name}: number templates must exclude the future — "
                                f"the அஃறிணை future collapses sg and pl")
        _assert_number_not_syncretic(path, raw)
    if slot == "honorificity":
        n = cc.get("number")
        if not n or n.get("addressee_cardinality") != "sg":
            raise TemplateError(
                f"{path.name}: every honorificity template must declare "
                f"confound_checks.number.addressee_cardinality = 'sg'. நீங்கள் is "
                f"honorific-sg / plain-pl syncretic, so without a guaranteed singular "
                f"addressee we would be scoring number as honorificity.")
        if n.get("device") not in ("singular_vocative", "named_individual", "sg_definite_np"):
            raise TemplateError(f"{path.name}: confound_checks.number.device invalid")
    if slot == "clusivity":
        env = av.get("overt_environment")
        if env not in OVERT_ENVIRONMENTS:
            raise TemplateError(
                f"{path.name}: clusivity templates must declare "
                f"avoidance.overt_environment ∈ {sorted(OVERT_ENVIRONMENTS)}; got {env!r}. "
                f"Tamil is pro-drop and the verb is -ஓம் either way, so a nominative "
                f"clusivity item carries no exponent of the slot anywhere.")
        if not cc.get("addressee_membership"):
            raise TemplateError(f"{path.name}: clusivity templates require "
                                f"confound_checks.addressee_membership")

    # ---- polarity pin (added 2026-08-08; a hole in the interaction lattice) -----------
    # `வரவில்லை` carries NO person, number, gender, honorificity, rationality or tense
    # (Lehmann §1.47, confirmed by execution) — negation with -வில்லை deletes every slot we
    # score, so a system can escape ANY slot by negating. `மாட்ட-` preserves them. Every
    # scored item therefore pins positive polarity, and `08` gets a distinct `AVOIDANT_NEG`
    # outcome so the escape is measured rather than counted as a wrong answer.
    if hc.get("polarity") != "positive":
        raise TemplateError(
            f"{path.name}: every scored template must pin holds_constant.polarity = "
            f"'positive'. Tamil -வில்லை negation deletes person, number, gender, "
            f"honorificity, rationality AND tense, so an unpinned item lets a system escape "
            f"the slot entirely.")

    # ---- structural -------------------------------------------------------------------
    values = tuple(raw["values"])
    if len(values) != raw["k"]:
        raise TemplateError(f"{path.name}: k={raw['k']} but {len(values)} values")
    if set(raw["target"]) != set(values):
        raise TemplateError(f"{path.name}: target keys {sorted(raw['target'])} != values")
    if raw["source"]["C2"] != raw["source"]["C3"]:
        raise TemplateError(f"{path.name}: source.C2 and source.C3 must be byte-identical"
                            f" — that identity is what makes the C2 chance "
                            f"rate exactly 1/k")
    # The C2/C3 source is shared by all k members of a set, so it may not contain a
    # value-dependent placeholder. `{CUE}` is the only one the builder substitutes per
    # value, and it is silently erased there — which would make the identity hold by
    # accident rather than by construction. Fail at load time instead.
    for cond in ("C2", "C3"):
        if "{CUE}" in raw["source"][cond]:
            raise TemplateError(
                f"{path.name}: source.{cond} contains {{CUE}}. The C2/C3 sentence is shared "
                f"byte-for-byte by all {raw['k']} members of the set; a "
                f"per-value placeholder there would break the 1/k chance rate.")
    return Template(
        id=raw["id"], version=raw["version"], slot=slot,
        slot_family=raw.get("slot_family"), k=raw["k"], values=values,
        register=raw.get("register", "literary"), stratum=raw["stratum"],
        holds_constant=hc, fillers=raw["fillers"], source=raw["source"],
        context_slot=raw["context_slot"], target=raw["target"],
        target_frame=raw["target_frame"], decides_on=raw["decides_on"],
        cue_source=raw.get("cue_source", "filler"),
        cue_slot=raw.get("cue_slot"),
        referent_number=raw.get("referent_number"), confound_checks=cc, avoidance=av,
        interp_eligible=bool(raw.get("interp", {}).get("eligible", False)),
        provenance=raw["provenance"], flags=tuple(raw.get("flags", ())),
        third_degree=raw.get("third_degree"),
        variety_variants=raw.get("variety_variants"),
        c0_probe=bool(raw.get("c0_probe", False)),
        c3_only=bool(raw.get("c3_only", False)), path=str(path))


def _assert_number_not_syncretic(path: Path, raw: dict) -> None:
    """The number control's own forms may not be ambiguous *about number*.

    ⛔ The released number items are built on the உயர்திணை pair `-ஆன்` / `-ஆர்கள்` (see the
    known issues in `data/benchmark/manifest.json`). `-ஆர்` and `-ஆர்கள்` are honorific-singular
    OR rational-plural and *nothing in the word form decides* (`labels.SUFFIX_SYNCRETISM`), so
    a form the model actually produced expanded to two readings that disagree about NUMBER —
    the one feature under test. DECISIONS.md D-2 resolves such ambiguity by filtering on the
    item's declared referent number, which for THIS slot would hand the model the answer, so
    the filter is unavailable by construction. Measured on real output: `-ஆர்` scored WRONG on
    397 singular items and CORRECT on 112 plural ones — the same string, opposite verdicts.

    A template opts into the rule by declaring `confound_checks.number_syncretism.required:
    true`. It is opt-in so that the released number templates, which predate the rule, still
    load and the documented defect stays inspectable.
    """
    from ..morph.labels import SUFFIX_SYNCRETISM
    from .features import VERB_BUNDLES

    spec = (raw.get("confound_checks") or {}).get("number_syncretism") or {}
    if not spec.get("required"):
        return
    for value, tv in raw["target"].items():
        for bundle in _bundles_of(tv):
            if bundle not in VERB_BUNDLES:
                continue
            suffix = VERB_BUNDLES[bundle][1]
            if suffix in SUFFIX_SYNCRETISM:
                raise TemplateError(
                    f"{path.name}: target {value} uses bundle {bundle!r}, whose exponent is "
                    f"in SUFFIX_SYNCRETISM ({SUFFIX_SYNCRETISM[suffix]}). Those readings "
                    f"disagree about NUMBER, and the number slot cannot context-filter on "
                    f"number without begging the question it is asking.")


def _bundles_of(target_value: dict[str, Any]) -> list[str]:
    out = []
    for part in target_value.values():
        if isinstance(part, dict) and "bundle" in part:
            out.append(part["bundle"])
    return out


def load_all(directory: Path = TEMPLATE_DIR, overlay: Path | None = None) -> list[Template]:
    """Load every template in `directory`.

    `overlay`, if given, is a second directory whose files replace same-id families; it is a
    convenience for experimenting with variants without editing the originals.
    """
    templates = [load_template(p) for p in sorted(directory.rglob("*.yaml"))]
    if overlay is not None:
        over = {t.id: t for t in (load_template(p)
                                  for p in sorted(Path(overlay).rglob("*.yaml")))}
        unknown = set(over) - {t.id for t in templates}
        if unknown:
            raise TemplateError(f"overlay introduces unknown template_id: {sorted(unknown)}. "
                                f"An overlay may only REPLACE a base family — a new family "
                                f"would change the allocation without saying so.")
        templates = [over.get(t.id, t) for t in templates]
    ids = [t.id for t in templates]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise TemplateError(f"duplicate template_id: {sorted(dupes)}")
    return templates
