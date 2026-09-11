# -*- coding: utf-8 -*-
"""FST *generation* with a round-trip assertion and three further screens.

The load-bearing design decision: **no Tamil verb, pronoun or noun form is ever typed
into a template file.** Templates carry feature bundles; this module turns a bundle into a
surface string by running ThamizhiMorph backwards, runs it forwards again and asserts the
analysis comes back, then applies the screens in `screens.py` that the round trip cannot
provide.

Four mechanical facts, each established by execution on 2026-08-08, each of which the naive
implementation gets wrong:

1. **`build/thamizhi-union.bin` is NOT invertible.** It holds *two* nets (merged lexicon +
   `verb-guess`). `flookup` without `-a` runs nets in *composition*, so `flookup -i` on the
   union pushes the analysis through the lexicon and then through the guesser and returns
   `+?` for everything. Nor can `build/thamizhi-lexicon.bin` be used: it merges `pronoun.fst`
   in, whose sigma carries `+1pl` as a MULTICHARACTER symbol, so `...+1pl=ஓம்` tokenizes to
   that symbol, the verb path rejects it, and `வந்தோம்` becomes ungenerable while `வந்தான்`
   generates fine — a symbol clash that looks exactly like a lexicon gap. Generation runs
   against the three dedicated nets from `scripts/build_gen_nets.sh`. Invertibility was
   flagged as a risk up front; it is real, and this is the fix.

2. **Analysis strings cannot be composed from a template.** The past allomorph
   (`=த்`/`=ட்`/`=ன்ற்`/`=இன்`), the strong/weak marker, and *which PNG slots exist at all*
   vary per lemma: `வா` has `+3sghe=ஆர்கள்` but **no** `+3sghe=ஆர்`, which is why `வந்தார்`
   returns `+?` while `செய்தார்` analyses. Paradigms are therefore *harvested* per lemma by
   enumerating the transducer, never built by string formatting.

3. **`+verb`, `+pron`, `+noun` and `=த்` are multicharacter symbols in the net's sigma.**
   A foma regex must quote them (`"+verb"`) or the filter matches nothing and the harvest
   silently returns an empty paradigm — a failure that looks exactly like "this lemma is not
   in the lexicon".

4. **The round trip does not imply correctness.** See `screens.py`. It is run against the
   same transducer that generated the form, so over-generation round-trips perfectly.

5. **Nor does surviving every screen.** A cell can have exactly one generated form, which
   round-trips and passes B/C/D, and still not be the Tamil word: the `-இன்` past class's
   அஃறிணை cells (`ஓடு` -> `ஓடினன` / `ஓடினது`, corpus counts 0 / 1 against `ஓடின` 311 and
   `ஓடியது` 740). Nothing internal to the transducer can see that. **Gate G** applies the
   corpus-measured repairs in `repairs.py`; `scripts/gold_attestation_gate.py` is the hard
   floor that stops the next one of these from shipping.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

from ..morph.labels import label_suffix_to_surface
from ..morph.normalize import normalize
from .features import IRREGULAR_EXCLUDE, PRON_ENTRIES, VERB_BUNDLES
from .repairs import repair_for
from .screens import VOWEL_SIGNS, orthographically_wellformed, stem_of

ROOT = Path(__file__).resolve().parents[2]

#: Generation nets. Built by `scripts/build_gen_nets.sh`; see that script for why the
#: checker's `thamizhi-union.bin` and `thamizhi-lexicon.bin` cannot be used here.
VERB_BIN = str(ROOT / "build" / "thamizhi-verbs.bin")
PRON_BIN = str(ROOT / "build" / "thamizhi-pron.bin")
NOUN_BIN = str(ROOT / "build" / "thamizhi-nouns.bin")

NO_ANALYSIS = "+?"
ZERO = "∅"

#: Never usable as gold: `caus` changes the lemma's meaning, `euph=அன்` is archaic literary
#: agreement (வந்தனன்), `sandhi*` are junction variants that only surface in composition,
#: and the noun net's `psp_*` / `vpart_*` are postposition compounds, not case forms.
_BLOCKLIST = ("+caus=", "+euph=", "+sandhi", "+cmpr", "+psp_", "+vpart_", "+aux=",
              "+neg", "+moodpart", "+negpart")

_PNG_RE = re.compile(r"\+(?P<png>[123](?:sg|pl)[a-z]*)(?:=(?P<suf>[^+]*))?(?=$|\+)")


class GenerationError(RuntimeError):
    """No round-trip-verified, screened surface exists for this (lemma, bundle)."""


# --------------------------------------------------------------------------- foma driver

_FOMA_NOISE = re.compile(r"^(Foma|Copyright|This is|There is|Type|defined |\?|\d)|"
                         r"kB\.|bytes\.|states,")


def _foma_upper_words(net_bin: str, prefix_regex: str, limit: int = 6000) -> list[str]:
    """Enumerate the analysis strings the net licenses under a filter.

    `Filter .o. Net` composes the filter's LOWER side against the net's UPPER side, and the
    net's upper side is the analysis (its transitions read `<+verb:0>`), so this restricts
    the enumeration to one lemma. `upper-words N` then prints them.
    """
    script = (f"load stack {net_bin}\ndefine Net;\n"
              f"regex [{prefix_regex}] .o. Net;\nupper-words {limit}\n")
    p = subprocess.run(["foma"], input=script, capture_output=True, text=True,
                       encoding="utf-8")
    return [ln.strip() for ln in p.stdout.splitlines()
            if ln.strip() and "+" in ln and not _FOMA_NOISE.search(ln.strip())]


def _flookup(net_bin: str, words: Sequence[str], inverse: bool) -> dict[str, list[str]]:
    if not words:
        return {}
    args = ["flookup", "-a", "-b"] + (["-i"] if inverse else []) + [net_bin]
    p = subprocess.run(args, input="\n".join(words) + "\n", capture_output=True, text=True,
                       encoding="utf-8")
    out: dict[str, list[str]] = {w: [] for w in words}
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        left, _, right = line.partition("\t")
        if right and right != NO_ANALYSIS:
            out.setdefault(left, []).append(right)
    return out


# --------------------------------------------------------------------------- forms

@dataclass(frozen=True)
class GenForm:
    """One FST-generated, round-trip-verified, screened surface form."""
    surface: str
    analysis: str                       # the exact analysis string that generated it
    bundle: str                         # our Tamil-free bundle name, e.g. 'past.3sg.masc'
    lemma: str
    pos: str                            # 'verb' | 'pron' | 'noun'
    #: character offsets of the DECIDING morpheme inside `surface`. Always non-None so
    #: the metric study's `hypothesis[span] == morpheme` assertion is always runnable; for a zero
    #: exponent (imperative 2sg) it is the empty span at the end of the word.
    morpheme_span: tuple[int, int] = (0, 0)
    morpheme: str = ""
    zero_exponent: bool = False
    verified_by: str = "fst-generate"
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        s, e = self.morpheme_span
        assert self.surface[s:e] == self.morpheme, (
            f"morpheme_span is wrong for {self.surface!r}: "
            f"[{s}:{e}] = {self.surface[s:e]!r} != {self.morpheme!r}")


def _png(analysis: str) -> tuple[str | None, str | None]:
    m = _PNG_RE.search(analysis)
    if not m:
        return None, None
    suf = m.group("suf")
    return m.group("png"), (None if suf in (None, "", ZERO) else suf)


def _morpheme_span(surface: str, suffix_label: str | None) -> tuple[tuple[int, int], str, bool]:
    """Locate the PNG exponent in the surface string.

    The FST writes bound suffixes with INDEPENDENT vowel letters (`=ஆன்`, U+0B86) while the
    surface uses COMBINING vowel signs (`ான்`, U+0BBE) — Trap B. Reuse the
    checker's own conversion table rather than re-deriving it here, or the two sides of the
    project will disagree about what the morpheme is.
    """
    if suffix_label is None:
        return (len(surface), len(surface)), "", True
    surf = label_suffix_to_surface(suffix_label)
    idx = surface.rfind(surf)
    if idx < 0 or not surf:
        return (len(surface), len(surface)), "", True
    return (idx, idx + len(surf)), surf, False


# --------------------------------------------------------------------------- harvester

@dataclass
class Paradigm:
    lemma: str
    pos: str
    forms: dict[str, list[GenForm]] = field(default_factory=dict)
    #: why this lemma or bundle was rejected — reported, never silently dropped
    rejected: dict[str, str] = field(default_factory=dict)
    admitted: bool = True

    def get(self, bundle: str) -> GenForm | None:
        fs = self.forms.get(bundle)
        return fs[0] if fs else None


class FstGenerator:
    """Harvest paradigms, generate from Tamil-free bundle names, screen, cache."""

    def __init__(self, cache_path: str | Path | None = None, *,
                 repairs: bool = True) -> None:
        self.cache_path = Path(cache_path or ROOT / "build" / "paradigms.json")
        #: Apply `repairs.REPAIRS` — the cells where the transducer generates a form that
        #: round-trips and is still not the Tamil word (`repairs.py` for the measurement).
        #: `repairs=False` reproduces the transducer's raw paradigms, which is what the
        #: released items were built from (see the known issues in the benchmark manifest).
        self.repairs = repairs
        self._par: dict[tuple[str, str], Paradigm] = {}
        for b in (VERB_BIN, PRON_BIN, NOUN_BIN):
            if not Path(b).exists():
                raise FileNotFoundError(f"{b} missing — run scripts/build_gen_nets.sh")

    # ------------------------------------------------------------------ verbs
    def harvest_verb(self, lemma: str) -> Paradigm:
        key = (lemma, "verb")
        if key in self._par:
            return self._par[key]
        par = Paradigm(lemma=lemma, pos="verb")
        if lemma in IRREGULAR_EXCLUDE:
            par.admitted = False
            par.rejected["*"] = f"irregular-exclude: {IRREGULAR_EXCLUDE[lemma]}"
            self._par[key] = par
            return par
        net = VERB_BIN
        analyses = [a for a in _foma_upper_words(net, f'{{{lemma}}} "+verb" ?*')
                    if a.startswith(lemma + "+") and "+fin" in a
                    and not any(b in a for b in _BLOCKLIST)]

        # map every analysis onto the bundle it satisfies (labels + required suffix)
        want: dict[str, list[str]] = {}
        for a in analyses:
            png, suf = _png(a)
            for bname, (labels, req_suf, _g) in VERB_BUNDLES.items():
                if bname == "opt.def":
                    if "+opt=" in a:
                        want.setdefault(bname, []).append(a)
                    continue
                if "+opt=" in a:
                    continue
                is_imp = "+imp=" in a
                if is_imp != bname.startswith("imp."):
                    continue
                if not is_imp:
                    tense = bname.split(".")[0]
                    if f"+{tense}=" not in a:
                        continue
                if png not in labels:
                    continue
                if req_suf is None:
                    if suf is not None:
                        continue
                elif suf != req_suf:
                    continue
                want.setdefault(bname, []).append(a)

        gen = _flookup(net, [a for v in want.values() for a in v], inverse=True)
        surfaces = sorted({s for v in gen.values() for s in v})
        back = _flookup(net, surfaces, inverse=False)

        survivors: dict[str, list[GenForm]] = {}
        for bname, alist in want.items():
            _labels, req_suf, _g = VERB_BUNDLES[bname]
            for a in alist:
                for s in gen.get(a, []):
                    ns = normalize(s, reject_latin=False)
                    if a not in back.get(s, []):                       # Gate A: round trip
                        par.rejected.setdefault(bname, "gate-A round trip")
                        continue
                    if not orthographically_wellformed(ns):            # Gate B: orthography
                        par.rejected.setdefault(bname, "gate-B orthography")
                        continue
                    span, morph, zero = _morpheme_span(ns, req_suf)
                    survivors.setdefault(bname, []).append(
                        GenForm(surface=ns, analysis=a, bundle=bname, lemma=lemma,
                                pos="verb", morpheme_span=span, morpheme=morph,
                                zero_exponent=zero))

        # Gate D FIRST, over every A/B survivor — not over what Gate C leaves behind.
        # Order matters: `விழு` licenses both விழுந்த்- and விழுத்த்-, but the ambiguity is
        # only visible in the bundles that have both. If Gate C runs first it deletes those
        # bundles and leaves `விழுத்தார்` looking unanimous, which is how a causative stem
        # ends up shipping as an intransitive past.
        past_stems = {stem_of(f.surface, f.morpheme_span)
                      for b, fs in survivors.items() if b.startswith("past.") for f in fs}
        if len(past_stems) > 1:
            par.admitted = False
            par.rejected["*"] = f"gate-D past stem not uniform: {sorted(past_stems)}"
            self._par[key] = par
            return par

        for bname, cands in survivors.items():
            stems = {stem_of(c.surface, c.morpheme_span) for c in cands}
            if len(stems) > 1:                                          # Gate C
                par.rejected[bname] = f"gate-C stem ambiguity {sorted(stems)}"
                continue
            uniq = {c.surface: c for c in cands}
            par.forms[bname] = sorted(uniq.values(), key=lambda f: f.surface)

        # Gate E — the optative is the bare root plus -க (`செய்` -> `செய்க`). ThamizhiMorph
        # tags the INFINITIVE as `+opt=க` for many lemmas (`நட` -> `நடக்க`, `பார்` ->
        # `பார்க்க`) and mis-stems it for others (`வா` -> `வாருக`, not `வருக`). Both
        # round-trip. Requiring `optative == imperative-2sg + க` removes the whole class,
        # and is the operational form of NATIVE-CHECK-HON-4's worry that the optative is
        # only productive over a closed list.
        root = par.forms.get("imp.2sg.fam")

        # Gate F — the imperative plural is root + -உங்கள், and ThamizhiMorph gets the
        # junction wrong for exactly two root shapes: pulli-final roots need gemination
        # (`கொள்` -> கொள்ளுங்கள், FST gives கொளுங்கள்) and short-vowel-final roots need a
        # glide (`நட` -> நடவுங்கள், FST gives நடுங்கள்). Both round-trip. Admitting only
        # roots whose last character is a VOWEL SIGN, with an affix no longer than -யுங்கள்,
        # keeps the shapes the FST is uniformly right about and drops `எடுக்குங்கள்` /
        # `மறுக்குங்கள்`, where it interpolates a spurious -க்கு-.
        i2 = par.forms.get("imp.2pl.pol")
        if i2 is not None:
            r = root[0].surface if root else ""
            keep = [f for f in i2 if r and r[-1] in VOWEL_SIGNS
                    and f.surface.startswith(r) and len(f.surface) - len(r) <= 6]
            if keep:
                par.forms["imp.2pl.pol"] = keep
            else:
                par.rejected["imp.2pl.pol"] = (
                    f"gate-F imperative-plural junction: root {r!r} -> "
                    f"{[f.surface for f in i2]}")
                del par.forms["imp.2pl.pol"]

        opt = par.forms.get("opt.def")
        if opt is not None:
            expect = (root[0].surface + "க") if root else None
            keep = [f for f in opt if expect and f.surface == expect]
            if keep:
                par.forms["opt.def"] = keep
            else:
                par.rejected["opt.def"] = (
                    f"gate-E optative is not root+க: got {[f.surface for f in opt]}, "
                    f"expected {expect!r}")
                del par.forms["opt.def"]

        # Gate G — PARADIGM REPAIR. Runs LAST, over what gates A-F leave behind, because it
        # rewrites the stem shape (`ஓடின`- -> `ஓடிய`-) and gate D asserts that every past
        # form of a lemma shares one stem. Running it earlier would make the lemma fail its
        # own uniformity check on a difference the repair deliberately introduces.
        if self.repairs:
            self._apply_repairs(par)

        self._par[key] = par
        return par

    def _apply_repairs(self, par: Paradigm) -> None:
        """Apply `repairs.REPAIRS` to a harvested verb paradigm — see `repairs.py`.

        The round trip cannot screen these cells: it runs against the transducer that
        generated the form. A repaired surface is therefore re-verified downstream by
        `scripts/build_lexicon.py` against (a) the corpus and (b) the *checker*, which is a
        strictly stronger test because the checker is the configuration that scores models.
        """
        for bundle in sorted(par.forms):
            forms = par.forms[bundle]
            rep = repair_for(forms[0].analysis, bundle)
            if rep is None:
                continue
            if rep.rewrite is None:                                   # withdraw
                par.rejected[bundle] = (
                    f"gate-G withdrawn ({rep.native_check}): {rep.why} "
                    f"[FST gave {forms[0].surface}; {rep.evidence}]")
                del par.forms[bundle]
                continue
            out = []
            for f in forms:
                stem = f.surface[:f.morpheme_span[0]]
                try:
                    surface = rep.rewrite(stem)
                except ValueError as e:
                    par.rejected[bundle] = f"gate-G repair inapplicable: {e}"
                    break
                if not orthographically_wellformed(surface):
                    par.rejected[bundle] = (
                        f"gate-G repair produced a malformed string: {surface}")
                    break
                _labels, req_suf, _g = VERB_BUNDLES[bundle]
                # `zero_exponent` repairs have no separable PNG string: the marker is fused
                # into the tense morph (`ஓடின`). Locating `req_suf` in them would point the
                # span at a character belonging to another morpheme — the final `ன` of
                # `-இன்` is not the `-அன` suffix — so the span is recorded as zero-width.
                span, morph, zero = _morpheme_span(
                    surface, None if rep.zero_exponent else req_suf)
                out.append(replace(f, surface=surface, morpheme_span=span, morpheme=morph,
                                   zero_exponent=zero, verified_by="fst-generate+repair",
                                   flags=f.flags + (rep.native_check,)))
            if out:
                par.forms[bundle] = out
            else:
                par.forms.pop(bundle, None)

    # ------------------------------------------------------------------ pronouns
    def harvest_pron(self, entry_name: str) -> Paradigm:
        e = PRON_ENTRIES[entry_name]
        lemma = e["lemma"]
        key = (entry_name, "pron")
        if key in self._par:
            return self._par[key]
        par = Paradigm(lemma=lemma, pos="pron")
        net = PRON_BIN
        analyses = [a for a in _foma_upper_words(net, f'{{{lemma}}} "+pron" ?*')
                    if a.startswith(lemma + "+") and not any(b in a for b in _BLOCKLIST)]
        want: dict[str, list[str]] = {}
        for a in analyses:
            tags = a.split("+")[1:]
            if not all(r in tags for r in e["require"]):
                continue
            cases = [t for t in tags if t in ("nom", "acc", "dat", "gen")]
            if len(cases) != 1:
                continue
            bundle = cases[0] + (".foc" if "foc" in tags else "")
            want.setdefault(bundle, []).append(a)

        gen = _flookup(net, [a for v in want.values() for a in v], inverse=True)
        surfaces = sorted({s for v in gen.values() for s in v})
        back = _flookup(net, surfaces, inverse=False)
        for bundle, alist in want.items():
            cands = []
            for a in alist:
                for s in gen.get(a, []):
                    ns = normalize(s, reject_latin=False)
                    if a not in back.get(s, []) or not orthographically_wellformed(ns):
                        continue
                    cands.append(GenForm(surface=ns, analysis=a, bundle=bundle, lemma=lemma,
                                         pos="pron", morpheme_span=(0, len(ns)),
                                         morpheme=ns, flags=tuple(e.get("flags", ()))))
            if cands:
                # deterministic: shortest form first, then lexicographic. நமது before நம்மது.
                par.forms[bundle] = sorted(cands, key=lambda f: (len(f.surface), f.surface))
        self._par[key] = par
        return par

    # ------------------------------------------------------------------ nouns
    def harvest_noun(self, lemma: str) -> Paradigm:
        key = (lemma, "noun")
        if key in self._par:
            return self._par[key]
        par = Paradigm(lemma=lemma, pos="noun")
        net = NOUN_BIN
        analyses = [a for a in _foma_upper_words(net, f'{{{lemma}}} "+noun" ?*')
                    if a.startswith(lemma + "+") and not any(b in a for b in _BLOCKLIST)
                    and "+foc" not in a]
        want: dict[str, list[str]] = {}
        for a in analyses:
            tags = a.split("+")[1:]
            cases = [t for t in tags if t in ("nom", "acc", "dat", "gen")]
            if len(cases) != 1:
                continue
            want.setdefault(("pl." if "pl" in tags else "sg.") + cases[0], []).append(a)
        gen = _flookup(net, [a for v in want.values() for a in v], inverse=True)
        surfaces = sorted({s for v in gen.values() for s in v})
        back = _flookup(net, surfaces, inverse=False)
        for bundle, alist in want.items():
            cands = []
            for a in alist:
                for s in gen.get(a, []):
                    ns = normalize(s, reject_latin=False)
                    if a not in back.get(s, []) or not orthographically_wellformed(ns):
                        continue
                    cands.append(GenForm(surface=ns, analysis=a, bundle=bundle, lemma=lemma,
                                         pos="noun", morpheme_span=(0, len(ns)), morpheme=ns))
            if cands:
                uniq = {c.surface: c for c in cands}
                if len(uniq) > 1:
                    par.rejected[bundle] = f"screen-C surface ambiguity {sorted(uniq)}"
                    continue
                par.forms[bundle] = list(uniq.values())
        self._par[key] = par
        return par

    # ------------------------------------------------------------------ API
    def generate(self, lemma: str, bundle: str, pos: str = "verb") -> GenForm:
        par = {"verb": self.harvest_verb, "pron": self.harvest_pron,
               "noun": self.harvest_noun}[pos](lemma)
        if not par.admitted:
            raise GenerationError(f"{pos} {lemma!r} rejected: {par.rejected.get('*')}")
        f = par.get(bundle)
        if f is None:
            raise GenerationError(
                f"{pos} {lemma!r} has no screened form for {bundle!r}; "
                f"have {sorted(par.forms)}; rejected {par.rejected}")
        return f

    def can(self, lemma: str, bundle: str, pos: str = "verb") -> bool:
        try:
            self.generate(lemma, bundle, pos)
            return True
        except GenerationError:
            return False

    # ------------------------------------------------------------------ cache
    def save(self) -> None:
        blob = {
            f"{pos}\t{lemma}": dict(
                admitted=par.admitted, rejected=par.rejected,
                forms={b: [dict(surface=f.surface, analysis=f.analysis, bundle=f.bundle,
                                lemma=f.lemma, pos=f.pos,
                                morpheme_span=list(f.morpheme_span), morpheme=f.morpheme,
                                zero_exponent=f.zero_exponent,
                                verified_by=f.verified_by, flags=list(f.flags))
                           for f in fs] for b, fs in par.forms.items()})
            for (lemma, pos), par in self._par.items()}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(blob, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    def load(self) -> bool:
        if not self.cache_path.exists():
            return False
        for k, v in json.loads(self.cache_path.read_text(encoding="utf-8")).items():
            pos, lemma = k.split("\t")
            par = Paradigm(lemma=lemma, pos=pos, admitted=v["admitted"],
                           rejected=v["rejected"])
            for b, fs in v["forms"].items():
                par.forms[b] = [GenForm(surface=f["surface"], analysis=f["analysis"],
                                        bundle=f["bundle"], lemma=f["lemma"], pos=f["pos"],
                                        morpheme_span=tuple(f["morpheme_span"]),
                                        morpheme=f["morpheme"],
                                        zero_exponent=f["zero_exponent"],
                                        verified_by=f.get("verified_by", "fst-generate"),
                                        flags=tuple(f["flags"])) for f in fs]
            self._par[(lemma, pos)] = par
        return True
