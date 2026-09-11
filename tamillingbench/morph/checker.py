# -*- coding: utf-8 -*-
"""MorphChecker — the deterministic suffix-level checker.

Given a Tamil surface form, returns the SET of feature-suffix bindings the form licenses
(வந்தான் → `+3sgm=ஆன்`), so that "did the system produce the masculine form?" is answered by
a transducer and not by a regex over string endings.

Two design commitments that fall out of the FST's actual behaviour:

  * `analyse` returns a SET. வந்தார்கள் has two correct analyses (`+3sghe` and `+3ple`).
    Any API that returns one is lying.
  * `check` is UNIVERSAL-AFTER-CONTEXT-FILTERING per DECISIONS.md D-2 — not existential.
    The existential result is computed too, as a reported robustness column, and the
    `undecidable` rate is a reported quantity with its own denominator. All three are
    outputs, not alternatives.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

from .fallback import (RuleHit, apply_fallback, load_overrides, rule_in_class_finite,
                       split_compound, strip_clitic)
from .labels import (NEG, POLARITY_BEARING_POS, POS, PNG_FEATURES, Slot, SUFFIX_SYNCRETISM,
                     TAG_FEATURES, NEGATION_LABELS, ZERO_MORPH, label_suffix_to_surface)
from .normalize import looks_like_split_artifact, normalize

Source = Literal["fst-lexicon", "fst-guesser", "fallback-rule", "override"]

DEFAULT_FST = "build/thamizhi-union.bin"
DEFAULT_LEXICON = "build/thamizhi-lexicon.bin"
DEFAULT_GUESSER = "third_party/thamizhi-morph/FST-Models/verb-guess.fst"

#: What flookup emits for an unanalysable input.
NO_ANALYSIS = "+?"

#: Words written to `flookup` before its output is drained. MUST stay small.
#:
#: ⛔ DEADLOCK, found 2026-08-08 by the corpus-prior stream and reproduced here. `_flookup`
#: writes every word, then reads. Both pipes are 64 KB on macOS: once flookup's stdout buffer
#: fills it blocks on write, and once our stdin buffer fills we block on write, and neither
#: side ever drains the other. The batch is therefore chunked; 250 keeps both directions
#: comfortably under 64 KB for Tamil forms and costs nothing measurable.
FLOOKUP_BATCH = 250


class ArtifactToken(ValueError):
    """R10 — a TTB tokenization artifact. Excluded from denominators, not counted a miss."""


# --------------------------------------------------------------------------- types

@dataclass(frozen=True)
class Binding:
    """One feature label bound to the surface suffix that realises it."""
    label: str      # '3sgm', 'past', 'negpart', 'imp' …
    suffix: str     # 'ஆன்' | 'இல்லை' | '' for a zero morph (the FST writes '∅')
    #: True when the morph starts the word, so its vowel is genuinely independent and must
    #: NOT be converted to a combining sign. Only override/fallback rules set this (the
    #: negative copula அல்ல *is* the whole word); every FST suffix is postconsonantal.
    word_initial: bool = False

    @property
    def surface_suffix(self) -> str:
        """The suffix as it actually appears on the surface — TRAP B.

        `binding.suffix` uses INDEPENDENT vowel letters (ஆ U+0B86); the surface uses
        COMBINING vowel signs (ா U+0BBE). `surface.endswith(binding.suffix)` is therefore
        False for every independent-vowel-initial suffix — measured 0/623 on UD tokens.
        THIS property is the only sanctioned fix; never loosen the match instead.
        """
        if self.word_initial:
            return self.suffix
        return label_suffix_to_surface(self.suffix)

    @property
    def is_zero(self) -> bool:
        return self.suffix in ("", ZERO_MORPH)


@dataclass(frozen=True)
class Analysis:
    lemma: str
    pos: str
    tags: tuple[str, ...]
    bindings: tuple[Binding, ...]
    source: Source
    raw: str                       # exact flookup output (or rule id), kept for audit
    why: str = ""

    @property
    def has_suffix_bindings(self) -> bool:
        """True iff this analysis binds a feature to a real surface string.

        The GUESSER EMITS NO BINDINGS (Trap A): the lexicon gives `+3sgm=ஆன்`,
        the guesser gives bare `+3sgm`. So "coverage" and "suffix-level checking" are
        DIFFERENT NUMBERS and must be reported separately. The metric study's morpheme-edit
        perturbations require this to be True.
        """
        return any(not b.is_zero for b in self.bindings)

    def features(self) -> list[dict[Slot, str]]:
        """Slot-level view. Returns >1 mapping when a tag is genuinely syncretic (2plh)."""
        base: dict[Slot, str] = {}
        readings: list[dict[Slot, str]] = [{}]

        for tag in self.tags:
            if tag in TAG_FEATURES:
                base.update(TAG_FEATURES[tag])

        labels = [b.label for b in self.bindings] + list(self.tags)

        # polarity
        if any(l in NEGATION_LABELS for l in labels):
            base[Slot.POLARITY] = NEG
        elif self.pos in POLARITY_BEARING_POS:
            base[Slot.POLARITY] = POS

        # Repair ThamizhiMorph's inconsistent treatment of -ஆர்/-ஆர்கள் before reading
        # features off the labels: the suffix licenses honorific-sg AND rational-pl, and
        # which one the FST happens to emit varies by lexical entry (see SUFFIX_SYNCRETISM).
        png_labels: tuple[str, ...] = ()
        for b in self.bindings:
            if b.label in PNG_FEATURES and b.suffix in SUFFIX_SYNCRETISM:
                png_labels = SUFFIX_SYNCRETISM[b.suffix]
                break
        if not png_labels:
            for l in labels:
                if l in PNG_FEATURES:
                    png_labels = (l,)
                    break

        # person-number-gender: may fan out into multiple readings
        if png_labels:
            bundles = tuple(bd for l in png_labels for bd in PNG_FEATURES[l])
            readings = [{**r, **b} for r in readings for b in bundles]

        out, seen = [], set()
        for r in readings:
            m = {**base, **r}
            key = tuple(sorted((k.value, v) for k, v in m.items()))
            if key not in seen:
                seen.add(key)
                out.append(m)
        return out


@dataclass(frozen=True)
class Result:
    surface: str            # NORMALIZED surface — what was actually looked up
    surface_raw: str        # exactly what the caller passed in
    analyses: tuple[Analysis, ...]
    iso: str = ""           # ISO 15919 canonical key (lazy; see MorphChecker(iso=True))

    @property
    def ok(self) -> bool:
        return len(self.analyses) > 0

    @property
    def suffix_bound(self) -> bool:
        """True iff at least one analysis is suffix-level, not merely feature-level."""
        return any(a.has_suffix_bindings and a.source in ("fst-lexicon", "fallback-rule",
                                                          "override")
                   for a in self.analyses)

    def features(self) -> list[dict[Slot, str]]:
        out: list[dict[Slot, str]] = []
        for a in self.analyses:
            out.extend(a.features())
        return out


@dataclass(frozen=True)
class CheckResult:
    """DECISIONS.md D-2. All three verdicts are outputs, never alternatives."""
    universal: bool          # HEADLINE — all context-surviving readings match
    existential: bool        # robustness column — some surviving reading matches
    undecidable: bool        # context failed to disambiguate (existential ∧ ¬universal)
    why: str
    n_analyses: int = 0
    n_surviving: int = 0
    n_matching: int = 0
    sources: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Truthiness is the HEADLINE metric, so `if check(...)` cannot silently be lenient."""
        return self.universal


# --------------------------------------------------------------------------- parsing

def _parse_analysis(surface: str, raw: str, source: Source) -> Analysis | None:
    """Parse one flookup analysis string into an Analysis.

    Wire format (verified with `od -c`): `surface \\t analysis`, one line per
    analysis, a BLANK LINE terminates each input word, and unanalysable input yields the
    single analysis string `+?`.

        'வா+verb+fin+sim+strong+past=த்+3sgm=ஆன்'
         └lemma┘ └───────── tags / bindings ─────┘
    """
    if raw == NO_ANALYSIS:
        return None
    parts = raw.split("+")
    lemma, rest = parts[0], parts[1:]
    if not rest:
        return None
    pos = rest[0]
    tags: list[str] = []
    bindings: list[Binding] = []
    for p in rest[1:]:
        if "=" in p:
            label, _, suf = p.partition("=")
            bindings.append(Binding(label, "" if suf == ZERO_MORPH else suf))
        else:
            tags.append(p)
    return Analysis(lemma=lemma, pos=pos, tags=tuple(tags), bindings=tuple(bindings),
                    source=source, raw=raw,
                    why=f"{source}: {raw}")


def _hit_to_analysis(h: RuleHit, source: Source, surface: str) -> Analysis:
    return Analysis(lemma=h.lemma, pos=h.pos, tags=h.tags,
                    bindings=tuple(Binding(l, s, word_initial=surface.startswith(s))
                                   for l, s in h.bindings),
                    source=source, raw=f"<{h.rule}>", why=h.why)


# --------------------------------------------------------------------------- checker

class MorphChecker:
    """Persistent-`flookup` morphological checker.

    `flookup` is a streaming process: one long-lived subprocess is held open rather than
    paying process-spawn cost per word.
    """

    def __init__(self, fst_path: str = DEFAULT_FST, *,
                 lexicon_path: str = DEFAULT_LEXICON, guesser_path: str = DEFAULT_GUESSER,
                 use_guesser: bool = True,
                 use_fallback: bool = True, use_overrides: bool = True,
                 repair_in_class: bool = True,
                 normalize: bool = True, iso: bool = False) -> None:
        self.fst_path = fst_path              # the published 2-net union (criterion 4)
        self.lexicon_path = lexicon_path      # merged lexicon net alone
        self.guesser_path = guesser_path      # verb-guess.fst alone
        self.use_guesser = use_guesser
        self.use_fallback = use_fallback
        self.use_overrides = use_overrides
        #: R14 / D-7.1 — the `-இன்` class அஃறிணை-plural reading. False reproduces the
        #: pre-repair behaviour bit-for-bit, which is what the published ablation needs.
        self.repair_in_class = repair_in_class
        self.normalize = normalize
        self.iso = iso
        self._overrides = load_overrides() if use_overrides else {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._forced: dict[str, dict[Slot, str]] = {}

    # ---------------------------------------------------------------- subprocess
    def _ensure(self, path: str) -> subprocess.Popen:
        p = self._procs.get(path)
        if p is None or p.poll() is not None:
            p = subprocess.Popen(
                ["flookup", "-a", "-b", path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1)
            self._procs[path] = p
        return p

    def _flookup(self, words: Sequence[str], path: str | None = None) -> dict[str, list[str]]:
        """One round-trip for a batch against one net. Returns surface -> [analyses].

        `flookup` is a streaming process, so a single long-lived subprocess per net serves
        the whole run. A BLANK LINE terminates each input word (verified with `od -c`), so
        exactly len(words) blank lines are consumed — this is what keeps the stream in sync
        and must not be replaced by a read-until-idle loop.
        """
        if not words:
            return {}
        if len(words) > FLOOKUP_BATCH:
            out: dict[str, list[str]] = {w: [] for w in words}
            for i in range(0, len(words), FLOOKUP_BATCH):
                for k, v in self._flookup(words[i:i + FLOOKUP_BATCH], path).items():
                    out.setdefault(k, []).extend(v)
            return out
        p = self._ensure(path or self.fst_path)
        assert p.stdin and p.stdout
        out: dict[str, list[str]] = {w: [] for w in words}
        for w in words:
            p.stdin.write(w + "\n")
        p.stdin.flush()
        seen = 0
        while seen < len(words):
            line = p.stdout.readline()
            if line == "":
                raise RuntimeError("flookup died mid-batch")
            line = line.rstrip("\n")
            if line == "":
                seen += 1
                continue
            surf, _, anal = line.partition("\t")
            if anal and anal != NO_ANALYSIS:
                out.setdefault(surf, []).append(anal)
        return out

    # ---------------------------------------------------------------- analysis
    def _norm(self, surface: str) -> str:
        return normalize(surface) if self.normalize else surface

    def is_artifact_token(self, surface: str) -> bool:
        """R10 definitive test: வ்/ய்-stranded shape AND the FST returns nothing.

        Deliberately does NOT additionally require the vowel-restored form to analyse.
        `உருவாகிய்` is a TTB clitic-split artifact whether or not `உருவாகி` happens to be in
        ThamizhiMorph's lexicon; adding that condition conflates "is an artifact" with "is
        an artifact we can independently confirm", and under-counts 16 → 9 on TTB-test.
        Real words ending in ய் (`செய்`, `வாய்`) are analysed by the FST and so never reach
        this branch — the FST-failure conjunct is what keeps the guard safe.
        """
        s = self._norm(surface)
        if not looks_like_split_artifact(s):
            return False
        return not self._flookup([s], self.lexicon_path).get(s) and \
            not self._flookup([s], self.guesser_path).get(s)

    def restored_form_analyses(self, surface: str) -> bool:
        """Auditability helper: does dropping the stranded consonant recover a real word?

        Reported alongside the artifact count so the correction stays falsifiable
        (the 16 forms are listed so the correction is auditable). On TTB-test this is true for 9 of the 16.
        """
        s = self._norm(surface)
        return bool(self._flookup([s[:-2]], self.lexicon_path).get(s[:-2]))

    def analyse(self, surface: str) -> Result:
        return self.analyse_batch([surface])[0]

    def analyse_batch(self, surfaces: Sequence[str]) -> list[Result]:
        """One flookup round-trip per net for the whole batch. Use this in hot loops.

        Priority union is done HERE rather than by relying on `flookup -a`'s net ordering,
        because -a does not report which net answered and source attribution is mandatory
        (Trap A: guesser analyses carry no suffix bindings and must never be
        treated as suffix-level). The semantics are identical and the equivalence is
        asserted in tests/test_checker.py, not assumed.
        """
        normed = [self._norm(s) for s in surfaces]
        uniq = list(dict.fromkeys(normed))
        lex = self._flookup(uniq, self.lexicon_path)
        need_guess = [w for w in uniq if not lex.get(w)]
        guess = (self._flookup(need_guess, self.guesser_path)
                 if (self.use_guesser and need_guess) else {})
        results: list[Result] = []
        for orig, s in zip(surfaces, normed):
            if lex.get(s):
                results.append(self._assemble(orig, s, lex[s], "fst-lexicon"))
            else:
                results.append(self._assemble(orig, s, guess.get(s, []), "fst-guesser"))
        return results

    def _assemble(self, orig: str, s: str, raws: list[str], source: Source) -> Result:
        # --- override layer FIRST: these forms get WRONG analyses, not misses (§8b).
        ovs = self._overrides.get(s, []) if self.use_overrides else []
        replaced = [o for o in ovs if o["mode"] == "replace"]
        augmented = [o for o in ovs if o["mode"] == "augment"]

        analyses: list[Analysis] = []
        if replaced:
            for o in replaced:
                analyses.append(Analysis(
                    lemma=o["lemma"], pos=o["pos"], tags=o["tags"],
                    bindings=tuple(Binding(l, suf, word_initial=s.startswith(suf))
                                   for l, suf in o["bindings"]),
                    source="override", raw=f"<{o['rule']}>",
                    why=f"{o['rule']} override: {o['note']}"))
                self._forced[s] = {**self._forced.get(s, {}), **o["features"]}
        else:
            for r in raws:
                a = _parse_analysis(s, r, source)
                if a is not None:
                    analyses.append(a)

            if not analyses and self.use_fallback:
                for h in apply_fallback(s):
                    analyses.append(_hit_to_analysis(h, "fallback-rule", s))
                    if h.features:
                        self._forced[s] = {**self._forced.get(s, {}), **h.features}

            if not analyses and self.use_fallback:
                analyses = self._rescue(s)

        # R13 — verbal nouns in -வது / -யது. ThamizhiMorph returns +nonfin+vpart with the
        # nominalizer swallowed into the lemma (செய்வது -> 'செய்வ'+vpart), so no PNG reaches
        # the feature layer and the NUMBER control slot fails. The form is a 3sg neuter
        # nominalization. ⚠ Grammatical claim — flagged for native-speaker audit.
        if analyses and s.endswith(("வது", "யது")) and all(
                not any(b.label in ("3sgn", "3pln") for b in a.bindings) for a in analyses):
            self._forced[s] = {**self._forced.get(s, {}), Slot.NUMBER: "SG",
                               Slot.GENDER: "NEUT", Slot.RATIONALITY: "AHRINAI"}
            analyses = [replace(a, why=a.why + " | R13 verbal noun -வது/-யது -> 3sgn")
                        for a in analyses]

        # R14 — the `-இன்` past class's அஃறிணை PLURAL (DECISIONS.md D-7.1). ThamizhiMorph
        # reads `ஓடின` only as `+nonfin+adjpart` and returns no number/rationality, so the
        # correct Tamil `நாய்கள் ஓடின` was scored WRONG. The rule, its guard against the
        # other past classes and its cost are documented in `fallback.rule_in_class_finite`.
        # ⚠ Grammatical claim — NATIVE-CHECK-RAT-6. `repair_in_class=False` ablates it.
        if self.repair_in_class and analyses:
            hit = rule_in_class_finite(s, [a.raw for a in analyses])
            if hit is not None:
                self._forced[s] = {**self._forced.get(s, {}), **hit.features}
                analyses = [replace(a, why=a.why + " | R14 -இன் class: also the அஃறிணை "
                                                   "plural finite form (D-7.1)")
                            for a in analyses]
                analyses.append(_hit_to_analysis(hit, "fallback-rule", s))

        for o in augmented:
            self._forced[s] = {**self._forced.get(s, {}), **o["features"]}
            analyses = [replace(a, why=a.why + f" | {o['rule']} override: {o['note']}",
                                source="override") for a in analyses]

        iso = ""
        if self.iso:
            from aksharamukha import transliterate as _tr
            iso = _tr.process("Tamil", "ISO", s)
        return Result(surface=s, surface_raw=orig, analyses=tuple(analyses), iso=iso)

    def _rescue(self, s: str, _depth: int = 0) -> list[Analysis]:
        """R11 (clitic strip) and R12 (compound split) — rules that need a RE-LOOKUP.

        They live here rather than in fallback.py because they are recursive: strip the
        clitic, then run the whole pipeline again on the host. `_depth` bounds the recursion
        at two peels (a clitic on a compound is the deepest attested case) so a pathological
        input cannot spin.

        The PNG comes from the host / auxiliary, which is correct: Tamil enclitics and
        compound auxiliaries do not alter agreement, they attach outside it.
        """
        if _depth >= 2:
            return []

        for host, label, cl in strip_clitic(s):
            inner = self._analyse_inner(host, _depth + 1)
            if inner:
                return [replace(a, source="fallback-rule",
                                bindings=a.bindings + (Binding(label, cl),),
                                why=f"R11 clitic -{cl} ({label}) stripped -> {host!r}; {a.why}")
                        for a in inner]

        for _main, aux in split_compound(s):
            inner = self._analyse_inner(aux, _depth + 1)
            if inner:
                return [replace(a, source="fallback-rule",
                                why=f"R12 compound: PNG taken from auxiliary {aux!r}; {a.why}")
                        for a in inner]
        return []

    def _analyse_inner(self, w: str, depth: int) -> list[Analysis]:
        """One pipeline pass on an already-normalized string, without re-normalizing."""
        lex = self._flookup([w], self.lexicon_path).get(w, [])
        if lex:
            return [a for a in (_parse_analysis(w, r, "fst-lexicon") for r in lex) if a]
        if self.use_guesser:
            g = self._flookup([w], self.guesser_path).get(w, [])
            if g:
                return [a for a in (_parse_analysis(w, r, "fst-guesser") for r in g) if a]
        hits = apply_fallback(w)
        if hits:
            for h in hits:
                if h.features:
                    self._forced[w] = {**self._forced.get(w, {}), **h.features}
            return [_hit_to_analysis(h, "fallback-rule", w) for h in hits]
        return self._rescue(w, depth)

    # ---------------------------------------------------------------- features
    def features(self, surface: str) -> list[Mapping[Slot, str]]:
        """Slot-level view: one mapping per (analysis x syncretism resolution)."""
        _r, pairs = self._readings(surface)
        return [f for _a, f in pairs]

    def _readings(self, surface: str) -> tuple[Result, list[tuple[Analysis, dict[Slot, str]]]]:
        r = self.analyse(surface)
        forced = self._forced.get(r.surface, {})
        pairs: list[tuple[Analysis, dict[Slot, str]]] = []
        seen: set = set()
        for a in r.analyses:
            for f in a.features():
                m = {**f, **forced}
                # D-2 quantifies over READINGS, not over analyses. Two analyses that yield
                # an identical feature bundle are one reading: after the -ஆர்/-ஆர்கள்
                # syncretism repair the FST's separate 3sghe and 3ple analyses both expand
                # to the same pair, and counting them twice would inflate every denominator.
                key = tuple(sorted((k.value, v) for k, v in m.items()))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((a, m))
        return r, pairs

    # ---------------------------------------------------------------- check (D-2)
    def check(self, surface: str, slot: Slot, expected: str, *,
              referent_number: str | None = None,
              referent_honorificity: str | None = None) -> CheckResult:
        """DECISIONS.md D-2 — universal after context filtering.

        Policy, in order of application:
          1. CONTEXT FILTER. Retain only readings compatible with the item's declared
             referent number (`referent_number` ∈ {'SG','PL',None}) and, where the item pins
             it, its declared honorificity. The ambiguity is in the ANALYSER, not in the item:
             our items specify the number of referents, so context resolves most of it.

             `referent_honorificity` was added 2026-08-08 for the NUMBER control, which is the
             one slot where filtering on number would beg the question: `வந்தார்கள்` is
             3sg-honorific OR 3pl, so a plural number item is undecidable on number alone.
             The design pins `hon = minus` on every number, gender and rationality template
             exactly so that the *other* pinned cell is available as the filter. Filtering a
             number item on number would be circular; filtering it on the honorificity its
             template already fixes is not.
          2. UNIVERSAL over survivors. Correct only if ALL surviving readings assign
             `expected` to `slot`. This is the headline number.
          3. EXISTENTIAL as a robustness column. Computed always, headlined never.
          4. UNDECIDABLE reported separately where context fails to disambiguate.

        A reading that leaves `slot` undefined counts as NON-matching: the form genuinely
        does not encode that feature, which is a clean failure, not an ambiguity.
        """
        r, pairs = self._readings(surface)
        n_analyses = len(pairs)
        if not pairs:
            return CheckResult(False, False, False,
                               f"no analysis for {r.surface!r} (source: none)",
                               0, 0, 0, ())

        survivors = list(pairs)
        applied = []
        if referent_number:
            survivors = [(a, f) for a, f in survivors
                         if f.get(Slot.NUMBER) in (None, referent_number)]
            applied.append(f"referent_number={referent_number}")
        if referent_honorificity:
            survivors = [(a, f) for a, f in survivors
                         if f.get(Slot.HONORIFICITY) in (None, referent_honorificity)]
            applied.append(f"referent_honorificity={referent_honorificity}")
        filt = (f"context filter {', '.join(applied)}: {len(pairs)}→{len(survivors)} readings"
                if applied else "no context filter (no referent features declared)")

        if not survivors:
            return CheckResult(False, False, False,
                               f"{filt}; every reading is incompatible with the declared "
                               f"referent number", n_analyses, 0, 0,
                               tuple(sorted({a.source for a, _ in pairs})))

        matching = [(a, f) for a, f in survivors if f.get(slot) == expected]
        universal = len(matching) == len(survivors)
        existential = len(matching) > 0
        undecidable = existential and not universal

        srcs = tuple(sorted({a.source for a, _ in survivors}))
        verdict = ("UNIVERSAL PASS" if universal
                   else "UNDECIDABLE (context did not disambiguate)" if undecidable
                   else "FAIL")
        why = (f"{verdict}: {slot.value}={expected}; {filt}; "
               f"{len(matching)}/{len(survivors)} surviving readings match; "
               f"sources={','.join(srcs)}; "
               f"readings={[dict((k.value, v) for k, v in f.items()) for _, f in survivors]}")
        return CheckResult(universal, existential, undecidable, why,
                           n_analyses, len(survivors), len(matching), srcs)

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        for p in self._procs.values():
            if p.poll() is None:
                try:
                    if p.stdin:
                        p.stdin.close()
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
        self._procs.clear()

    def __enter__(self) -> "MorphChecker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
