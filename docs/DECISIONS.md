# Design decisions and native-speaker rulings

Comments in the code, notes inside the item records and some report fields cite decisions by
ID (for example `D-2` or `D-7.1`). This file says what each ID means. Rulings marked
*native-speaker ruling* were made by a native speaker of Tamil and take precedence over
corpus counts and over the reference grammars.

## D-1: Open-weight systems only

The panel uses open-weight models and dedicated MT systems. No commercial API model is
evaluated. This keeps every run reproducible from pinned revisions, and it means no
benchmark item is ever sent to a third-party service, so the blind split cannot leak through
API logging. The cost is that the paper cannot say whether a frontier system behaves
differently. The Gemma 3 size series (1B to 27B) is the partial substitute.

## D-2: Scoring ambiguous forms (context-filtered universal)

Some Tamil forms have more than one analysis. For example, வந்தார்கள் can be an honorific
singular or a plural. The ambiguity belongs to the analyser, not to the item, because every
item declares how many referents it has. Scoring therefore works in four steps:

1. **Context filter.** Keep only the analyses that agree with the item's declared referent
   number.
2. **Universal over what survives.** The output counts as correct only if *every* surviving
   analysis matches the target. This is the headline number.
3. **Existential as a robustness column.** Correct if *any* surviving analysis matches.
   Reported beside the headline, never instead of it.
4. **Undecidable items are reported, not dropped.** When the filter cannot decide, the item
   is marked `undecidable` and counted with its own denominator.

The strict policy is the headline. The lenient policy shows the conclusion does not depend on
that strictness.

## D-4: Release terms and the blind split

Data are CC BY-SA 4.0. This licence is required, not chosen: some material derives from
UD_Tamil-MWTT, which is ShareAlike. Code is Apache-2.0. UD_Tamil-TTB (CC BY-NC-SA 3.0) is
kept out of the release, and `scripts/check_release_licence.py` checks for leakage from it.

About 30% of template families are held out as a blind split. The holdout is at the template
level because the generator is public: anyone could regenerate held-out *items*, but not
items from templates they have never seen. The blind split is not in this repository.

## D-5: The corpus prior is a range over three corpora

Output distributions are compared against a Tamil corpus prior pooled from Tamil Wikipedia,
AI4Bharat Sangraha (verified Tamil) and CC-100 Tamil. The prior is reported as a
between-corpus range, never a single number, because the choice of corpus moves it more than
the number of tokens counted. What can be claimed differs by slot:

- **Clusivity:** usable. The inclusive form is the majority in every corpus.
- **Gender:** usable with a bracket. The feminine rate is low everywhere (3.0 to 12.2%).
- **Rationality:** the bracket is mandatory, since the range crosses 50%.
- **Honorificity:** not reportable from corpus counts. நீங்கள் is ambiguous between
  honorific singular and plain plural, தாங்கள் is overwhelmingly the third-person reflexive
  in running text, and verb-borne second-person forms are rare in written Tamil.

For the prompted models the honest wording is that output rates *deviate from general Tamil
text frequency*. Their training mixtures are not public, so we cannot say the rates deviate
from their own training frequency.

## D-6: Native-speaker rulings, first round

**D-6.1 Honorificity is binary in the indicative** *(native-speaker ruling)*. Second-person
address groups as {நீ, நீர்} familiar against {நீங்கள், தாங்கள்} polite. நீர் is a variant
of நீ, and தாங்கள் a variant of நீங்கள். There is no third degree, so the honorificity
indicative families are two-way with 50% chance. Either member of a group realises its
value.

**D-6.2 Regional variants (overruled by D-9).** An earlier reading accepted both -அன and
-அது for plural non-human subjects as a regional difference. The released gold lists still
carry the -அது alternative, tagged `variety`; D-9 overrules it (see the known issues in
`data/benchmark/manifest.json`).

**D-6.3 Two forms that had zero corpus attestations are valid Tamil** *(native-speaker
ruling)*. நீந்தினாள் ("she swam") is fine, and its zero count is a gap in the corpus, not a
sign that the form is ungrammatical. ஓடினது is acceptable alongside ஓடியது.

**D-6.4 Annotation package.** The native-speaker verification task was split by slot so that
several annotators could work in parallel without reducing the task.

## D-7: Native-speaker rulings, second round

**D-7.1 ஓடின is a finite verb** *(native-speaker ruling)*. "நாய்கள் ஓடின" is a complete
sentence. The off-the-shelf analyser read ஓடின only as an adjectival participle, so it scored
correct Tamil as wrong. The checker was repaired to add the finite third-person plural
non-human reading for the -இன் past class while keeping the participle reading. This is the
fifth analyser repair described in the paper; the released scoring in `outputs/scored` uses
it, and `outputs/scored_before_repair` shows the effect.

**D-7.2 Optatives are ordinary written Tamil** *(native-speaker ruling)*. வருக is well
attested in the corpora (381 occurrences). இறங்குக, ஏறுக, கழுவுக and நீந்துக have no
occurrences, which reflects corpus coverage rather than grammaticality.

**D-7.3 The imperative is three-way** *(native-speaker ruling)*. வருக is a third, higher
level above வாருங்கள். So second-person address is binary in the indicative (D-6.1) and
ternary in the imperative (வா / வாருங்கள் / வருக). Chance rates are therefore set per
template family, not per slot: 33.3% for the imperative family, 50% for the others.

**D-7.4 அறம், அருள், அன்பு and அருளி are unisex names** *(native-speaker ruling)*, so gender
items built on them carry no gender cue.

## D-8: The number control uses inanimate referents

For plural inanimate subjects (விமானங்கள்), the verb must be plural: வந்தன, not வந்தது. A
number control built on inanimate referents therefore cannot mark a correct Tamil form wrong.
The released number items use human referents instead, which is why that control cannot be
read (known issues, `data/benchmark/manifest.json`).

## D-9: -அது is singular and -அன is plural, with no regional split

*Native-speaker ruling; overrules D-6.2.* -அது is the non-human (அஃறிணை) singular and -அன
the non-human plural, for animals and objects alike. நாய்கள் வந்தது is ungrammatical: a
plural subject with a singular verb. The regional framing came from a single citation and was
never confirmed by a speaker. Only -அன is correct for a plural non-human subject; the -அது
alternatives tagged `variety` in the released gold lists should be treated as over-accepted.
