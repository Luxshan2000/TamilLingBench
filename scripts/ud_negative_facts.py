#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count the UD annotation negative results directly from the CoNLL-U files.

These are the annotation facts the paper opens with (its Table of UD counts). Every one is a
direct FEATS count over the released CoNLL-U files. They are the strongest available motivation because
they are annotation facts rather than arguments, so a reviewer cannot dispute them --
but that only holds if we count them ourselves rather than quoting a number from a note.

Writes results/ud-negative-facts.json; the paper's numbers are read from that file, never
retyped.

LICENCE: UD_Tamil-TTB is CC BY-NC-SA 3.0. This script emits *counts only* and no TTB
text, which is what the release gate in scripts/check_release_licence.py permits.
UD_Tamil-MWTT is CC BY-SA 4.0.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UD = ROOT / "data" / "ud"

TTB = sorted((UD / "UD_Tamil-TTB").glob("*.conllu"))
MWTT = sorted((UD / "UD_Tamil-MWTT").glob("*.conllu"))

# The 1PL pronouns whose FEATS are claimed to be byte-identical in TTB.
NAAM = "நாம்"            # nam   (inclusive)
NAANGAL = "நாங்கள்"  # nankal (exclusive)
# The negation strategies claimed to be mislabelled Polarity=Pos.
ILLAI_FORMS = ("வில்லை",  # -villai
               "இல்லை")        # illai
# The masculine finite exponents claimed absent from TTB.
AAN = "ான்"   # -an
AAL = "ாள்"   # -al


def read_tokens(paths):
    """Yield (form, upos, feats_dict) for every real token (no MWT ranges, no empties)."""
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 10 or "-" in f[0] or "." in f[0]:
                continue
            feats = {}
            if f[5] != "_":
                for kv in f[5].split("|"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        feats[k] = v
            yield f[1], f[3], feats


def feat_counts(tokens, key):
    return Counter(fe[key] for _, _, fe in tokens if key in fe)


def main() -> int:
    if not TTB or not MWTT:
        raise SystemExit(f"CoNLL-U files not found under {UD}")

    ttb = list(read_tokens(TTB))
    mwtt = list(read_tokens(MWTT))
    both = ttb + mwtt

    out: dict[str, object] = {
        "_source": "scripts/ud_negative_facts.py",
        "_licence_note": (
            "Counts only. UD_Tamil-TTB is CC BY-NC-SA 3.0 and no TTB text is reproduced "
            "here or anywhere in the release path. UD_Tamil-MWTT is CC BY-SA 4.0."
        ),
        "_treebank_files": {
            "ttb": [p.name for p in TTB],
            "mwtt": [p.name for p in MWTT],
        },
        "ud_ttb_tokens": len(ttb),
        "ud_mwtt_tokens": len(mwtt),
    }

    # --- Fact 1: TTB has no Gender=Fem, and no -an/-al finite verbs -------------------
    ttb_gender = feat_counts(ttb, "Gender")
    mwtt_gender = feat_counts(mwtt, "Gender")
    out["ud_ttb_gender_fem_tokens"] = ttb_gender.get("Fem", 0)
    out["ud_ttb_gender_masc_tokens"] = ttb_gender.get("Masc", 0)
    out["ud_mwtt_gender_fem_tokens"] = mwtt_gender.get("Fem", 0)
    out["ud_mwtt_gender_masc_tokens"] = mwtt_gender.get("Masc", 0)
    out["ud_ttb_aan_aal_finite_verbs"] = sum(
        1 for form, upos, fe in ttb
        if upos in ("VERB", "AUX")
        and fe.get("VerbForm") == "Fin"
        and (form.endswith(AAN) or form.endswith(AAL))
    )

    # --- Fact 2: Clusivity in zero tokens; nam/nankal byte-identical FEATS ------------
    out["ud_clusivity_tokens_both_treebanks"] = sum(
        1 for _, _, fe in both if "Clusivity" in fe
    )
    naam_feats = {tuple(sorted(fe.items())) for form, _, fe in both if form == NAAM}
    nangal_feats = {tuple(sorted(fe.items())) for form, _, fe in both if form == NAANGAL}
    out["ud_naam_tokens"] = sum(1 for form, _, _ in both if form == NAAM)
    out["ud_naangal_tokens"] = sum(1 for form, _, _ in both if form == NAANGAL)
    out["ud_naam_naangal_feats_identical"] = bool(
        naam_feats and nangal_feats and naam_feats == nangal_feats
    )

    # --- Fact 3: -villai/illai labelled Polarity=Pos in all occurrences ---------------
    illai = [(form, fe) for form, _, fe in both
             if any(form.endswith(x) for x in ILLAI_FORMS)]
    out["ud_illai_tokens_both_treebanks"] = len(illai)
    illai_pol = Counter(fe.get("Polarity", "(none)") for _, fe in illai)
    out["ud_illai_polarity_values"] = dict(illai_pol)
    out["ud_illai_all_polarity_pos"] = bool(
        illai and all(fe.get("Polarity") == "Pos" for _, fe in illai)
    )

    # --- Facts 5-9 (survey 8.1): zero-token features across both treebanks ------------
    for feat in ("Aspect", "Deixis", "Evident"):
        out[f"ud_{feat.lower()}_tokens_both_treebanks"] = sum(
            1 for _, _, fe in both if feat in fe
        )
    polite = feat_counts(both, "Polite")
    out["ud_polite_form_tokens"] = polite.get("Form", 0)
    out["ud_polite_infm_tokens"] = polite.get("Infm", 0)
    out["ud_polite_elev_tokens"] = polite.get("Elev", 0)
    out["ud_polite_distinct_values"] = len(polite)

    voice = feat_counts(both, "Voice")
    out["ud_voice_act_tokens"] = voice.get("Act", 0)
    out["ud_voice_pass_tokens"] = voice.get("Pass", 0)
    out["ud_voice_cau_tokens"] = voice.get("Cau", 0)

    mood = feat_counts(both, "Mood")
    out["ud_mood_values"] = dict(mood)
    out["ud_mood_ind_tokens"] = mood.get("Ind", 0)
    out["ud_mood_pot_nec_des_tokens"] = (
        mood.get("Pot", 0) + mood.get("Nec", 0) + mood.get("Des", 0)
    )

    # --- The framing count: how many of the categories are unexpressable -------------
    # A category counts as unexpressable in the standard annotation iff the feature that
    # would carry it occurs in zero tokens, or occurs with only one value so no contrast
    # can be encoded. Computed, not asserted.
    unexpressable = {
        "honorific contrast": out["ud_polite_distinct_values"] <= 1,
        "clusivity": out["ud_clusivity_tokens_both_treebanks"] == 0,
        "deixis": out["ud_deixis_tokens_both_treebanks"] == 0,
        "aspect": out["ud_aspect_tokens_both_treebanks"] == 0,
        "causation": out["ud_voice_cau_tokens"] == 0,
        "modality": out["ud_mood_pot_nec_des_tokens"] == 0,
        "evidentiality": out["ud_evident_tokens_both_treebanks"] == 0,
    }
    out["ud_unexpressable_categories"] = sorted(k for k, v in unexpressable.items() if v)
    out["ud_n_unexpressable_categories"] = sum(unexpressable.values())

    # MWTT is the validation treebank: report what it *does* have, since the paper uses
    # it as the gold set precisely because it exemplifies paradigms.
    out["ud_mwtt_aan_verb_aux_tokens"] = sum(
        1 for form, upos, _ in mwtt if upos in ("VERB", "AUX") and form.endswith(AAN)
    )
    out["ud_mwtt_polarity_neg_tokens"] = sum(
        1 for _, _, fe in mwtt if fe.get("Polarity") == "Neg"
    )

    dest = ROOT / "results" / "ud-negative-facts.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")
    print(f"  unexpressable categories ({out['ud_n_unexpressable_categories']}): "
          f"{', '.join(out['ud_unexpressable_categories'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
