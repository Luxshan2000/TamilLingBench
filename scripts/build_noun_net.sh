#!/usr/bin/env bash
# Build build/thamizhi-nouns.bin — ThamizhiMorph's NOUN LEXICON, used by
# `tamillingbench/corpus/slots.py` as a homograph veto when counting finite verb forms.
#
# ⚠ LEXICON ONLY. `noun-guess.fst` is deliberately excluded: a noun *guesser* recognises
# almost any Tamil-shaped string, so vetoing on it would delete the entire verb count.
# The veto's value comes from it being high-precision and low-recall — it removes
# கால்வாய் ('canal', analysed by the verb net as `+fut+2sg=ஆய்`) and ஆண்டாள் (a proper
# name) while touching none of வந்தான்/வந்தாள்/வந்தார்/வந்தாய்/வந்தன/வந்தனர்.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/third_party/thamizhi-morph/FST-Models"
OUT="$ROOT/build/thamizhi-nouns.bin"

mkdir -p "$ROOT/build"
[ -f "$MODELS/noun.fst" ] || { echo "missing $MODELS/noun.fst" >&2; exit 1; }

cd "$MODELS"
foma <<EOF
load stack noun.fst
save stack $OUT
EOF

echo "built $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
