#!/usr/bin/env bash
# Build the three GENERATION nets.
#
# Generation is NOT done against build/thamizhi-union.bin. Two mechanical reasons, both
# established by execution:
#
#   1. The union holds TWO nets (merged lexicon + verb-guess). `flookup` without `-a` runs
#      nets in COMPOSITION, so `flookup -i union.bin` pushes the analysis through the lexicon
#      and then through the guesser and returns `+?` for everything.
#
#   2. Worse and subtler: the checker's merged lexicon includes pronoun.fst, whose sigma
#      contains the MULTICHARACTER symbol `+1pl` (from `நாம்+pron+1pl+incl+nom`). The verb
#      nets spell 1st-plural agreement as the plain characters `+`,`1`,`p`,`l`. Once the two
#      are unioned, flookup tokenizes the input `...+1pl=ஓம்` to the multichar symbol, the
#      verb path cannot accept it, and `வந்தோம்` becomes ungenerable — while `வந்தான்`
#      generates fine, so the failure looks like a lexicon gap rather than a symbol clash.
#      Keeping the verb nets in their own binary is the fix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="$ROOT/third_party/thamizhi-morph/FST-Models"
mkdir -p "$ROOT/build"
cd "$M"
foma <<EOF
load stack verb-c3.fst
load stack verb-c4.fst
load stack verb-c11.fst
load stack verb-c12.fst
load stack verb-c62.fst
load stack verb-c-rest.fst
union net
save stack $ROOT/build/thamizhi-verbs.bin
EOF
cp "$M/pronoun.fst" "$ROOT/build/thamizhi-pron.bin"
cp "$M/noun.fst"    "$ROOT/build/thamizhi-nouns.bin"
for f in thamizhi-verbs thamizhi-pron thamizhi-nouns; do
  echo "built build/$f.bin ($(wc -c < "$ROOT/build/$f.bin" | tr -d ' ') bytes)"
done
