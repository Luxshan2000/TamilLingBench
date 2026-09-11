#!/usr/bin/env bash
# Build build/thamizhi-union.bin — the 2-net foma stack the checker drives.
#
# WHY THIS SCRIPT EXISTS. Three mechanical facts force this exact recipe:
#   1. `flookup` reads exactly ONE file. `flookup -a lex.fst guess.fst` SILENTLY ignores the
#      second argument's nets; `cat a.fst b.fst > u.bin` yields "File format error".
#   2. `flookup -a` is PRIORITY union: the first net that produces output wins and later nets
#      never run. That is what we want for guesser-as-fallback and catastrophic among lexicon
#      nets — under -a, `நீ` returns only the verb imperative and the 2sg pronoun reading is
#      suppressed, which would corrupt every honorificity item.
#   3. The fix is foma's `union net`: collapse the lexicon nets into ONE net that returns ALL
#      analyses, then push the guesser on top as net #2 so -a still means guesser-as-fallback.
#
# This is the ONLY sanctioned way to build the binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/third_party/thamizhi-morph/FST-Models"
OUT="$ROOT/build/thamizhi-union.bin"

mkdir -p "$ROOT/build"

for f in pronoun verb-c3 verb-c4 verb-c11 verb-c12 verb-c62 verb-c-rest verb-guess; do
  [ -f "$MODELS/$f.fst" ] || { echo "missing $MODELS/$f.fst" >&2; exit 1; }
done

LEX="$ROOT/build/thamizhi-lexicon.bin"

cd "$MODELS"

# (a) the merged LEXICON net alone. The checker drives this one, because `flookup -a` does
#     not report WHICH net produced an analysis and the guesser emits no suffix bindings —
#     so source attribution ("fst-lexicon" vs "fst-guesser") has to come from *which file
#     answered*, not from a `=`-in-the-string heuristic. That heuristic is wrong: pronoun.fst
#     is a lexicon net and its analyses (நாம்+pron+1pl+incl+nom) contain no `=` either.
foma <<EOF
load stack pronoun.fst
load stack verb-c3.fst
load stack verb-c4.fst
load stack verb-c11.fst
load stack verb-c12.fst
load stack verb-c62.fst
load stack verb-c-rest.fst
union net
save stack $LEX
EOF

# (b) the 2-net union, kept as the frozen published artifact and as the
#     subject of the checker tests. Python-side priority union over (a) + verb-guess.fst
#     is equivalent; tests/test_checker.py asserts that equivalence rather than assuming it.
foma <<EOF
load stack pronoun.fst
load stack verb-c3.fst
load stack verb-c4.fst
load stack verb-c11.fst
load stack verb-c12.fst
load stack verb-c62.fst
load stack verb-c-rest.fst
union net
load stack verb-guess.fst
save stack $OUT
EOF

echo "built $LEX ($(wc -c < "$LEX" | tr -d ' ') bytes)"
echo "built $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
