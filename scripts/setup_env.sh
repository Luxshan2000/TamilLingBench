#!/usr/bin/env bash
# Idempotent installer for the CPU/analysis environment.
# Safe to re-run: every step checks before acting.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== foma"
if ! command -v foma >/dev/null || ! command -v flookup >/dev/null; then
  if command -v brew >/dev/null; then brew install foma
  elif command -v apt-get >/dev/null; then sudo apt-get install -y foma-bin
  else echo "install foma manually" >&2; exit 1; fi
fi
foma -v | head -1

echo "== python 3.12 venv"
command -v uv >/dev/null || { echo "uv required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
VIRTUAL_ENV="$ROOT/.venv" uv pip install -q -r requirements-cpu.txt
.venv/bin/python --version

echo "== ThamizhiMorph FST models (~500 KB; the full clone drags ~110 MB we do not need)"
# NB: FST-Models/ReadMe.md documents filenames that DO NOT EXIST (verbs-c3.fst, verb-rest.fst,
# verb-guesses.fst, nouns.fst, noun-guesser.fst). Use this list, not the ReadMe.
mkdir -p third_party/thamizhi-morph/FST-Models
for f in verb-c-rest verb-c3 verb-c4 verb-c11 verb-c12 verb-c62 verb-guess \
         pronoun noun noun-guess part adj adv; do
  t="third_party/thamizhi-morph/FST-Models/$f.fst"
  [ -s "$t" ] || curl -sfL -o "$t" \
    "https://raw.githubusercontent.com/sarves/thamizhi-morph/master/FST-Models/$f.fst"
done
ls third_party/thamizhi-morph/FST-Models/*.fst | wc -l

echo "== UD treebanks pinned at r2.18"
mkdir -p data/ud
for r in UD_Tamil-MWTT UD_Tamil-TTB; do
  [ -d "data/ud/$r" ] || git clone -q --depth 1 \
    "https://github.com/UniversalDependencies/$r.git" "data/ud/$r"
  git -C "data/ud/$r" fetch -q --depth 1 origin tag r2.18 2>/dev/null || true
  git -C "data/ud/$r" checkout -q r2.18 2>/dev/null || true
done

echo "== FST union"
./scripts/build_fst_union.sh >/dev/null
echo "== smoke test"
.venv/bin/python scripts/smoke_test.py
