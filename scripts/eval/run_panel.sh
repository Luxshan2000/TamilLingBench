#!/bin/bash
# One model per PROCESS. A Python loop would leak GPU memory between models.
# Smallest first, so a harness bug surfaces on the cheapest model.
#
# Everything is written under $EXP (default outputs/), flushed per cell, so a run that dies at
# model 7 of 12 leaves the first 6 on disk and syncable.
#
# `EXP` (output root) and `ITEMS` default to outputs/ and the released items. To keep a new
# run apart from the released outputs, point EXP elsewhere:
#   EXP=outputs/my_run bash scripts/eval/run_panel.sh
cd "$(dirname "$0")/../.." || exit 1
# Activate the GPU environment first, or point VENV at it.
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
set -u

EXP="${EXP:-outputs}"
ITEMS="${ITEMS:-data/benchmark/items.jsonl}"
mkdir -p "$EXP/raw" "$EXP/run_manifests" "$EXP/scored"

MODELS="${MODELS:-gemma3-1b gemma3n-e2b gemma3-4b gemma3n-e4b qwen3-8b nllb-3.3b gemma3-12b llama31-8b qwen3-14b sarvam-translate gemma3-27b qwen3-32b}"
BS="${BS:-64}"
# the robustness run is a FIXED STRATIFIED SUBSET (600 items = 5 slots x 3
# conditions x 40, fixed seed, balanced across set_id), not the whole benchmark. The two
# smallest models were run on every item; everything above that uses the subset, which is
# what the mechanical robustness rule is defined over.
SUBSET="${SUBSET:-configs/prompt_robustness_subset.txt}"

echo "############ items=$ITEMS  exp=$EXP  sha256=$(sha256sum "$ITEMS" | cut -c1-16)"

for M in $MODELS; do
  echo "############ $(date -u +%FT%TZ) canonical $M"
  python scripts/eval/generate.py --model "$M" --batch-size "$BS" --items "$ITEMS" \
      --prompt-set canonical --out "$EXP/raw" --reports "$EXP/run_manifests" \
    || echo "!!! FAILED canonical $M"
  echo "############ $(date -u +%FT%TZ) robustness $M"
  python scripts/eval/generate.py --model "$M" --batch-size "$BS" --items "$ITEMS" \
      --prompt-set robustness --robustness-subset "$SUBSET" \
      --out "$EXP/raw" --reports "$EXP/run_manifests" \
    || echo "!!! FAILED robustness $M"
done
echo "############ $(date -u +%FT%TZ) PANEL DONE"
