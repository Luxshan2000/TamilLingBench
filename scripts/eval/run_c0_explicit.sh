#!/bin/bash
# Corrected-C0 capability ceiling for every LLM in the panel.
#
# The shipped c0_instruction names the benchmark's internal label code ("realising the
# clusivity value 'EXCL' on the pron"), which no model can interpret — and C0 gates every
# default-filling claim. P0_explicit states the target morpheme, copied off the item's own
# gold_targets. 154 items per model, so this is minutes, and it is what makes the C0 gate
# trustworthy. Waits for both generation sessions so nothing contends for the GPU.
cd "$(dirname "$0")/../.." || exit 1
# Activate the GPU environment first, or point VENV at it.
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
set -u
EXP="${EXP:-outputs}"
ITEMS="${ITEMS:-data/benchmark/items.jsonl}"
WAIT_SESSIONS="${WAIT_SESSIONS:-eval it2}"
while :; do
  busy=0
  for s in $WAIT_SESSIONS; do tmux has-session -t "$s" 2>/dev/null && busy=1; done
  [ "$busy" -eq 0 ] && break
  sleep 60
done
echo "############ $(date -u +%FT%TZ) starting corrected-C0 arm"
MODELS="${MODELS:-gemma3-1b gemma3n-e2b gemma3-4b gemma3n-e4b qwen3-8b gemma3-12b llama31-8b qwen3-14b sarvam-translate gemma3-27b qwen3-32b}"
for M in $MODELS; do
  python scripts/eval/generate.py --model "$M" --batch-size 16 --prompt-set c0_explicit \
      --conditions C0 --items "$ITEMS" --out "$EXP/raw" --reports "$EXP/run_manifests" \
    || echo "!!! FAILED c0 $M"
done
echo "############ $(date -u +%FT%TZ) C0-EXPLICIT DONE"
