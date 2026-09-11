#!/bin/bash
# IndicTrans2 runs from its OWN venv (transformers<5), because IndicTransToolkit 1.1.x does
# not import under transformers 5.12.1 and the IndicProcessor is mandatory.
# Waits for the main panel's tmux session to exit so the two never contend for the GPU.
cd "$(dirname "$0")/../.." || exit 1
set -u
# EXP / ITEMS / WAIT_SESSION can be overridden; see run_panel.sh.
EXP="${EXP:-outputs}"
IT2_VENV="${IT2_VENV:-.venv-it2}"   # created by scripts/eval/setup_it2_venv.sh
ITEMS="${ITEMS:-data/benchmark/items.jsonl}"
WAIT_SESSION="${WAIT_SESSION:-eval}"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do sleep 60; done
echo "############ $(date -u +%FT%TZ) main panel finished; starting IndicTrans2"

for PS in canonical robustness; do
  "$IT2_VENV/bin/python" scripts/eval/generate.py --model indictrans2-1b \
      --batch-size 32 --prompt-set "$PS" --items "$ITEMS" \
      --out "$EXP/raw" --reports "$EXP/run_manifests" \
    || echo "!!! FAILED $PS indictrans2-1b"
done

# NMT greedy control: the NMT arm is deployed with beams, the LLM arm greedily.
# Running beams on one and not the other is a confound, so both settings are produced.
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
python scripts/eval/generate.py --model nllb-3.3b --batch-size 32 --prompt-set canonical \
    --items "$ITEMS" --decoding nmt_greedy_control --out "$EXP/raw_greedy" \
    --reports "$EXP/run_manifests" || echo "!!! FAILED nllb greedy control"
"$IT2_VENV/bin/python" scripts/eval/generate.py --model indictrans2-1b --batch-size 32 \
    --prompt-set canonical --items "$ITEMS" --decoding nmt_greedy_control \
    --out "$EXP/raw_greedy" --reports "$EXP/run_manifests" || echo "!!! FAILED it2 greedy control"

echo "############ $(date -u +%FT%TZ) IT2 + greedy control DONE"
