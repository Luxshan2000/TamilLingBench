#!/bin/bash
# IndicTrans2 backend only.
#
# MEASURED, not assumed: under `transformers 5.12.1`, IndicTransToolkit 1.1.x fails
# at import with
#     ImportError: cannot import name 'PreTrainedTokenizerBase' from
#     'transformers.tokenization_utils'
# The processor is MANDATORY (script normalisation, entity placeholdering, language-tag
# prefix); skipping it silently degrades the baseline, so the right response is a pinned venv,
# not a shim. Backends already run in separate processes, so this costs nothing
# architecturally.
#
# `--system-site-packages` is deliberate: the aarch64 torch 2.10.0+cu130 build is
# NVIDIA-provided and a clean venv would resolve to a wrong or missing wheel.
set -eu
VENV="${IT2_VENV:-.venv-it2}"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
# transformers is pinned BELOW the major version that breaks the toolkit; torch is inherited.
"$VENV/bin/pip" install --quiet "transformers>=4.51,<5" "IndicTransToolkit" "sentencepiece" "PyYAML"
"$VENV/bin/python" - <<'PY'
import transformers
print("transformers", transformers.__version__)
try:
    from IndicTransToolkit.processor import IndicProcessor
except Exception:
    from IndicTransToolkit import IndicProcessor
ip = IndicProcessor(inference=True)
out = ip.preprocess_batch(["The train arrives at six o'clock."],
                          src_lang="eng_Latn", tgt_lang="tam_Taml")
print("preprocess round-trip:", out)
assert out and out[0].startswith("eng_Latn"), "IndicProcessor did not prefix the language tag"
print("IT2 VENV OK")
PY
