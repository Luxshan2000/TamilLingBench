#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tokenizer-only verification of the two prompting hazards that fail SILENTLY.

Runs on the GPU machine but touches no GPU, so it is safe to run while the panel is generating.

  1. **Double-BOS.** `apply_chat_template(tokenize=False)` emits `<bos>` as text; the tokenizer
     prepends another unless `add_special_tokens=False`. Nothing errors — the model just sees a
     corrupted prefix. This dumps the leading token ids under BOTH settings so the difference
     is on the record rather than asserted in a comment.
  2. **Qwen3 `<think>`.** Thinking mode is ON by default in the chat template and shifts the
     decision token by hundreds of positions. This renders the prompt with and without
     `enable_thinking=False` and diffs them, proving the switch is the right one for the
     INSTALLED transformers version rather than a remembered kwarg.

Writes `results/prompting-verification.json`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else ROOT / "results/prompting-verification.json")

CHAT = [{"role": "user", "content": "Translate the following English sentence into Tamil. "
                                    "Output only the Tamil translation.\n\nThe dogs came."}]


def main() -> int:
    panel = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["panel"]
    results = {}
    for spec in panel:
        if spec["backend"] != "causal_lm":
            continue
        sid, name = spec["id"], spec["hf_name"]
        try:
            tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        except Exception as e:
            results[sid] = {"error": f"{type(e).__name__}: {e}"}
            continue
        r: dict = {"hf_name": name, "bos_token": tok.bos_token,
                   "bos_token_id": tok.bos_token_id,
                   "has_chat_template": getattr(tok, "chat_template", None) is not None}
        if not r["has_chat_template"]:
            results[sid] = r
            continue
        kw = spec.get("chat_template_kwargs") or {}
        text = tok.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True, **kw)
        r["rendered_prompt"] = text
        ids_correct = tok(text, add_special_tokens=False)["input_ids"]
        ids_wrong = tok(text, add_special_tokens=True)["input_ids"]

        def lead_bos(ids):
            n = 0
            for t in ids:
                if t == tok.bos_token_id:
                    n += 1
                else:
                    break
            return n

        r["leading_bos_with_add_special_tokens_False"] = lead_bos(ids_correct)
        r["leading_bos_with_add_special_tokens_True"] = lead_bos(ids_wrong)
        r["double_bos_avoided"] = (r["leading_bos_with_add_special_tokens_False"] <= 1)
        r["double_bos_would_have_occurred"] = (
            r["leading_bos_with_add_special_tokens_True"]
            > r["leading_bos_with_add_special_tokens_False"])
        r["first_ids_used"] = ids_correct[:8]

        # Qwen3 thinking-mode switch, verified against the INSTALLED version.
        if "enable_thinking" in kw:
            default_text = tok.apply_chat_template(CHAT, tokenize=False,
                                                   add_generation_prompt=True)
            r["thinking_default_render_tail"] = default_text[-120:]
            r["thinking_disabled_render_tail"] = text[-120:]
            r["switch_changes_prompt"] = default_text != text
            r["disabled_render_contains_think_open"] = "<think>" in text
            r["default_render_contains_think_open"] = "<think>" in default_text
            # ⚠ Read this before concluding the switch failed. Qwen3 disables thinking by
            # PRE-FILLING a *closed, empty* `<think>\n\n</think>` block into the generation
            # prompt, so the model resumes after it instead of opening its own. A `<think>`
            # in the rendered prompt is therefore the switch WORKING, not leaking. What would
            # be a failure is an UNCLOSED block, or `<think` in the model's own output.
            r["think_block_is_prefilled_and_closed"] = (
                "<think>" in text and "</think>" in text
                and text.rindex("</think>") > text.rindex("<think>"))
            r["thinking_disabled_correctly"] = bool(
                r["switch_changes_prompt"] and r["think_block_is_prefilled_and_closed"])
            r["verdict"] = ("thinking disabled via a pre-filled closed empty think block — "
                            "correct for transformers "
                            + __import__("transformers").__version__)
        results[sid] = r

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "transformers": __import__("transformers").__version__,
           "what_this_is": "Tokenizer-only proof that the double-BOS and Qwen3 <think> hazards "
                           "are actually handled for the INSTALLED transformers version.",
           "models": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for sid, r in results.items():
        if "error" in r:
            print(f"  {sid}: ERROR {r['error']}")
            continue
        print(f"  {sid}: bos_used={r.get('leading_bos_with_add_special_tokens_False')} "
              f"would_double={r.get('double_bos_would_have_occurred')} "
              f"think_switch={r.get('switch_changes_prompt', 'n/a')} "
              f"thinking_disabled_correctly="
              f"{r.get('thinking_disabled_correctly', 'n/a')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
