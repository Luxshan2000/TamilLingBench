#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The LLM-prompted metric arm, testing pre-registered prediction P2.

⚠ THIS IS NOT GEMBA-ESA AND IS NEVER REPORTED AS GEMBA-ESA.

GEMBA-ESA as published (Kocmi & Federmann; `MicrosoftTranslator/GEMBA`) is a GPT-4-class prompted
metric and needs an `OPENAI_API_KEY`. `DECISIONS.md` D-1 removed the frontier-API arm from this
project, so the published metric cannot be run. Reporting P2 as
simply "untested" would leave a pre-registered prediction unexamined when a defensible test is
available, so this runs the SAME TWO-STAGE PROTOCOL with an open-weight judge and labels the
result `gembaesa_openweight` everywhere.

What is faithful to GEMBA-ESA:
  * two stages -- error-span elicitation first, then a 0-100 score conditioned on those spans;
  * reference-free (source and hypothesis only);
  * the stage-1 spans are kept, because they are the second localisation signal Step 7 wants;
  * no Indic few-shot exemplar, matching upstream, whose exemplars are en->de / en->cs / zh->en
    only. That absence is itself worth a sentence in the paper.

What differs, and must be stated wherever this number appears:
  * the judge is an open-weight model, not GPT-4o;
  * the judge is chosen from OUTSIDE the panel under test, so no system grades its own output.

Determinism: greedy decoding, no sampling. The model snapshot revision is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(os.environ.get("MB_BASE",
                           Path(__file__).resolve().parent.parent / "outputs/metric_blindness"))
IN = BASE / "inputs"
RAW = BASE / "raw"

SPAN_PROMPT = """You are annotating a machine translation for errors.

Source (English): {src}
Translation (Tamil): {mt}

Based on the source segment and machine translation surrounded with triple backticks, identify \
error types in the translation and classify them. The categories of errors are: accuracy \
(addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, \
inconsistency, punctuation, register, spelling), style (awkward), terminology \
(inappropriate for context, inconsistent use) or non-translation.

List each error on its own line in the form
  <error span> | <category> | <major or minor>
List nothing else. If there are no errors, write exactly: NO ERRORS
"""

SCORE_PROMPT = """You are scoring a machine translation.

Source (English): {src}
Translation (Tamil): {mt}
Errors identified:
{spans}

Based on the source segment, the machine translation and the annotated error spans, score the \
translation with one number from 0 to 100, where 0 means "no meaning preserved" and 100 means \
"perfect meaning and grammar".

Reply with the number only.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B",
                    help="If the judge IS a system under test, pass --exclude-system so it never "
                         "grades its own output.")
    ap.add_argument("--exclude-system", default=None,
                    help="system_id whose records are dropped from the judged subsample. Set "
                         "this to the judge's own system_id: the pre-registration names "
                         "self-judging as a confound, and excluding the judge's own outputs "
                         "removes it outright rather than caveating it.")
    ap.add_argument("--n-per-slot", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = [json.loads(l) for l in open(BASE / "scoring_set.jsonl")]
    # Stratified, deterministic subsample: the first n per (arm, slot) in file order.
    seen, sub = {}, []
    for r in recs:
        if r["arm"] not in ("controlled", "avoidance"):
            continue
        if args.exclude_system and r["system_id"] == args.exclude_system:
            continue
        k = (r["arm"], r["slot"])
        if seen.get(k, 0) >= args.n_per_slot:
            continue
        seen[k] = seen.get(k, 0) + 1
        sub.append(r)
    print(f"judging {len(sub)} records: " + ", ".join(f"{a}|{s}={n}" for (a, s), n in seen.items()),
          flush=True)

    variants = ["good", "bad"] + (["avoided"] if any("avoided" in r["variants"] for r in sub) else [])
    jobs = [(i, v) for i, r in enumerate(sub) for v in variants if v in r["variants"]]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    # ⛔ Two load-time traps on this box, both hit:
    #  1. `device_map="auto"` silently CPU-OFFLOADS when another job holds memory and the run then
    #     crawls without erroring ("Some parameters are on the meta device..."; 160 prompts in
    #     28 minutes). Pin to the GPU and assert nothing landed on `meta`.
    #  2. On UNIFIED memory the host and device copies come out of the same 121 GB, so loading a
    #     32B bf16 checkpoint needs ~2x64 GB and is OOM-killed (exit 137) at ~89% of shards even
    #     with 111 GB "available". Keep the judge under ~half of total memory.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map={"": 0})
    model.eval()
    offloaded = [n for n, p_ in model.named_parameters() if p_.device.type == "meta"]
    if offloaded:
        raise SystemExit(f"{len(offloaded)} parameters were offloaded to CPU/meta "
                         f"(first: {offloaded[0]}). Free GPU memory and re-run; a partially "
                         f"offloaded judge is not slow-but-correct, it is unusable.")
    rev = getattr(model.config, "_commit_hash", None)

    def generate(prompts: list[str], max_new: int) -> list[str]:
        outs = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip())
            if i % (args.batch_size * 10) == 0:
                print(f"  {i}/{len(prompts)}", flush=True)
        return outs

    print("=== stage 1: error spans ===", flush=True)
    spans = generate([SPAN_PROMPT.format(src=sub[i]["source"], mt=sub[i]["variants"][v])
                      for i, v in jobs], args.max_new_tokens)
    print("=== stage 2: score ===", flush=True)
    scores_txt = generate([SCORE_PROMPT.format(src=sub[i]["source"], mt=sub[i]["variants"][v],
                                               spans=s or "NO ERRORS")
                           for (i, v), s in zip(jobs, spans)], 8)

    out = []
    n_unparsed = 0
    for (i, v), sp, st in zip(jobs, spans, scores_txt):
        m = re.search(r"\d{1,3}", st)
        val = float(m.group()) if m else None
        if val is None:
            n_unparsed += 1
        r = sub[i]
        out.append({"item_id": r["item_id"], "system_id": r["system_id"],
                    "prompt_id": r["prompt_id"], "arm": r["arm"], "slot": r["slot"],
                    "variant": v, "score": val, "raw_score": st, "spans_raw": sp,
                    "no_errors": sp.strip().upper().startswith("NO ERRORS")})
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "gembaesa_openweight.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
    (RAW / "gembaesa_openweight.meta.json").write_text(json.dumps({
        "metric": "gembaesa_openweight",
        "NOT_GEMBA_ESA": "GEMBA-ESA as published requires GPT-4-class access via OPENAI_API_KEY; "
                         "D-1 removed the frontier-API arm. This is the same two-stage protocol "
                         "with an open-weight judge and must never be labelled GEMBA-ESA.",
        "judge_model": args.model, "judge_revision": rev,
        "judge_is_under_test": bool(args.exclude_system),
        "excluded_system": args.exclude_system,
        "self_judging": False,
        "decoding": "greedy, do_sample=False",
        "few_shot": "none — upstream GEMBA's exemplars are en-de / en-cs / zh-en; there is no "
                    "Indic exemplar, which is itself reportable",
        "n_records": len(sub), "n_judgements": len(out), "n_unparsed_scores": n_unparsed,
        "toship_key": None,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {len(out)} judgements ({n_unparsed} unparsed scores)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
