#!/usr/bin/env python
"""Numerical fidelity check.  BLOCKING GATE 3.

**A weight-folding bug looks exactly like "the feature isn't represented yet."**  Nothing
downstream — probe, lenses, patching — may run until this passes in fp32.

200 real Tamil chat-formatted prompts from the benchmark's interp pairs, batch size 1,
raw HF vs TransformerLens on identical token ids, compared at the **decision position**
(the first token index at which the two members of a minimal pair diverge):

  F1  top-1 token agreement                              100 % (200/200)
  F2  mean KL(TL ‖ HF) at the decision position          < 1e-4 nats
  F3  max abs error on the LOGIT DIFFERENCE              < 0.01
  F4  Pearson r between HF and TL logit differences      > 0.999

F3/F4 matter more than F1/F2: the logit difference is the statistic the paper reports, so
it is the quantity that must be faithful.  fp32 is the gate; bf16 is run and recorded
separately, which separates "folding bug" (fails fp32) from "precision" (passes fp32,
drifts in bf16).

An earlier gpt2 fidelity pass (max|HF-TL| = 4.578e-05) does NOT substitute:
gpt2 is ASCII, text-only, has no interleaved attention, no embedding scale, no chat
template.

On failure, run with --bisect: caches both models' per-layer residuals on one prompt and
reports the first layer where they diverge.  Layer 0 => embedding scale; uniform growth =>
LayerNorm folding; a jump at every Nth layer => sliding/global misassignment.

    python scripts/interp/02_fidelity.py --model google/gemma-3-1b-it --n 200
    python scripts/interp/02_fidelity.py --model google/gemma-3-4b-pt --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402

THRESH = {"F1_top1_rate": 1.0, "F2_mean_kl": 1e-4, "F3_max_abs_ld_err": 0.01,
          "F4_pearson_r": 0.999}


# --------------------------------------------------------------------------- prompts

def build_prompts(pairs_path: str, tok, chat_template: str, n: int, seed: int,
                  tok_key: str = "gemma-3-4b"):
    """Chat-formatted Tamil prompts with a per-pair decision position.

    Each item is one forward pass over `prompt + tamil_pos` (batch size 1).  Because the
    model is causal, the logits at the decision position `p` are unaffected by the tokens
    after it, so a single forward gives BOTH a genuinely Tamil-script-bearing context AND
    the exact statistic the study reports.  Feeding only `a[:k]` would leave many prompts
    with no Tamil in them at all (the benchmark's C1 pairs mostly diverge at Tamil token 0).
    """
    with open(pairs_path) as fh:
        pairs = json.load(fh)

    bos = tok.bos_token_id
    rng = random.Random(seed)
    rng.shuffle(pairs)

    out, dropped = [], {"prefix_containment": 0, "no_tamil": 0, "double_bos": 0,
                        "decode_mismatch": 0, "missing_fields": 0}

    for rec in pairs:
        if len(out) >= n:
            break
        src = rec.get("src_en_pos")
        ta_pos, ta_neg = rec.get("tamil_pos"), rec.get("tamil_neg")
        if not (src and ta_pos and ta_neg):
            dropped["missing_fields"] += 1
            continue

        text = B.render_prompt(tok, src, chat_template)
        pids = tok(text, add_special_tokens=False)["input_ids"]

        # Single-BOS assertion.  Gemma's template emits <bos>; a
        # tokenizer that prepends another shifts every position by one.
        if sum(1 for t in pids if t == bos) != 1:
            dropped["double_bos"] += 1
            continue

        ids_a = tok(ta_pos, add_special_tokens=False)["input_ids"]
        ids_b = tok(ta_neg, add_special_tokens=False)["input_ids"]
        k, off = B.divergence_index(pids, ids_a, ids_b)
        if k is None:
            dropped["prefix_containment"] += 1
            continue

        full = pids + ids_a
        p = k - 1                                    # decision position
        if p < 0 or p >= len(full):
            dropped["missing_fields"] += 1
            continue

        # No BPE re-tokenization surprise across the shared prefix.
        if tok.decode((pids + ids_a)[:k]) != tok.decode((pids + ids_b)[:k]):
            dropped["decode_mismatch"] += 1
            continue

        if not any("஀" <= c <= "௿" for c in tok.decode(full)):
            dropped["no_tamil"] += 1
            continue

        out.append({
            "pair_id": rec["pair_id"], "slot": rec["slot"], "condition": rec["condition"],
            "template_id": rec.get("template_id"),
            "ids": full, "p": p, "k": k,
            "tok_pos": (pids + ids_a)[k], "tok_neg": (pids + ids_b)[k],
            "divergence_offset": off, "n_tokens": len(full),
            "n_prompt_tokens": len(pids),
        })
    return out, dropped, len(pairs)


# --------------------------------------------------------------------------- stats

def pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    d = (x.norm() * y.norm()).clamp_min(1e-30)
    return float((x @ y) / d)


@torch.no_grad()
def collect(backend, items, device):
    """Full logits at each item's decision position, plus the two contrast logits."""
    V = None
    logits, ld, top1 = [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        ids = torch.tensor([it["ids"]], dtype=torch.long, device=device)   # batch size 1
        row = backend.logits(ids)[0, it["p"]].float().cpu()
        if V is None:
            V = row.numel()
        logits.append(row)
        ld.append(float(row[it["tok_pos"]] - row[it["tok_neg"]]))
        top1.append(int(row.argmax()))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(items)}  ({(time.time() - t0) / (i + 1):.2f}s/item)",
                  flush=True)
    return torch.stack(logits), ld, top1, time.time() - t0


def compare(hf_logits, hf_ld, hf_top1, tl_logits, tl_ld, tl_top1):
    n = len(hf_ld)
    agree = sum(1 for a, b in zip(hf_top1, tl_top1) if a == b)
    kl = torch.nn.functional.kl_div(
        tl_logits.double().log_softmax(-1), hf_logits.double().log_softmax(-1),
        log_target=True, reduction="none").sum(-1)
    ld_err = [abs(a - b) for a, b in zip(hf_ld, tl_ld)]
    dmax = (hf_logits - tl_logits).abs().max(dim=-1).values

    res = {
        "n": n,
        "F1_top1_agree": agree, "F1_top1_rate": agree / n,
        "F2_mean_kl": float(kl.mean()), "F2_max_kl": float(kl.max()),
        "F3_max_abs_ld_err": max(ld_err), "F3_mean_abs_ld_err": sum(ld_err) / n,
        "F4_pearson_r": pearson(hf_ld, tl_ld),
        "max_abs_logit_err": float(dmax.max()), "mean_max_abs_logit_err": float(dmax.mean()),
        "ld_hf_mean": sum(hf_ld) / n, "ld_tl_mean": sum(tl_ld) / n,
    }
    res["F1_pass"] = res["F1_top1_rate"] >= THRESH["F1_top1_rate"]
    res["F2_pass"] = res["F2_mean_kl"] < THRESH["F2_mean_kl"]
    res["F3_pass"] = res["F3_max_abs_ld_err"] < THRESH["F3_max_abs_ld_err"]
    res["F4_pass"] = res["F4_pearson_r"] > THRESH["F4_pearson_r"]
    res["pass"] = all(res[f"F{i}_pass"] for i in (1, 2, 3, 4))
    return res


# --------------------------------------------------------------------------- bisect

@torch.no_grad()
def bisect(hf, tl, item, device, tol=1e-3):
    """First layer at which the two residual streams diverge (diagnostic)."""
    ids = torch.tensor([item["ids"]], dtype=torch.long, device=device)
    layers = list(range(tl.n_layers))
    a = hf.read_resid(ids, layers, pos=item["p"])
    b = tl.read_resid(ids, layers, pos=item["p"])
    rows = []
    for L in layers:
        d = float((a[L] - b[L]).abs().max())
        rel = d / max(float(a[L].abs().max()), 1e-12)
        rows.append({"layer": L, "max_abs": d, "rel": rel})
    firsts = [r for r in rows if r["rel"] > tol]
    return {"per_layer": rows,
            "first_divergent_layer": firsts[0]["layer"] if firsts else None,
            "tol": tol}


# --------------------------------------------------------------------------- main

def run_dtype(model_id, rev, dtype, items, device, tok_id, chat_src, do_bisect):
    print(f"\n{'=' * 78}\n[{dtype}] HF reference pass\n{'=' * 78}")
    hf = B.get_backend("hf", model_id, revision=rev, dtype=dtype, device=device,
                       tokenizer_id=tok_id).load()
    hf_logits, hf_ld, hf_top1, hf_s = collect(hf, items, device)
    hf_loader = getattr(hf, "_loader", None)
    print(f"  HF loader: {hf_loader}   {hf_s:.0f}s")

    bis = None
    if do_bisect:
        tl_tmp = B.get_backend("tl", model_id, revision=rev, dtype=dtype, device=device,
                               tokenizer_id=tok_id).load()
        bis = bisect(hf, tl_tmp, items[0], device)
        tl_tmp.unload()
        del tl_tmp

    hf.unload()
    del hf
    B.free_cuda()

    print(f"\n{'=' * 78}\n[{dtype}] TransformerLens pass\n{'=' * 78}")
    tl = B.get_backend("tl", model_id, revision=rev, dtype=dtype, device=device,
                       tokenizer_id=tok_id).load()
    tl_logits, tl_ld, tl_top1, tl_s = collect(tl, items, device)
    n_layers, d_model = tl.n_layers, tl.d_model
    tl.unload()
    del tl
    B.free_cuda()

    res = compare(hf_logits, hf_ld, hf_top1, tl_logits, tl_ld, tl_top1)
    res.update({"dtype": dtype, "hf_seconds": round(hf_s, 1), "tl_seconds": round(tl_s, 1),
                "hf_loader": hf_loader, "n_layers": n_layers, "d_model": d_model})
    if bis:
        res["bisect"] = bis
    return res


def report(tag, r):
    def mark(k):
        return "PASS" if r[f"{k}_pass"] else "**FAIL**"
    print(f"\n{'-' * 78}")
    print(f"{tag}  n={r['n']}")
    print(f"  F1 top-1 agreement      {r['F1_top1_agree']}/{r['n']} "
          f"({r['F1_top1_rate'] * 100:.2f}%)   need 100%          {mark('F1')}")
    print(f"  F2 mean KL(TL||HF)      {r['F2_mean_kl']:.3e}  (max {r['F2_max_kl']:.3e})"
          f"   need < 1e-4      {mark('F2')}")
    print(f"  F3 max |Δ logit-diff|   {r['F3_max_abs_ld_err']:.3e} "
          f"(mean {r['F3_mean_abs_ld_err']:.3e})  need < 0.01   {mark('F3')}")
    print(f"  F4 Pearson r(ld)        {r['F4_pearson_r']:.9f}   need > 0.999        "
          f"{mark('F4')}")
    print(f"  (aux) max |HF-TL| over the full vocab: {r['max_abs_logit_err']:.3e}")
    print(f"  VERDICT: {'PASS' if r['pass'] else 'FAIL'}")
    print("-" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--pairs", default="build/interp_pairs.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--chat-template-source", default=B.GEMMA3_CHAT_TEMPLATE_SOURCE)
    ap.add_argument("--out-dir", default="results/interp")
    ap.add_argument("--bisect", action="store_true",
                    help="also run the per-layer residual bisection diagnostic")
    args = ap.parse_args()

    rev = args.revision or B.cached_revision(args.model)
    tag = args.model.split("/")[-1]
    torch.manual_seed(args.seed)

    print("=" * 78)
    print(f"Fidelity gate (GATE 3)   {args.model}")
    print(f"revision: {rev}   host: {platform.node()}   n={args.n}   batch size 1")
    print("=" * 78)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    chat_template = None
    if tok.chat_template is None:
        chat_template = B.gemma_chat_template(args.chat_template_source)
        print(f"\n[chat template] {args.model} ships none; borrowed from "
              f"{args.chat_template_source} (byte-identical across all gemma-3 -it sizes)")

    items, dropped, n_pool = build_prompts(args.pairs, tok, chat_template, args.n,
                                           args.seed)
    print(f"\n[prompts] {len(items)} built from a pool of {n_pool}; dropped {dropped}")
    if len(items) < args.n:
        print(f"!! only {len(items)} prompts available, wanted {args.n}")
    from collections import Counter
    print(f"[prompts] slots: {dict(Counter(i['slot'] for i in items))}")
    print(f"[prompts] conditions: {dict(Counter(i['condition'] for i in items))}")
    print(f"[prompts] divergence buckets: "
          f"{dict(Counter(min(i['divergence_offset'], 2) for i in items))}")
    print(f"[prompts] token lengths: min {min(i['n_tokens'] for i in items)} "
          f"max {max(i['n_tokens'] for i in items)}")
    print(f"[prompts] example decoded tail: "
          f"{tok.decode(items[0]['ids'][items[0]['p'] - 6:items[0]['p'] + 1])!r}")

    results = {}
    for dt in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
        r = run_dtype(args.model, rev, dt, items, args.device, args.model,
                      args.chat_template_source, args.bisect and dt == "float32")
        results[dt] = r
        report(f"[{dt}] {args.model}", r)

    gate = results.get("float32", {}).get("pass", False)
    print("\n" + "=" * 78)
    print(f"GATE 3 [{args.model}] fp32: {'PASS' if gate else 'FAIL'}"
          + (f"   |   bf16: {'PASS' if results.get('bfloat16', {}).get('pass') else 'FAIL'}"
             if "bfloat16" in results else ""))
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "fidelity_report.json")
    blob = {}
    if os.path.isfile(path):
        with open(path) as fh:
            blob = json.load(fh)
    blob[args.model] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "revision": rev,
        "n_prompts": len(items), "n_requested": args.n, "seed": args.seed,
        "batch_size": 1,
        "pairs_source": args.pairs,
        "prompt_id": "P1_minimal (configs/prompts.yaml), chat-formatted",
        "chat_template_source": (args.chat_template_source if chat_template
                                 else args.model),
        "dropped": dropped, "pool_size": n_pool,
        "thresholds": THRESH,
        "gate_pass_fp32": gate,
        "results": results,
        "items": [{k: v for k, v in i.items() if k != "ids"} for i in items],
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(f"[written] {path}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
