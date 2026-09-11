#!/usr/bin/env python
"""Prompt construction, divergence location, pair screening.

Builds the minimal-pair set the probe/lens/patching all read, for the reduced slot scope
(**gender, number, rationality**; honorificity and clusivity are out of scope for this study).

Two requirements, both enforced here:

* **The divergence token is computed empirically per pair, per tokenizer — never assumed.**
  `வந்தான்` and `வந்தார்கள்` share the codepoint prefix `வந்தா`; a BPE merge can push the
  divergence to the 2nd or 3rd token.  A pair where one form is a strict token-prefix of the
  other is not expressible as a single next-token choice and is DROPPED.

* **Token-length matching is a correctness condition for patching, not a nicety.**  Patching
  restores the residual stream at a position *index*; if the + and - prompts differ in
  length before the decision point, index p is a different linguistic site in the two runs.
  Violators are dropped and the drop rate is reported per slot.

⚠ **Gender is promoted into the interp arm here.**  The generator's interpretability subset
excludes it by design (it is the behavioural calibration slot), so `interp_pairs.json`
contains ZERO gender pairs.  This script pairs gender directly from `items_tok.jsonl` using
the same rules the generator uses, so gender is screened identically to the other two slots.
Both inputs are build intermediates that include held-out items, so they are not released.

    python scripts/interp/03_pairs.py --out results/interp/pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402

SLOTS = ("number", "rationality", "gender")
TAMIL = ("஀", "௿")


def has_tamil(s: str) -> bool:
    return any(TAMIL[0] <= c <= TAMIL[1] for c in s)


def build(items_path, tok, chat_template, slots, conditions):
    rows = [json.loads(l) for l in open(items_path)]
    by_set = defaultdict(list)
    for r in rows:
        if r["slot"] in slots and r["condition"] in conditions:
            by_set[(r["set_id"], r["condition"])].append(r)

    pairs, drops = [], Counter()
    for (set_id, cond), members in sorted(by_set.items()):
        if cond == "C1":
            # A C1 pair is two members of one contrast set carrying different gold values.
            members = sorted(members, key=lambda m: m["gold_value"])
            if len(members) < 2 or members[0]["gold_value"] == members[1]["gold_value"]:
                drops["c1_not_a_pair"] += 1
                continue
            pos, neg = members[0], members[1]
            ta_pos = (pos["gold_targets"] or [{}])[0].get("tamil")
            ta_neg = (neg["gold_targets"] or [{}])[0].get("tamil")
            src_pos, src_neg = pos["source"], neg["source"]
            val_pos, val_neg = pos["gold_value"], neg["gold_value"]
            gold, base = "pos", pos
        else:
            # C3 has no gold by design, so the pair is the item's own two
            # contrast realisations and the label is which form the model actually emits.
            base = members[0]
            ct = base.get("contrast_targets") or []
            gt = base.get("gold_targets") or []
            forms = ([gt[0]] if gt else []) + list(ct)
            if len(forms) < 2:
                drops["c3_lt2_contrasts"] += 1
                continue
            ta_pos, ta_neg = forms[0].get("tamil"), forms[1].get("tamil")
            src_pos = src_neg = base["source"]
            val_pos = forms[0].get("value") or base.get("gold_value") or "A"
            val_neg = forms[1].get("value") or "B"
            gold, pos, neg = None, base, base

        if not (ta_pos and ta_neg and src_pos and src_neg) or ta_pos == ta_neg:
            drops["missing_or_identical_forms"] += 1
            continue

        # --- prompts -----------------------------------------------------
        ctx_pos = ((base.get("source_context") or "") + " " + src_pos).strip() \
            if base.get("source_context") else src_pos
        ctx_neg = ((neg.get("source_context") or "") + " " + src_neg).strip() \
            if neg.get("source_context") else src_neg

        t_pos = B.render_prompt(tok, ctx_pos, chat_template)
        t_neg = B.render_prompt(tok, ctx_neg, chat_template)
        p_pos = tok(t_pos, add_special_tokens=False)["input_ids"]
        p_neg = tok(t_neg, add_special_tokens=False)["input_ids"]

        if sum(1 for t in p_pos if t == tok.bos_token_id) != 1 or \
           sum(1 for t in p_neg if t == tok.bos_token_id) != 1:
            drops["double_bos"] += 1
            continue

        # ⚠ patching correctness: identical prompt length in BOTH runs.
        if len(p_pos) != len(p_neg):
            drops["prompt_length_mismatch"] += 1
            continue

        ids_a = tok(ta_pos, add_special_tokens=False)["input_ids"]
        ids_b = tok(ta_neg, add_special_tokens=False)["input_ids"]
        k, off = B.divergence_index(p_pos, ids_a, ids_b)
        if k is None:
            drops["prefix_containment"] += 1
            continue
        if tok.decode((p_pos + ids_a)[:k]) != tok.decode((p_pos + ids_b)[:k]):
            drops["bpe_retokenization"] += 1
            continue
        if not has_tamil(ta_pos) or not has_tamil(ta_neg):
            drops["no_tamil"] += 1
            continue

        p = k - 1
        pairs.append({
            "pair_id": f"IP-{base['slot'][:3].upper()}-{set_id}-{cond}",
            "set_id": set_id, "template_id": base["template_id"],
            "slot": base["slot"], "condition": cond, "gold": gold,
            "value_pos": val_pos, "value_neg": val_neg,
            "src_en_pos": ctx_pos, "src_en_neg": ctx_neg,
            "tamil_pos": ta_pos, "tamil_neg": ta_neg,
            # Full sequences: prompt + the full Tamil form.  Because the model is causal,
            # logits at p are unaffected by tokens after p, so one forward per member gives
            # both a Tamil-bearing context and the exact decision-position statistic.
            "ids_pos": p_pos + ids_a, "ids_neg": p_neg + ids_b,
            "n_prompt_tokens": len(p_pos),
            "p": p, "k": k, "divergence_offset": off,
            "tok_pos": (p_pos + ids_a)[k], "tok_neg": (p_pos + ids_b)[k],
            "tok_pos_str": tok.decode([(p_pos + ids_a)[k]]),
            "tok_neg_str": tok.decode([(p_pos + ids_b)[k]]),
            "fertility_pos": len(ids_a), "fertility_neg": len(ids_b),
        })
    return pairs, drops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="build/items_tok.jsonl")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--chat-template-source", default=B.GEMMA3_CHAT_TEMPLATE_SOURCE)
    ap.add_argument("--slots", default=",".join(SLOTS))
    ap.add_argument("--conditions", default="C1,C3")
    ap.add_argument("--max-per-cell", type=int, default=200)
    ap.add_argument("--out", default="results/interp/pairs.json")
    ap.add_argument("--tsv", default="results/interp/tokenization_appendix.tsv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    chat_template = None if tok.chat_template else B.gemma_chat_template(
        args.chat_template_source)

    slots = tuple(s.strip() for s in args.slots.split(","))
    conds = tuple(c.strip() for c in args.conditions.split(","))
    pairs, drops = build(args.items, tok, chat_template, slots, conds)

    # Prefer divergence bucket 0 when trimming a cell.
    cells = defaultdict(list)
    for p in pairs:
        cells[(p["slot"], p["condition"])].append(p)
    kept = []
    per_cell = {}
    for key, ps in sorted(cells.items()):
        ps.sort(key=lambda r: (min(r["divergence_offset"], 2), r["pair_id"]))
        sel = ps[:args.max_per_cell]
        kept.extend(sel)
        per_cell["/".join(key)] = {
            "available": len(ps), "selected": len(sel),
            "buckets": dict(Counter(min(r["divergence_offset"], 2) for r in sel)),
            "n_templates": len({r["template_id"] for r in sel}),
            "mean_fertility_pos": round(sum(r["fertility_pos"] for r in sel) / len(sel), 2),
            "mean_fertility_neg": round(sum(r["fertility_neg"] for r in sel) / len(sel), 2),
        }

    print(f"built {len(kept)} pairs; drops = {dict(drops)}")
    for k, v in per_cell.items():
        print(f"  {k:<22} {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"generated": datetime.now(timezone.utc).isoformat(),
                   "model": args.model, "items": args.items,
                   "chat_template_source": args.chat_template_source,
                   "drops": dict(drops), "per_cell": per_cell,
                   "pairs": kept}, fh, ensure_ascii=False)
    print(f"[written] {args.out}")

    with open(args.tsv, "w") as fh:
        fh.write("item_id\tslot\tcondition\tform_pos\tform_neg\ttokens_pos\ttokens_neg\t"
                 "divergence_offset\ttok_pos_str\ttok_neg_str\n")
        for r in kept:
            fh.write(f"{r['pair_id']}\t{r['slot']}\t{r['condition']}\t{r['tamil_pos']}\t"
                     f"{r['tamil_neg']}\t{r['fertility_pos']}\t{r['fertility_neg']}\t"
                     f"{r['divergence_offset']}\t{r['tok_pos_str']}\t{r['tok_neg_str']}\n")
    print(f"[written] {args.tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
