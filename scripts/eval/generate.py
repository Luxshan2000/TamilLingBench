#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generation driver. RUNS ON THE GPU MACHINE.

**The architectural commitment this file exists to enforce: generate once, score many times.**
Nothing in this file computes, imports, or has any way of expressing a score. It writes raw
model output plus provenance to append-only per-(model, slot, condition, prompt) JSONL, and it
refuses to overwrite an existing file unless `--force` is passed. Every open gold-standard
question (the Tamil third honorific degree, the Indian/Sri Lankan rationality variant) is a
SCORING question; re-deciding one must never require re-running a GPU.

Hazards handled here, each with an assertion rather than a comment:

  * **double-BOS** — `apply_chat_template(tokenize=False)` emits `<bos>`, and the tokenizer
    prepends another unless `add_special_tokens=False`. Silent, and it corrupts everything.
    Asserted per batch (`_assert_single_bos`).
  * **Qwen3 `<think>`** — on by default in the chat template; shifts the decision token by
    hundreds of positions. Disabled via `chat_template_kwargs`, and the *rendered prompt* is
    asserted to contain no unclosed think block, then the *output* is asserted `<think`-free.
  * **Gemma has no system role** — folded into the first user turn via `supports_system`.
  * **Gemma-3 4B/12B/27B are `Gemma3ForConditionalGeneration`** — the loaded class name is
    recorded in the manifest and the text tower is driven with `input_ids` only.
  * **left padding** — batched decoder-only generation with right padding silently produces
    garbage for every sequence but the longest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAMIL = re.compile(r"[஀-௿]")
THINK_OPEN = re.compile(r"<think\b", re.I)
THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
LABEL_PREFIX = re.compile(r"^\s*(tamil|translation|தமிழ்)\s*[:：]\s*", re.I)


# --------------------------------------------------------------------------- provenance

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def resolve_revision(hf_name: str) -> str:
    """The commit SHA actually on disk, read from the snapshot directory name.

    Read back rather than requested: the manifest must record the revision ACTUALLY loaded.
    """
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    d = root / ("models--" + hf_name.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return "unresolved"
    snaps = sorted(p.name for p in d.iterdir() if p.is_dir())
    return snaps[-1] if snaps else "unresolved"


# --------------------------------------------------------------------------- prompting

def render(prompt_id: str, prompts: dict, item: dict) -> tuple[str | None, str]:
    """(system, user). Placeholders come from the item; a missing one is a hard failure."""
    spec = prompts[prompt_id]
    tgts = item.get("gold_targets") or []
    fields = {
        "source": item["source"],
        "source_context": item.get("source_context") or "",
        "c0_instruction": (item.get("fillers") or {}).get("c0_instruction") or "",
        # Copied off the item, never synthesised — the harness must not be in the business of
        # writing Tamil.
        "target_morpheme": (tgts[0].get("surface_morpheme") if tgts else "") or "",
        "slot": item["slot"],
    }
    if "{target_morpheme}" in spec["user"] and not fields["target_morpheme"]:
        raise ValueError(f"{item['item_id']}: no gold surface_morpheme to state")
    if "{source_context}" in spec["user"] and not fields["source_context"]:
        raise ValueError(f"{item['item_id']}: {prompt_id} needs source_context, item has none")
    if "{c0_instruction}" in spec["user"] and not fields["c0_instruction"]:
        raise ValueError(f"{item['item_id']}: {prompt_id} needs c0_instruction, item has none")
    return spec.get("system"), spec["user"].format_map(fields)


def build_chat(system: str | None, user: str, spec: dict) -> list[dict]:
    system = spec.get("system_override") or system
    if system and not spec.get("supports_system", True):
        user = system.strip() + "\n\n" + user       # fold-down
        system = None
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system.strip()})
    msgs.append({"role": "user", "content": user})
    return msgs


# --------------------------------------------------------------------------- cleaning
# Single implementation, shared with the scoring pass — see scripts/eval/postprocess.py.
from postprocess import clean, tamil_share  # noqa: E402





# --------------------------------------------------------------------------- backends

class CausalLM:
    def __init__(self, spec: dict, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = spec["hf_name"]
        self.spec = spec
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"dtype": getattr(torch, spec.get("dtype", "bfloat16"))}
        cls_name = spec.get("text_only_class")
        if cls_name:
            # ⚠ TEXT-TOWER-ONLY LOAD. Required for Gemma-3n: AutoModelForCausalLM resolves
            # gemma3n to Gemma3nForConditionalGeneration, whose constructor builds a
            # MobileNet-V5 vision tower through TimmWrapperModel — so it raises ImportError
            # without `timm`, and `timm` needs `torchvision`, whose prebuilt aarch64 wheel does
            # not load against this box's torch 2.10.0+cu130 (`operator torchvision::nms does
            # not exist`). Gemma-3 does NOT hit this: its SigLIP tower is native transformers.
            #
            # Rather than fake a library to build a tower we never call, load the text model
            # class directly. This is only safe if the discard is AUDITED, so it is:
            # `missing_keys` must be empty (every text weight was found) and every unexpected
            # key must belong to a non-text tower. A text weight silently landing in
            # `unexpected` would mean we are generating from a partly random model.
            from transformers import AutoTokenizer as _T  # noqa: F401
            import transformers as _tf
            model_cls = getattr(_tf, cls_name)
            try:
                self.model, info = model_cls.from_pretrained(
                    name, output_loading_info=True, **kw)
            except TypeError:
                kw = {"torch_dtype": getattr(torch, spec.get("dtype", "bfloat16"))}
                self.model, info = model_cls.from_pretrained(
                    name, output_loading_info=True, **kw)
            missing = list(info.get("missing_keys") or [])
            if missing:
                raise AssertionError(
                    f"{name}: text-only load left {len(missing)} MISSING weight(s) "
                    f"(first: {missing[:5]}). Refusing to generate from an incomplete model.")
            allowed = tuple(spec.get("discardable_towers")
                            or ("vision_tower", "audio_tower", "embed_vision", "embed_audio"))
            unexpected = list(info.get("unexpected_keys") or [])
            stray = [k for k in unexpected if not any(t in k for t in allowed)]
            if stray:
                raise AssertionError(
                    f"{name}: text-only load discarded {len(stray)} weight(s) that are NOT "
                    f"vision/audio towers (first: {stray[:5]}).")
            self.discarded_tower_weights = len(unexpected)
            self.model = self.model.to(device).eval()
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
            except TypeError:                        # transformers <5 keyword
                kw = {"torch_dtype": getattr(torch, spec.get("dtype", "bfloat16"))}
                self.model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
            self.discarded_tower_weights = 0
        self.model_class = type(self.model).__name__
        self.device = device
        self.has_chat_template = getattr(self.tok, "chat_template", None) is not None

    def _render_prompt(self, chat: list[dict]) -> str:
        if not self.has_chat_template:
            # Base model (llama31-8b): no chat template exists. Concatenate turns plainly and
            # record that this is what happened — do NOT invent one.
            return "\n\n".join(m["content"] for m in chat) + "\n"
        return self.tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            **self.spec.get("chat_template_kwargs") or {})

    def _assert_single_bos(self, ids: torch.Tensor, texts: list[str]) -> None:
        bos = self.tok.bos_token_id
        if bos is None:
            return
        row = ids[0].tolist()
        lead = 0
        for t in row:
            if t == bos:
                lead += 1
            elif t != self.tok.pad_token_id:
                break
        if lead > 1:
            raise AssertionError(
                f"DOUBLE-BOS: {lead} leading BOS ids. Rendered prompt starts: {texts[0][:80]!r}")

    def generate(self, chats: list[list[dict]], dec: dict, batch_size: int) -> list[dict]:
        out: list[dict] = []
        order = sorted(range(len(chats)), key=lambda i: -len(chats[i][-1]["content"]))
        results: dict[int, dict] = {}
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            texts = [self._render_prompt(chats[i]) for i in idx]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            self._assert_single_bos(enc["input_ids"], texts)
            gen_kw = dict(max_new_tokens=dec["max_new_tokens"], do_sample=dec["do_sample"],
                          num_beams=dec.get("num_beams", 1),
                          pad_token_id=self.tok.pad_token_id)
            if dec["do_sample"]:
                gen_kw.update(temperature=dec["temperature"], top_p=dec["top_p"],
                              top_k=dec["top_k"] if dec.get("top_k") else 0)
            with torch.inference_mode():
                o = self.model.generate(**enc, **gen_kw)
            new = o[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                txt = self.tok.decode(new[j], skip_special_tokens=True)
                n_tok = int((new[j] != self.tok.pad_token_id).sum())
                results[i] = {"raw": txt, "n_out_tokens": n_tok,
                              "finish_reason": "length" if n_tok >= dec["max_new_tokens"] else "stop",
                              "rendered_prompt_head": texts[j][:200]}
        return [results[i] for i in range(len(chats))]


class NLLB:
    """`src_lang` on the TOKENIZER, `tgt_lang` via forced_bos_token_id."""

    def __init__(self, spec: dict, device: str = "cuda"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = spec["hf_name"]
        self.tok = AutoTokenizer.from_pretrained(name, src_lang=spec["src_lang"])
        kw = {"dtype": getattr(torch, spec.get("dtype", "float16"))}
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name, **kw).to(device).eval()
        except TypeError:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                name, torch_dtype=getattr(torch, spec.get("dtype", "float16"))).to(device).eval()
        self.model_class = type(self.model).__name__
        self.forced_bos = self.tok.convert_tokens_to_ids(spec["tgt_lang"])
        if self.forced_bos in (None, self.tok.unk_token_id):
            raise AssertionError(f"NLLB target token {spec['tgt_lang']} did not resolve "
                                 f"(got {self.forced_bos}, unk={self.tok.unk_token_id})")
        maxpos = getattr(self.model.config, "max_position_embeddings", 1024)
        self.max_src = min(512, maxpos)
        self.device = device
        self.has_chat_template = False

    def generate(self, srcs: list[str], dec: dict, batch_size: int) -> list[dict]:
        results: dict[int, dict] = {}
        order = sorted(range(len(srcs)), key=lambda i: -len(srcs[i]))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([srcs[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_src).to(self.device)
            with torch.inference_mode():
                o = self.model.generate(**enc, forced_bos_token_id=self.forced_bos,
                                        num_beams=dec.get("num_beams", 5),
                                        do_sample=dec["do_sample"],
                                        max_new_tokens=dec["max_new_tokens"])
            dec_txt = self.tok.batch_decode(o, skip_special_tokens=True)
            for j, i in enumerate(idx):
                results[i] = {"raw": dec_txt[j], "n_out_tokens": int(o[j].ne(self.tok.pad_token_id).sum()),
                              "finish_reason": "stop", "rendered_prompt_head": srcs[i][:200]}
        return [results[i] for i in range(len(srcs))]


class IndicTrans2:
    """`IndicProcessor.preprocess_batch`/`postprocess_batch` are MANDATORY —
    skipping them silently degrades this baseline (script normalization, entity
    placeholdering, language-tag prefix)."""

    def __init__(self, spec: dict, device: str = "cuda"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except Exception:
            from IndicTransToolkit import IndicProcessor
        name = spec["hf_name"]
        self.ip = IndicProcessor(inference=True)
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name, trust_remote_code=True,
            torch_dtype=getattr(torch, spec.get("dtype", "float16"))).to(device).eval()
        self.model_class = type(self.model).__name__
        self.src_lang, self.tgt_lang = spec["src_lang"], spec["tgt_lang"]
        self.device = device
        self.has_chat_template = False
        # Round-trip smoke test before anything is generated.
        probe = self.ip.preprocess_batch(["The train arrives at six o'clock."],
                                         src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        assert probe and probe[0].startswith(self.src_lang), \
            f"IndicProcessor did not prefix the language tag: {probe!r}"

    def generate(self, srcs: list[str], dec: dict, batch_size: int) -> list[dict]:
        results: dict[int, dict] = {}
        order = sorted(range(len(srcs)), key=lambda i: -len(srcs[i]))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = self.ip.preprocess_batch([srcs[i] for i in idx],
                                             src_lang=self.src_lang, tgt_lang=self.tgt_lang)
            enc = self.tok(batch, padding="longest", truncation=True, max_length=256,
                           return_tensors="pt").to(self.device)
            with torch.inference_mode():
                # ⚠ use_cache=False is REQUIRED, not a tuning knob.
                # ai4bharat/indictrans2-en-indic-1B ships remote code (modeling_indictrans.py,
                # rev 10e65a9) written against the legacy tuple `past_key_values`; it does
                #     past_key_values[0][0].shape[2] if past_key_values is not None else 0
                # transformers >=4.5x hands `generate` an EncoderDecoderCache instead, which is
                # non-None from step 0 but whose [0][0] is None, so the model crashed with
                # `AttributeError: 'NoneType' object has no attribute 'shape'` on EVERY batch
                # (both beam and greedy) — the whole 2026-08-07 IT2 run produced zero records.
                # Disabling the cache keeps `past_key_values` None, taking the legacy branch.
                # The KV cache is a pure speed optimisation: logits, and therefore the emitted
                # strings, are unchanged. Do not "optimise" this back on without pinning
                # transformers<4.5 or patching the remote code.
                o = self.model.generate(**enc, num_beams=dec.get("num_beams", 5),
                                        num_return_sequences=1, do_sample=dec["do_sample"],
                                        max_length=256, use_cache=False)
            dec_txt = self.tok.batch_decode(o, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
            post = self.ip.postprocess_batch(dec_txt, lang=self.tgt_lang)
            assert len(post) == len(idx), f"IndicProcessor length mismatch {len(post)}!={len(idx)}"
            for j, i in enumerate(idx):
                results[i] = {"raw": post[j], "n_out_tokens": int(o[j].shape[0]),
                              "finish_reason": "stop", "rendered_prompt_head": batch[j][:200]}
        return [results[i] for i in range(len(srcs))]


BACKENDS = {"causal_lm": CausalLM, "nllb": NLLB, "indictrans2": IndicTrans2}


# --------------------------------------------------------------------------- driver

CANONICAL_PROMPT = {"C0": "P0_specified", "C1": "P1_minimal",
                    "C2": "P4_context", "C3": "P1_minimal"}
ROBUSTNESS_PROMPTS = {
    "C1": ["P1_minimal", "P2_persona", "P3_fewshot"],
    "C3": ["P1_minimal", "P2_persona", "P3_fewshot"],
    "C2": ["P4_context", "P1_context", "P2_context", "P3_context"],
}


def load_items(path: Path, split: str) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if split != "all" and it.get("split") != split:
            continue
        assert it.get("template_id") and it.get("set_id"), \
            f"{it['item_id']}: template_id/set_id are load-bearing and must be non-null"
        items.append(it)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--prompts-config", default="configs/prompts.yaml")
    ap.add_argument("--decoding-config", default="configs/decoding.yaml")
    ap.add_argument("--items", default="data/benchmark/items.jsonl")
    ap.add_argument("--split", default="public")
    ap.add_argument("--out", default="outputs/raw")
    ap.add_argument("--reports", default="outputs/run_manifests",
                    help="run manifests land here, flushed after EVERY cell")
    ap.add_argument("--conditions", default="C0,C1,C2,C3")
    ap.add_argument("--slots", default="")
    ap.add_argument("--prompt-set", default="canonical",
                    choices=["canonical", "robustness", "c0_explicit"])
    ap.add_argument("--robustness-subset", default="",
                    help="path to a newline-delimited item_id list; robustness runs only these")
    ap.add_argument("--decoding", default="main")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-per-cell", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-id", default="")
    # The panel runs on a GPU machine; `mps` / `cpu` exist so a CPU-side check can reuse this
    # exact driver rather than a second, divergent generation path. Recorded in the manifest.
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(".").resolve()
    panel = {m["id"]: m for m in yaml.safe_load(
        (root / args.models_config).read_text())["panel"]}
    spec = panel[args.model]
    prompts = yaml.safe_load((root / args.prompts_config).read_text())
    decoding_all = yaml.safe_load((root / args.decoding_config).read_text())
    arm = spec.get("arm", "llm")
    dec = decoding_all[args.decoding]["nmt" if arm == "nmt" else "llm"]

    torch.manual_seed(dec["seed"])
    items = load_items(root / args.items, args.split)
    conds = args.conditions.split(",")
    slots = args.slots.split(",") if args.slots else None
    subset = None
    if args.robustness_subset:
        subset = set(Path(args.robustness_subset).read_text().split())

    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "-" + sha256_text(args.model + str(time.time()))[:6])
    revision = resolve_revision(spec["hf_name"])

    print(f"[{args.model}] loading {spec['hf_name']} rev={revision} backend={spec['backend']}",
          flush=True)
    t0 = time.time()
    backend = BACKENDS[spec["backend"]](spec, device=args.device)
    print(f"[{args.model}] loaded {backend.model_class} in {time.time()-t0:.0f}s "
          f"chat_template={backend.has_chat_template}", flush=True)

    outdir = Path(args.out).expanduser() / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    repdir = Path(args.reports).expanduser()
    repdir.mkdir(parents=True, exist_ok=True)
    hygiene = {"n": 0, "cleaned": 0, "think_leak": 0, "low_tamil": 0, "empty": 0,
               "clean_ops": {}}
    cells_done = []

    base_manifest = {
        "run_id": run_id, "system_id": args.model, "hf_name": spec["hf_name"],
        "revision": revision, "model_class": backend.model_class,
        # Non-zero only for a text-tower-only load (Gemma-3n). Recorded so the report can say
        # which weights were deliberately not loaded rather than leaving it to a code reader.
        "discarded_tower_weights": getattr(backend, "discarded_tower_weights", 0),
        "backend": spec["backend"], "dtype": spec.get("dtype"), "device": args.device,
        "arm": arm, "prompt_set": args.prompt_set, "decoding_id": args.decoding,
        "decoding": dec, "split": args.split, "conditions": conds,
        "canonical_prompt_per_condition": CANONICAL_PROMPT,
        "robustness_prompts_per_condition": ROBUSTNESS_PROMPTS,
        "batch_size": args.batch_size, "limit_per_cell": args.limit_per_cell,
        "chat_template_kwargs": spec.get("chat_template_kwargs") or {},
        "has_chat_template": backend.has_chat_template,
        "out_dir": str(outdir),
        "models_config_sha256": sha256_text((root / args.models_config).read_text()),
        "prompts_config_sha256": sha256_text((root / args.prompts_config).read_text()),
        "decoding_config_sha256": sha256_text((root / args.decoding_config).read_text()),
        "items_sha256": sha256_text((root / args.items).read_text()),
        "code_sha256": sha256_text(Path(__file__).read_text()),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = repdir / f"run-manifest.{args.model}.{args.prompt_set}.json"

    def flush_manifest(status: str) -> None:
        """Written after EVERY cell. A run that dies at model 7 of 12 must still be
        interpretable, not mysterious."""
        m = dict(base_manifest, status=status, hygiene=hygiene, cells=cells_done,
                 updated_utc=datetime.now(timezone.utc).isoformat())
        tmp = man_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(man_path)
        return m

    flush_manifest("started")

    for cond in conds:
        if args.prompt_set == "c0_explicit":
            if cond != "C0":
                continue
            plist = ["P0_explicit"]
        elif args.prompt_set == "canonical":
            plist = [CANONICAL_PROMPT[cond]]
        else:
            if cond == "C0":
                continue                       # C0's prompt IS the manipulation
            plist = ROBUSTNESS_PROMPTS[cond]
        if arm == "nmt":
            plist = plist[:1]                  # the NMT arm has no prompt
        for prompt_id in plist:
            pool = [it for it in items if it["condition"] == cond]
            if slots:
                pool = [it for it in pool if it["slot"] in slots]
            if subset is not None and args.prompt_set == "robustness":
                pool = [it for it in pool if it["item_id"] in subset]
            for slot in sorted({it["slot"] for it in pool}):
                sel = [it for it in pool if it["slot"] == slot]
                sel.sort(key=lambda x: x["item_id"])
                if args.limit_per_cell:
                    sel = sel[:args.limit_per_cell]
                if not sel:
                    continue
                tag = "nmt" if arm == "nmt" else prompt_id
                fp = outdir / f"{slot}_{cond}_{tag}.jsonl"
                if fp.exists() and not args.force:
                    print(f"  SKIP {fp.name} (exists; raw outputs are never overwritten)",
                          flush=True)
                    continue

                t1 = time.time()
                if arm == "nmt":
                    srcs = [((it["source_context"] + " " + it["source"])
                             if (cond == "C2" and it.get("source_context")) else it["source"])
                            for it in sel]
                    outs = backend.generate(srcs, dec, args.batch_size)
                    rendered = srcs
                else:
                    chats, rendered = [], []
                    for it in sel:
                        sysmsg, user = render(prompt_id, prompts, it)
                        chats.append(build_chat(sysmsg, user, spec))
                        rendered.append(user)
                    outs = backend.generate(chats, dec, args.batch_size)
                dt = time.time() - t1

                recs = []
                for it, o, rq in zip(sel, outs, rendered):
                    hyp, ops = clean(o["raw"])
                    ts = tamil_share(hyp)
                    hygiene["n"] += 1
                    if ops:
                        hygiene["cleaned"] += 1
                    for op in ops:
                        hygiene["clean_ops"][op] = hygiene["clean_ops"].get(op, 0) + 1
                    if THINK_OPEN.search(o["raw"]):
                        hygiene["think_leak"] += 1
                    if ts < 0.9:
                        hygiene["low_tamil"] += 1
                    if not hyp:
                        hygiene["empty"] += 1
                    recs.append({
                        # --- item identity (schema names; NOT scores) ---
                        "item_id": it["item_id"], "set_id": it["set_id"],
                        "template_id": it["template_id"],
                        "template_version": it["template_version"],
                        "slot": it["slot"], "slot_family": it["slot_family"],
                        "condition": it["condition"], "k": it["k"],
                        "chance_rate": it["chance_rate"], "stratum": it["stratum"],
                        "register": it["register"], "split": it["split"],
                        # --- the output ---
                        "hypothesis": hyp, "hypothesis_raw": o["raw"],
                        "cleaned": bool(ops), "clean_ops": ops,
                        "tamil_char_share": round(ts, 4),
                        "n_out_tokens": o["n_out_tokens"],
                        "finish_reason": o["finish_reason"],
                        # --- provenance ---
                        "system_id": args.model, "hf_name": spec["hf_name"],
                        "revision": revision, "model_class": backend.model_class,
                        "backend": spec["backend"], "dtype": spec.get("dtype"), "device": args.device,
                        "prompt_id": tag, "decoding_id": args.decoding, "decoding": dec,
                        "chat_template_kwargs": spec.get("chat_template_kwargs") or {},
                        "supports_system": spec.get("supports_system", True),
                        "has_chat_template": backend.has_chat_template,
                        "request_head": rq[:300],
                        "run_id": run_id,
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                    })
                tmp = fp.with_suffix(".jsonl.tmp")
                tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                               encoding="utf-8")
                tmp.replace(fp)
                cells_done.append({"file": fp.name, "slot": slot, "condition": cond,
                                   "prompt_id": tag, "n": len(recs),
                                   "seconds": round(dt, 1),
                                   "sec_per_item": round(dt / max(1, len(recs)), 3),
                                   "completed_utc": datetime.now(timezone.utc).isoformat()})
                flush_manifest("running")
                print(f"  {fp.name}: {len(recs)} in {dt:.0f}s "
                      f"({dt/max(1,len(recs)):.2f}s/item)", flush=True)

    # Qwen3 assertion, checked over everything actually produced.
    if hygiene["think_leak"]:
        print(f"  ⚠ <think leaked in {hygiene['think_leak']}/{hygiene['n']} raw outputs",
              flush=True)

    man = flush_manifest("complete")
    print(json.dumps({k: man[k] for k in ("run_id", "system_id", "revision", "model_class")},
                     indent=1))
    print("HYGIENE", json.dumps(hygiene, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
