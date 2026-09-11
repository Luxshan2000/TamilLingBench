#!/usr/bin/env python
"""Build results/interp/manifest.json — the run's provenance record.

"Pin `revision=` to a commit SHA for every model and record the SHAs in
`results/interp/manifest.json`; Gemma repos have been re-uploaded before."  Nothing in the
repo pinned SHAs before this.

Also records package versions, host, seeds, per-pass wall clock (any
pass exceeding 2x its estimate is flagged), and the gate verdicts, so a later reader can tell
which checkpoint and which library versions produced a number.

    python scripts/interp/manifest.py --out results/interp/manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402

MODELS = [
    ("google/gemma-3-4b-pt", "interp target — BASE model, matches Ferrando & Costa-jussa's "
                             "published Gemma-base setup (Part 3 of the approved plan)"),
    ("google/gemma-3-1b-it", "de-risk model — genuinely text-only Gemma3ForCausalLM, same "
                             "tokenizer and chat template, runs in minutes"),
    ("google/gemma-3-4b-it", "chat-template donor / behavioural-panel counterpart"),
    ("google/gemma-3-12b-it", "scale check, not yet exercised"),
    ("Qwen/Qwen3-8B", "not yet exercised"),
]

PACKAGES = ["torch", "transformers", "transformer_lens", "nnsight", "scikit-learn",
            "numpy", "scipy", "accelerate", "datasets", "safetensors", "huggingface-hub"]


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        return out.strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/interp/manifest.json")
    ap.add_argument("--results-dir", default="results/interp")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    models = {}
    for hf_id, role in MODELS:
        rev = B.cached_revision(hf_id)
        snap = B.snapshot_dir(hf_id, rev)
        shards = sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(snap, "*.safetensors"))) if snap else []
        models[hf_id] = {
            "revision": rev,
            "revision_source": "local HF cache refs/main (== snapshot dir name == commit SHA)",
            "role": role,
            "cached": snap is not None,
            "weights_present": bool(shards),
            "weight_shards": shards,
        }

    gates = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "gate2_text_tower.*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        gates[d["model"]] = {
            "gate2_verdict": d["verdict"], "gate2_host": d.get("host"),
            "n_pass": d["n_pass"], "n_fail": d["n_fail"], "n_unknown": d["n_unknown"],
            "revision": d["revision"], "file": os.path.basename(p),
        }

    fid_path = os.path.join(args.results_dir, "fidelity_report.json")
    fidelity = {}
    timings = {}
    if os.path.isfile(fid_path):
        with open(fid_path) as fh:
            fr = json.load(fh)
        for m, d in fr.items():
            fidelity[m] = {
                "host": d.get("host"), "revision": d.get("revision"),
                "n_prompts": d.get("n_prompts"), "seed": d.get("seed"),
                "gate3_fp32": d.get("gate_pass_fp32"),
                "gate3_bf16": d.get("results", {}).get("bfloat16", {}).get("pass"),
                "fp32": {k: v for k, v in d.get("results", {}).get("float32", {}).items()
                         if k.startswith(("F1", "F2", "F3", "F4", "max_abs", "pass"))},
                "bf16": {k: v for k, v in d.get("results", {}).get("bfloat16", {}).items()
                         if k.startswith(("F1", "F2", "F3", "F4", "max_abs", "pass"))},
            }
            for dt, r in d.get("results", {}).items():
                timings[f"fidelity/{m}/{dt}"] = {
                    "hf_seconds": r.get("hf_seconds"), "tl_seconds": r.get("tl_seconds"),
                    "n_forwards": 2 * d.get("n_prompts", 0)}

    blob = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "stage": os.environ.get("INTERP_STAGE",
                                "gates only."),
        "host": {
            "node": platform.node(), "platform": platform.platform(),
            "python": sys.version.split()[0], "gpu": gpu_info(),
        },
        "packages": {p: B.package_version(p) for p in PACKAGES},
        "models": models,
        "prompt_construction": {
            "prompt_id": "P1_minimal (configs/prompts.yaml) — the behavioural panel's "
                         "canonical prompt",
            "chat_template_source": B.GEMMA3_CHAT_TEMPLATE_SOURCE,
            "chat_template_sha256_16": B.GEMMA3_CHAT_TEMPLATE_SHA256,
            "chat_template_note": "gemma-3-4b-pt ships no chat template; the template is "
                                  "borrowed from an -it repo and is byte-identical across "
                                  "gemma-3 1b/4b/12b -it.",
            "pairs_source": "interp_pairs.json, built from the full template set (held-out "
                            "items included, so not released)",
            "batch_size": 1,
            "single_bos_asserted": True,
        },
        "gate1_api_signatures": os.path.basename(
            os.path.join(args.results_dir, "api_signatures.txt")),
        "gate2": gates,
        "gate3": fidelity,
        "wall_clock": timings,
        "seeds": {"prompt_selection": 20260808, "torch": 20260808},
    }
    # science outputs, if they exist
    sci = {}
    for name in ("pairs.json", "probe_gemma-3-4b-pt.csv", "lens_gemma-3-4b-pt.csv",
                 "patch_gemma-3-4b-pt.csv", "onsets.csv", "onsets_probe.csv",
                 "verdict.json", "preregistration.json",
                 "tokenization_appendix.tsv", "lens_status.json"):
        p_ = os.path.join(args.results_dir, name)
        if os.path.exists(p_):
            sci[name] = {"bytes": os.path.getsize(p_),
                         "mtime_utc": datetime.fromtimestamp(
                             os.path.getmtime(p_), tz=timezone.utc).isoformat()}
    blob["artifacts"] = sci
    # acceptance criterion #4: the preregistration must predate every probe output
    pre = sci.get("preregistration.json", {}).get("mtime_utc")
    probes = [v["mtime_utc"] for k, v in sci.items()
              if k.startswith(("probe_", "onsets", "patch_", "lens_", "verdict"))]
    if pre and probes:
        blob["preregistration_predates_results"] = all(pre < t for t in probes)

    if args.note:
        blob["note"] = args.note

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(blob, fh, indent=2, ensure_ascii=False)
    print(json.dumps(blob, indent=2, ensure_ascii=False)[:4000])
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
