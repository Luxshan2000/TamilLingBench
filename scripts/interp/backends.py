"""Backend abstraction for the interpretability work (the escape hatch if TL is unfaithful).

Everything downstream of the gates reads and writes the residual stream through this
interface, so that swapping TransformerLens for nnsight is a one-file change rather than a
rewrite.  That swap is a *planned* fallback: if TL builds
Gemma-3 unfaithfully (interleaved local/global attention, two RoPE bases, the sqrt(d_model)
embedding scale), nnsight wraps the HF module graph directly and therefore cannot have a
weight-folding bug.

The interface is deliberately tiny:

    b = get_backend("tl", hf_id, revision=..., dtype=..., device=...)
    b.load()
    ids     = b.to_tokens(text)                 # [1, T] int64, NO implicit BOS
    logits  = b.logits(ids)                     # [1, T, V] float32
    resid   = b.read_resid(ids, layers)         # {layer: [d_model]} at a position
    logits2 = b.write_resid(ids, layer, pos, v) # patched forward

Nothing here is Gemma-specific; the Gemma-3 structural assertions live in
01_load_and_verify.py.

NOTE ON VERSIONS (verified on the GPU machine 2026-08-08):
  torch 2.10.0+cu130, transformers 5.12.1, transformer_lens 3.7.0, nnsight 0.7.0.
  `transformer_lens` exposes **no** `__version__` attribute in 3.7.0 — use
  importlib.metadata.version("transformer_lens").
"""

from __future__ import annotations

import gc
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

# --------------------------------------------------------------------------- helpers

HUB = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
HUB = os.path.join(HUB, "hub") if not HUB.endswith("hub") else HUB


def package_version(name: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def cached_revision(hf_id: str) -> str | None:
    """The commit SHA of the locally cached snapshot.

    Gemma repos have been re-uploaded before, so every load must pin a
    revision and every run must record the SHA it used.  The snapshot directory name *is*
    the commit SHA, and `refs/main` records what `main` pointed at when it was fetched.
    """
    d = os.path.join(HUB, "models--" + hf_id.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as fh:
            return fh.read().strip()
    snaps = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
    return os.path.basename(snaps[-1]) if snaps else None


def snapshot_dir(hf_id: str, revision: str | None = None) -> str | None:
    d = os.path.join(HUB, "models--" + hf_id.replace("/", "--"), "snapshots")
    if revision:
        p = os.path.join(d, revision)
        return p if os.path.isdir(p) else None
    snaps = sorted(glob.glob(os.path.join(d, "*")))
    return snaps[-1] if snaps else None


DTYPES = {"float32": torch.float32, "fp32": torch.float32,
          "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float16": torch.float16, "fp16": torch.float16}


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- base

@dataclass
class Backend:
    hf_id: str
    revision: str | None = None
    dtype: torch.dtype = torch.float32
    device: str = "cuda"
    tokenizer_id: str | None = None      # borrow another repo's tokenizer/chat template
    extra: dict = field(default_factory=dict)

    model: Any = None
    tok: Any = None

    name: str = "base"

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "Backend":
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        free_cuda()

    # -- introspection -----------------------------------------------------
    @property
    def n_layers(self) -> int:
        raise NotImplementedError

    @property
    def d_model(self) -> int:
        raise NotImplementedError

    def resid_site(self, layer: int) -> str:
        """Name of the read/write site for the residual stream *leaving* block `layer`."""
        raise NotImplementedError

    # -- forward -----------------------------------------------------------
    def to_tokens(self, text: str) -> torch.Tensor:
        """Tokenize WITHOUT adding any special token.  BOS must come from the chat
        template, never from the tokenizer, or Gemma gets a double BOS."""
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def read_resid(self, ids: torch.Tensor, layers: Sequence[int], pos: int = -1):
        raise NotImplementedError

    @torch.no_grad()
    def write_resid(self, ids: torch.Tensor, layer: int, pos: int, vec: torch.Tensor):
        raise NotImplementedError


# --------------------------------------------------------------------------- HF

class HFBackend(Backend):
    """Raw `transformers` reference implementation.  This is the *ground truth* the
    fidelity gate compares against; it has no hook plumbing beyond forward hooks."""

    name = "hf"

    def load(self) -> "HFBackend":
        from transformers import AutoConfig, AutoTokenizer

        tok_id = self.tokenizer_id or self.hf_id
        self.tok = AutoTokenizer.from_pretrained(tok_id)
        self.hf_cfg = AutoConfig.from_pretrained(self.hf_id, revision=self.revision)

        model = None
        errs = []
        for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
            try:
                import transformers

                cls = getattr(transformers, loader)
                model = cls.from_pretrained(
                    self.hf_id, revision=self.revision, dtype=self.dtype,
                    device_map=None, **self.extra,
                )
                self._loader = loader
                break
            except Exception as exc:                      # try the next entry point
                errs.append(f"{loader}: {type(exc).__name__}: {exc}")
        if model is None:
            raise RuntimeError("no HF loader worked:\n  " + "\n  ".join(errs))

        self.model = model.to(self.device).eval()
        return self

    @property
    def text_config(self):
        return getattr(self.hf_cfg, "text_config", self.hf_cfg)

    @property
    def n_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.text_config.hidden_size

    def _decoder(self):
        """The text-tower decoder module, whatever wrapper it is nested under."""
        m = self.model
        for path in ("model.language_model", "model.model.language_model",
                     "language_model.model", "model.text_model", "model", "transformer"):
            cur = m
            ok = True
            for part in path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok and hasattr(cur, "layers"):
                return cur
        raise RuntimeError("could not locate the decoder stack on the HF model")

    def resid_site(self, layer: int) -> str:
        return f"decoder.layers[{layer}].output"

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=ids)
        return out.logits.float()

    @torch.no_grad()
    def read_resid(self, ids, layers, pos: int = -1):
        dec = self._decoder()
        store: dict[int, torch.Tensor] = {}
        handles = []

        def mk(L):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[L] = h[0, pos].detach().float().cpu()
            return hook

        for L in layers:
            handles.append(dec.layers[L].register_forward_hook(mk(L)))
        try:
            self.model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()
        return store

    @torch.no_grad()
    def write_resid(self, ids, layer: int, pos: int, vec: torch.Tensor):
        dec = self._decoder()

        def hook(_mod, _inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            h = h.clone()
            h[:, pos, :] = vec.to(h.dtype).to(h.device)
            return (h,) + out[1:] if tup else h

        handle = dec.layers[layer].register_forward_hook(hook)
        try:
            return self.model(input_ids=ids).logits.float()
        finally:
            handle.remove()


# --------------------------------------------------------------------------- TL

class TLBackend(Backend):
    """TransformerLens 3.7.0.

    Loaded with **no weight processing**, because folded
    LayerNorm / centred writing weights change the logits by exactly the kind of small
    amount that a "the feature isn't represented yet" result would be built on.
    """

    name = "tl"

    def load(self) -> "TLBackend":
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer

        tok_id = self.tokenizer_id or self.hf_id
        self.tok = AutoTokenizer.from_pretrained(tok_id)

        kwargs = dict(device=self.device, dtype=self.dtype, tokenizer=self.tok)
        if self.revision:
            kwargs["revision"] = self.revision
        kwargs.update(self.extra)

        # `from_pretrained_no_processing` is the convenience wrapper; the *semantics* is
        # what matters.  Fall back to the explicit flags if the wrapper
        # is gone in this version.
        if hasattr(HookedTransformer, "from_pretrained_no_processing"):
            self.model = HookedTransformer.from_pretrained_no_processing(self.hf_id, **kwargs)
        else:
            self.model = HookedTransformer.from_pretrained(
                self.hf_id, fold_ln=False, center_writing_weights=False,
                center_unembed=False, fold_value_biases=False,
                refactor_factored_attn_matrices=False, **kwargs,
            )
        self.model.eval()
        return self

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.model.cfg.d_model

    def resid_site(self, layer: int) -> str:
        from transformer_lens import utils

        return utils.get_act_name("resid_post", layer)

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(ids).float()

    @torch.no_grad()
    def read_resid(self, ids, layers, pos: int = -1):
        names = {self.resid_site(L): L for L in layers}
        _, cache = self.model.run_with_cache(ids, names_filter=lambda n: n in names)
        return {L: cache[n][0, pos].detach().float().cpu() for n, L in names.items()}

    @torch.no_grad()
    def write_resid(self, ids, layer: int, pos: int, vec: torch.Tensor):
        # TL 3.7.0 invokes hooks as `hook(tensor, hook=self)` — the second parameter
        # MUST be named `hook` or the call raises TypeError.
        def hook(act, hook):
            act[:, pos, :] = vec.to(act.dtype).to(act.device)
            return act

        return self.model.run_with_hooks(
            ids, fwd_hooks=[(self.resid_site(layer), hook)]
        ).float()


# --------------------------------------------------------------------------- nnsight

class NNSightBackend(Backend):
    """nnsight 0.7.0 fallback.

    nnsight traces the HF module graph, so there is no weight conversion and therefore no
    folding bug possible.  Hook "names" become module paths, which is why every caller
    goes through `read_resid` / `write_resid` instead of naming sites itself.

    ⚠ Unexercised as of the gate run — TL passed, so this path stayed cold.  The module
    path below is the Gemma-3 text tower under transformers 5.x and must be re-verified
    before any result is produced through this backend.
    """

    name = "nnsight"

    def load(self) -> "NNSightBackend":
        from nnsight import LanguageModel
        from transformers import AutoTokenizer

        tok_id = self.tokenizer_id or self.hf_id
        self.tok = AutoTokenizer.from_pretrained(tok_id)
        self.model = LanguageModel(
            self.hf_id, device_map=self.device, dtype=self.dtype,
            tokenizer=self.tok, dispatch=True,
            **({"revision": self.revision} if self.revision else {}),
        )
        return self

    def _layers(self):
        m = self.model
        for path in ("model.language_model.layers", "model.model.language_model.layers",
                     "model.model.layers", "model.layers"):
            cur = m
            ok = True
            for part in path.split("."):
                try:
                    cur = getattr(cur, part)
                except Exception:
                    ok = False
                    break
            if ok:
                self._layer_path = path
                return cur
        raise RuntimeError("could not locate the decoder layers on the nnsight model")

    @property
    def n_layers(self) -> int:
        return len(self._layers())

    @property
    def d_model(self) -> int:
        return self.model.config.text_config.hidden_size if hasattr(
            self.model.config, "text_config") else self.model.config.hidden_size

    def resid_site(self, layer: int) -> str:
        self._layers()
        return f"{self._layer_path}[{layer}].output[0]"

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        with self.model.trace(ids):
            out = self.model.output.logits.save()
        return out.float()

    @torch.no_grad()
    def read_resid(self, ids, layers, pos: int = -1):
        L = self._layers()
        saved = {}
        with self.model.trace(ids):
            for i in layers:
                saved[i] = L[i].output[0][0, pos].save()
        return {i: v.detach().float().cpu() for i, v in saved.items()}

    @torch.no_grad()
    def write_resid(self, ids, layer: int, pos: int, vec: torch.Tensor):
        L = self._layers()
        with self.model.trace(ids):
            L[layer].output[0][:, pos, :] = vec
            out = self.model.output.logits.save()
        return out.float()


# --------------------------------------------------------------------------- factory

BACKENDS = {"hf": HFBackend, "tl": TLBackend, "nnsight": NNSightBackend}


def get_backend(kind: str, hf_id: str, *, revision: str | None = None,
                dtype: str | torch.dtype = "float32", device: str = "cuda",
                tokenizer_id: str | None = None, **extra) -> Backend:
    if kind not in BACKENDS:
        raise ValueError(f"unknown backend {kind!r}; have {sorted(BACKENDS)}")
    dt = DTYPES[dtype] if isinstance(dtype, str) else dtype
    return BACKENDS[kind](hf_id=hf_id, revision=revision, dtype=dt, device=device,
                          tokenizer_id=tokenizer_id, extra=extra)


# --------------------------------------------------------------------------- prompts

# Gemma-3 chat template.  ⚠ `google/gemma-3-4b-pt` ships **no** chat template (verified
# 2026-08-08: `chat_template` absent from its tokenizer_config.json), so the interp prompts
# borrow the template from an `-it` repo, because `tok.apply_chat_template` raises on
# the base model.
#
# Which `-it` repo does not matter: the chat template is byte-identical across
# gemma-3-1b-it, gemma-3-4b-it and gemma-3-12b-it (sha256 7de1c58e208eda46…, 1532 bytes,
# checked 2026-08-08), and all three share the tokenizer (bos_token_id 2, |V| 262145).
# 1b-it is the default because it is the smallest repo to have cached.
GEMMA3_CHAT_TEMPLATE_SOURCE = "google/gemma-3-1b-it"
GEMMA3_CHAT_TEMPLATE_SHA256 = "7de1c58e208eda46"      # first 16 hex chars

# configs/prompts.yaml :: P1_minimal — the canonical prompt of the behavioural panel.
P1_MINIMAL = ("Translate the following English sentence into Tamil. "
              "Output only the Tamil translation.\n\n{source}")


def gemma_chat_template(tokenizer_id: str = GEMMA3_CHAT_TEMPLATE_SOURCE) -> str:
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(tokenizer_id)
    if t.chat_template is None:
        raise RuntimeError(f"{tokenizer_id} has no chat template to borrow")
    return t.chat_template


def render_prompt(tok, src_en: str, chat_template: str | None = None) -> str:
    """Chat-formatted P1_minimal prompt, ending at the model turn's generation prompt."""
    msgs = [{"role": "user", "content": P1_MINIMAL.format(source=src_en)}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        **({"chat_template": chat_template} if chat_template else {}),
    )


def assert_single_bos(ids: torch.Tensor, bos_id: int, what: str = "") -> int:
    """Gemma's chat template emits `<bos>` and most tokenizers prepend another.  A double
    BOS shifts every position by one and silently corrupts the decision index."""
    n = int((ids == bos_id).sum().item())
    if n != 1:
        raise AssertionError(f"expected exactly 1 BOS, got {n} {what}")
    return n


def divergence_index(prompt_ids: list[int], ids_a: list[int], ids_b: list[int]):
    """First index at which the two continuations differ.

    Returns (k, offset) or (None, None) when one form is a strict token-prefix of the
    other — such a pair is not expressible as a single next-token choice and is dropped.
    """
    a = list(prompt_ids) + list(ids_a)
    b = list(prompt_ids) + list(ids_b)
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    if k == min(len(a), len(b)):
        return None, None
    return k, k - len(prompt_ids)
