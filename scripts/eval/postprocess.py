# -*- coding: utf-8 -*-
"""Output hygiene. THE SINGLE IMPLEMENTATION, imported by both sides.

`generate.py` (GPU machine) applies it when writing `hypothesis`; `score.py` (local) re-applies it
to the preserved `hypothesis_raw`. That re-application is deliberate and is the whole reason
`hypothesis_raw` is durable: a cleaning bug found after a 3-hour panel run is fixed by
re-scoring, not by re-generating.

Every operation is recorded in `clean_ops`. Nothing is stripped silently, and the per-model
rate is reported — a model needing cleaning on 30% of outputs is a caveat, not a footnote.

The rule that matters most in practice: **Gemma-3 appends a parenthetical romanisation** to an
otherwise correct Tamil translation
(`அவர்கள் ... பேசினார்கள். (Avargal ... pesinaargal.)`). Left in, it drags the Tamil-character
share of a *correct* output below the 0.9 acceptance threshold and misreports the model as
producing non-Tamil. Measured on gemma-3-4b before this rule: 915/1025 C2 outputs flagged
low-Tamil; nearly all were correct Tamil plus a gloss.
"""
from __future__ import annotations

import re

TAMIL = re.compile(r"[஀-௿]")
THINK_OPEN = re.compile(r"<think\b", re.I)
THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
LABEL_PREFIX = re.compile(r"^\s*(tamil|translation|தமிழ்|மொழிபெயர்ப்பு)\s*[:：]\s*", re.I)
#: A line the model has itself labelled as the TRANSLATION, or as the CONTEXT.
#:
#: ⚠ THIS IS A CORRECTNESS FIX, NOT COSMETICS. On a C2 item the prompt carries a CONTEXT
#: sentence and a TRANSLATE sentence, and the context is where the disambiguating referent
#: lives ("Meena had been waiting..."). sarvam-translate answers with BOTH lines labelled in
#: Tamil:
#:     சூழல்: மீனா ... காத்திருந்தாள்.        <- the CONTEXT, re-translated (Meena -> -ாள் FEM)
#:     மொழிபெயர்ப்பு: நண்பகலில் வந்தார்கள்.   <- the actual answer (-ஆர்கள், epicene)
#: `take_first_tamil_line` picked the first line, i.e. the *context*, whose gender is handed
#: to the model for free by the proper noun. That scored sarvam-translate as CORRECT/committed
#: on gender at 71.8% while its real answer avoided exactly like its own base model. 27.9% of
#: sarvam-translate's outputs take this shape; no other system in the panel emits it at all.
#: A labelled translation line therefore WINS, and a labelled context line is never eligible.
TRANS_LINE = re.compile(r"^\s*(மொழிபெயர்ப்பு|translation|tamil|தமிழ்)\s*[:：]\s*", re.I)
CTX_LINE = re.compile(r"^\s*(சூழல்|சூழ்நிலை|context)\s*[:：]\s*", re.I)
TRAILING_PAREN = re.compile(r"\s*[\(\[]([^()\[\]]*)[\)\]]\s*$")
#: A markdown/explanatory tail the instruct models add after the translation.
EXPLAIN_TAIL = re.compile(r"\n\s*(\*\*|##|Explanation\b|Note\b|Here'?s\b)", re.I)


def tamil_share(s: str) -> float:
    nw = [c for c in s if not c.isspace()]
    return sum(bool(TAMIL.match(c)) for c in nw) / len(nw) if nw else 0.0


def clean(raw: str) -> tuple[str, list[str]]:
    ops: list[str] = []
    s = raw

    if THINK_OPEN.search(s):
        s2 = THINK_BLOCK.sub("", s)
        if s2 != s:
            ops.append("strip_think_block")
            s = s2
        if THINK_OPEN.search(s):          # unclosed — flag loudly, never silently keep
            ops.append("LEAKED_THINK_UNCLOSED")
    s = s.strip()

    if s.startswith("```"):
        s = FENCE.sub("", s).strip()
        ops.append("strip_code_fence")

    m = EXPLAIN_TAIL.search(s)
    if m and TAMIL.search(s[:m.start()]):
        s = s[:m.start()].strip()
        ops.append("strip_explanation_tail")

    if LABEL_PREFIX.match(s):
        s = LABEL_PREFIX.sub("", s, count=1)
        ops.append("strip_label_prefix")

    lines = [l for l in s.splitlines() if l.strip()]
    if len(lines) > 1:
        labelled = [l for l in lines if TRANS_LINE.match(l)]
        if labelled:
            # The model told us which line is the answer. Believe it over line order.
            s = TRANS_LINE.sub("", labelled[0], count=1).strip()
            ops.append("take_labelled_translation_line")
        else:
            # No explicit answer label: a line the model labelled CONTEXT is still never the
            # answer, so drop those before falling back to first-Tamil-line.
            body = [l for l in lines if not CTX_LINE.match(l)]
            if body and len(body) != len(lines):
                ops.append("drop_context_labelled_line")
            else:
                body = lines
            tam = [l for l in body if TAMIL.search(l)]
            pick = tam[0] if tam else body[0]
            if pick != s:
                s = pick
                ops.append("take_first_tamil_line")
    s = s.strip()

    if len(s) >= 2 and s[0] in "\"'“‘" and s[-1] in "\"'”’":
        s = s[1:-1].strip()
        ops.append("strip_quotes")

    # Trailing parenthetical romanisation / gloss. Only stripped when the parenthetical is
    # NOT Tamil and what remains still IS — so a legitimate Tamil parenthetical survives.
    for _ in range(3):
        m = TRAILING_PAREN.search(s)
        if not m:
            break
        inner, head = m.group(1), s[:m.start()].strip()
        if tamil_share(inner) < 0.2 and TAMIL.search(head):
            s = head
            ops.append("strip_trailing_transliteration")
        else:
            break

    return s.strip(), ops
