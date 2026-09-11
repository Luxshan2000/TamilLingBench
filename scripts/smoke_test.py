#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Environment smoke test. Exit 0 = every component proven
on real Tamil rather than on an import statement."""
import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import subprocess
import sys
import shutil

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FAIL = []


def check(name, cond, got=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{got}]" if got else ""))
    if not cond:
        FAIL.append(name)


# 1. Python band
check("python 3.10<=v<=3.12", (3, 10) <= sys.version_info[:2] <= (3, 12),
      ".".join(map(str, sys.version_info[:3])))

# 2. foma present and applying Tamil
check("foma on PATH", shutil.which("foma") is not None)
check("flookup on PATH", shutil.which("flookup") is not None)

# 3. Real ThamizhiMorph model returns the suffix-bound analysis
FST = "third_party/thamizhi-morph/FST-Models/verb-c-rest.fst"
out = subprocess.run(["flookup", FST], input="வந்தான்\n",
                     capture_output=True, text=True).stdout
check("ThamizhiMorph verb-c-rest binds +3sgm=ஆன்", "+3sgm=ஆன்" in out, out.strip())

PRON = "third_party/thamizhi-morph/FST-Models/pronoun.fst"
out = subprocess.run(["flookup", PRON], input="நாம்\nநாங்கள்\n",
                     capture_output=True, text=True).stdout
check("pronoun.fst distinguishes clusivity",
      "+1pl+incl+nom" in out and "+1pl+excl+nom" in out)

# 3b. The union must NOT shadow the pronoun reading of நீ.
out = subprocess.run(["flookup", "-a", "build/thamizhi-union.bin"], input="நீ\n",
                     capture_output=True, text=True).stdout
check("union keeps BOTH readings of நீ (union net, not bare -a)",
      "+pron+2sg+nom" in out and "+verb+" in out, out.replace("\n", " ").strip())

# 4. TamilNormalizer composes the two-part vowel sign
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
norm = IndicNormalizerFactory().get_normalizer("ta")
check("get_normalizer('ta') is TamilNormalizer", type(norm).__name__ == "TamilNormalizer")
# NB: escapes on purpose — an editor that NFC-normalizes this file would otherwise
# silently turn the decomposed input into the composed one and void the test.
DECOMPOSED = "\u0B95\u0BC6\u0BBE"   # க + COMBINING e + COMBINING aa  (decomposed)
COMPOSED   = "\u0B95\u0BCA"          # க + COMBINING o               (composed)
check("two-part vowel sign ெ+ா -> ொ", norm.normalize(DECOMPOSED) == COMPOSED)

# 5. open-tamil grapheme split
import tamil.utf8
check("get_letters('வந்தான்')", tamil.utf8.get_letters("வந்தான்") == ["வ", "ந்", "தா", "ன்"])

# 6. open-tamil morphology is STILL broken -> guard against silent future use
from solthiruthi.morphology import RemoveVerbSuffixTense
check("solthiruthi.morphology confirmed broken (expected)",
      RemoveVerbSuffixTense().apply("வந்தான்") == ("வந்தான்", False))

# 7. Aksharamukha ISO 15919, incl. round-trip and the ள/ழ contrast
from aksharamukha import transliterate as tr
check("Tamil->ISO", tr.process("Tamil", "ISO", "வந்தான்") == "vantāṉ")
check("ISO keeps ழ != ள",
      tr.process("Tamil", "ISO", "வழி") == "vaḻi" and tr.process("Tamil", "ISO", "வளி") == "vaḷi")
forms = ["வந்தான்", "வந்தாள்", "வந்தார்", "வந்தார்கள்", "வந்தன", "வரவில்லை", "வராமல்", "வராதே",
         "போகமாட்டான்", "நாம்", "நாங்கள்", "சொன்னான்", "கொடுத்தார்", "வேண்டாம்", "இல்லாமல்",
         "அல்ல", "அன்று"]
rt = sum(tr.process("ISO", "Tamil", tr.process("Tamil", "ISO", w)) == w for w in forms)
check(f"ISO round-trip {rt}/{len(forms)}", rt == len(forms))

# 8. PyStemmer present (inventory source only)
import Stemmer
check("PyStemmer has 'tamil'", "tamil" in Stemmer.algorithms())

# 9. hfst importable
import hfst
check("hfst importable", hasattr(hfst, "HfstInputStream"), hfst.__version__)

# 10. UD treebanks present with expected token counts
def ntok(p):
    n = 0
    for line in open(p, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) == 10 and "-" not in f[0] and "." not in f[0]:
            n += 1
    return n


check("MWTT test = 2584 tokens", ntok("data/ud/UD_Tamil-MWTT/ta_mwtt-ud-test.conllu") == 2584,
      ntok("data/ud/UD_Tamil-MWTT/ta_mwtt-ud-test.conllu"))
check("TTB test  = 1989 tokens", ntok("data/ud/UD_Tamil-TTB/ta_ttb-ud-test.conllu") == 1989,
      ntok("data/ud/UD_Tamil-TTB/ta_ttb-ud-test.conllu"))

# 11. The checker itself imports and the D-2 policy is wired up
sys.path.insert(0, ".")
from tamillingbench.morph import MorphChecker, Slot
with MorphChecker() as c:
    check("checker binds +3sgm=ஆன்",
          any(b.label == "3sgm" and b.suffix == "ஆன்"
              for a in c.analyse("வந்தான்").analyses for b in a.bindings))
    check("வந்தார்கள் keeps BOTH 3sghe and 3ple", len(c.analyse("வந்தார்கள்").analyses) >= 2)
    cr_sg = c.check("வந்தார்கள்", Slot.HONORIFICITY, "POLITE", referent_number="SG")
    cr_none = c.check("வந்தார்கள்", Slot.HONORIFICITY, "POLITE")
    check("D-2 context filter resolves வந்தார்கள் under referent_number=SG", cr_sg.universal)
    check("D-2 marks it undecidable without context", cr_none.undecidable)

# --- GPU machine only; skipped when torch is absent ---
try:
    import torch
    from transformer_lens.supported_models import OFFICIAL_MODEL_NAMES
    check("TL lists gemma-3-4b-it", "google/gemma-3-4b-it" in OFFICIAL_MODEL_NAMES)
    check("TL lists gemma-3-12b-it", "google/gemma-3-12b-it" in OFFICIAL_MODEL_NAMES)
    check("TL lists Qwen3-8B", "Qwen/Qwen3-8B" in OFFICIAL_MODEL_NAMES)
    check("CUDA available", torch.cuda.is_available())     # expected FAIL on the Mac
except ImportError:
    print("SKIP  torch/transformer_lens absent (CPU-only box)")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
