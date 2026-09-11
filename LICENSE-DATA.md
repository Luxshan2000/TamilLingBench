# Data licence

## Benchmark data: CC BY-SA 4.0

The benchmark items, templates, lexicons and derived result files in `data/`, `templates/` and
`results/` are licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-sa/4.0/).

This licence is required rather than chosen. Some material derives from
[UD_Tamil-MWTT](https://github.com/UniversalDependencies/UD_Tamil-MWTT), which is CC BY-SA 4.0,
and its ShareAlike term carries over. UD_Tamil-TTB (CC BY-NC-SA 3.0) is not redistributed here:
only aggregate counts computed from it appear in the results, and
`scripts/check_release_licence.py` checks the release for leakage.

## Code: Apache-2.0

Everything in `tamillingbench/`, `scripts/` and `tests/` is licensed under the Apache License 2.0
(see `LICENSE`).

## Model outputs: each model's own terms

`outputs/` holds the raw translations produced by the evaluated systems, together with their
scored records. They are included so that every number in the paper can be checked. They are
not covered by the CC BY-SA grant above: each system's outputs remain subject to the licence or
terms of use of the model that produced them.

| System | Licence (Hugging Face model card) |
|---|---|
| Gemma 3 (1B, 4B, 12B, 27B), Gemma 3n (E2B, E4B) | Gemma Terms of Use |
| Qwen3 (8B, 14B, 32B) | Apache-2.0 |
| Llama 3.1 8B Instruct | Llama 3.1 Community License |
| Sarvam-Translate | GPL-3.0 |
| NLLB-200 3.3B | CC BY-NC 4.0 (non-commercial) |
| IndicTrans2 en-indic 1B | MIT |

If you plan any use beyond research reproduction, check the current terms of the relevant model.
