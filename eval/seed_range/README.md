# Seed-range eval harness

Implements the eval design in dn-intelligence
`04_analysis/reports/2026-09-01-seed-range-instrument-accuracy-eval-design.md`,
which extends the 2026-09-01 decision recorded in dn-intelligence
`07_decisions/imported/2026-09-01-seed-range.md`:

> Seed-range exploration is the top priority, starting with piano and
> acoustic instruments. Acoustic instruments may use strict good-seed
> boundaries in production to improve stability. Synthesizers should use
> the opposite approach: remove bad seeds while preserving variety.

## Status: code lands ready-to-run, not yet run

This harness was written and committed by an agent session that did **not**
have a working path to the trained Foundation-1 checkpoint, a GPU, or the
production inference stack (`dnl-inference-backend` behind
`kubectl port-forward` into the TEST EKS cluster) — every credential/network
path it tried was either expired, invalid, or denied by egress policy. See
the "What actually ran" section of the linked report for the exact blockers
and what a human needs to unblock. **Nothing in this directory has been
exercised against real Foundation-1 output.** Treat it as reviewed-but-untested
code, not a validated tool, until someone runs it on a GPU host with the
checkpoint loaded.

## What's here

| File | Purpose |
|---|---|
| `generate_seed_sweep.py` | For one `(prompt, instrument_family)`, generates the same prompt across a list of seeds via `stable_audio_tools.inference.generation.generate_diffusion_cond`. Writes WAV + JSON sidecar per take (prompt, seed, cfg_scale, steps, checkpoint fingerprint, timestamp) — the logging schema dn-intelligence `02_model/musicgen-control-surface.md` §"Log every time" asks for. |
| `score_seed_sweep.py` | Scores a sweep directory: an instrument-accuracy proxy (CLAP audio-vs-family-text similarity, reusing the same `msclap` approach as `data-engineering/main_pipeline/CLAP_tagger`) and a sound-quality proxy (clipping/silence/spectral-flatness heuristics via librosa). CPU-only capable. Writes one row per take to a CSV. |
| `analyze_seed_generalization.py` | The methodological check this whole exercise hinges on: does a seed that scores well for one prompt in a family also score well for a *different* prompt in the same family? Splits prompts into two folds, Spearman-correlates per-seed mean scores across folds. A weak/negative correlation means "good seed" is prompt-specific noise, not a property worth hard-coding into production — see the report for why this matters before anyone ships a seed allowlist. |

## How to run it for real (once unblocked)

1. Load the trained Foundation-1 checkpoint the same way `run_gradio.py`
   does (see `README_DNL.md`), on a GPU host.
2. Pick an instrument family and 3-5 different prompts within it (start
   with **Piano**, per the decision). For each prompt, call
   `generate_seed_sweep.run_seed_sweep(model, prompt, "Piano", seeds=range(0, 64), out_dir=...)`
   with the SAME seed list every time — the generalization test needs seeds
   to repeat across prompts.
3. Run `score_seed_sweep.py <sweep_dir> <out.csv>` per prompt directory (or
   point it at one directory holding all of them).
4. Concatenate the CSVs and run
   `analyze_seed_generalization.py <csv...> --family Piano`.
5. Whatever the verdict, spot-check the top and bottom seeds by ear and
   against real Label Studio Instrument Accuracy / Sound Quality ratings
   (`04_analysis/benchmark-methodology.md`) before trusting the CLAP proxy
   at face value — it is a screening tool, not the gate.
6. Repeat for a synth-family prompt set to test the opposite policy
   ("remove bad seeds while preserving variety" — the desired outcome
   there is a much weaker per-seed effect, since variety is the point).

## Why CLAP for the instrument-accuracy proxy, not a new classifier

`data-engineering/main_pipeline/CLAP_tagger/new_classification.py` already
uses `msclap` in production to auto-tag training audio against text labels,
CPU-fallback included. Foundation-1's own conditioner is also CLAP-based
(`models/foundation1_3s/model_config_3s.json`, `"type": "clap_text"`).
Reusing the same embedding approach for scoring keeps this harness
consistent with tooling the team already trusts, instead of introducing a
new external classifier (e.g. YAMNet/PANNs) with its own calibration
unknowns.
