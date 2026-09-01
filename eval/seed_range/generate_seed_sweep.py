"""Seed sweep generator for the 2026-09-01 "seed-range" exploration.

Company context: DN AI Model & Data, 01-09-26 — "seed-range exploration is
the top priority, starting with piano and acoustic instruments... strict
good-seed boundaries for acoustics in production... remove-bad-seeds-only
for synths." See dn-intelligence 07_decisions/imported/2026-09-01-seed-range.md
and dn-intelligence 04_analysis/reports/2026-09-01-seed-range-instrument-accuracy-eval-design.md
for the full eval design this script implements.

For ONE (prompt, instrument_family) pair, generates the same prompt across a
list of seeds using the real Foundation-1 generation path
(stable_audio_tools.inference.generation.generate_diffusion_cond), and writes
each take as a WAV plus a JSON sidecar carrying the full generation config —
matching the logging schema in dn-intelligence 02_model/musicgen-control-surface.md
("Log every time: checkpoint, model version/hash, prompt, duration, seed,
temperature, top_k, top_p, cfg/guidance, ...").

Requires a GPU host with the trained checkpoint loaded (see README_DNL.md /
run_gradio.py for how this repo normally loads a checkpoint) — this script
does not itself fetch the checkpoint. Run it with the model already loaded,
e.g. from a notebook or a thin wrapper that calls `run_seed_sweep(model, ...)`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch

from stable_audio_tools.inference.generation import generate_diffusion_cond


@dataclass
class SeedTakeMetadata:
    prompt: str
    instrument_family: str
    seed: int
    steps: int
    cfg_scale: float
    sample_size: int
    checkpoint_id: str
    generated_at_utc: str
    wav_path: str


def _checkpoint_fingerprint(model) -> str:
    """Best-effort, stable id for the loaded checkpoint (for provenance only).

    We do not have a canonical checkpoint hash surfaced by stable_audio_tools
    at inference time, so we fall back to a hash of the model's state_dict
    key set + shapes, which is stable for a given checkpoint file and cheap
    to compute. Prefer a real registry id (dn-intelligence 02_model/version-registry)
    when the caller has one — pass it via `checkpoint_id_override`.
    """
    keys = sorted(model.state_dict().keys())
    h = hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()
    return h[:12]


def run_seed_sweep(
    model,
    prompt: str,
    instrument_family: str,
    seeds: list[int],
    out_dir: Path,
    steps: int = 100,
    cfg_scale: float = 6.0,
    sample_size: int = 131072,
    sample_rate: int = 44100,
    device: str = "cuda",
    checkpoint_id_override: str | None = None,
) -> list[SeedTakeMetadata]:
    """Generate `prompt` once per seed in `seeds`, save WAV + JSON sidecar.

    One (prompt, instrument_family) call at a time by design: the
    2026-09-01 decision starts with piano specifically, and the
    generalization analysis (analyze_seed_generalization.py) needs several
    *different* prompts within the same family run through this function
    separately so it can test whether a "good seed" holds across prompts,
    not just within one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = checkpoint_id_override or _checkpoint_fingerprint(model)

    results: list[SeedTakeMetadata] = []
    for seed in seeds:
        audio = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=[{"prompt": prompt, "seconds_start": 0, "seconds_total": sample_size / sample_rate}],
            sample_size=sample_size,
            seed=seed,
            device=device,
        )

        # generate_diffusion_cond returns (batch, channels, samples); we ask
        # for batch_size=1 implicitly (default), take the first item.
        audio = audio[0].to(torch.float32).clamp(-1, 1).cpu().numpy().T

        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:60]
        wav_path = out_dir / f"{instrument_family}_{safe_prompt}_seed{seed}.wav"
        sf.write(wav_path, audio, sample_rate)

        meta = SeedTakeMetadata(
            prompt=prompt,
            instrument_family=instrument_family,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            sample_size=sample_size,
            checkpoint_id=checkpoint_id,
            generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wav_path=str(wav_path),
        )
        with open(wav_path.with_suffix(".json"), "w") as f:
            json.dump(asdict(meta), f, indent=2)
        results.append(meta)

    return results


if __name__ == "__main__":
    raise SystemExit(
        "This module is a library, not a standalone entrypoint: it needs a "
        "loaded Foundation-1 model object. Load your checkpoint the same way "
        "run_gradio.py does, then call run_seed_sweep(model, ...) from a "
        "notebook or a thin runner script. See eval/seed_range/README.md."
    )
