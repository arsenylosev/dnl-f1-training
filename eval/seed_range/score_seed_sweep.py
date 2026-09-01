"""Automated scoring for seed-sweep takes produced by generate_seed_sweep.py.

Two proxy metrics, both real and CPU-runnable (no GPU required), meant to
pre-screen a seed sweep BEFORE spending human Label Studio time on it — they
are proxies for the company's actual gates (Sound Quality / Instrument
Accuracy 0-5 star, see dn-intelligence 04_analysis/benchmark-methodology.md),
not a replacement for them. Every score this script writes must still be
spot-checked against real Label Studio ratings before anyone treats a
seed-range recommendation as production-ready.

1. Instrument-accuracy proxy (IA_proxy): zero-shot CLAP audio-vs-text
   similarity, scored against the 9 frozen instrument families
   (dn-intelligence 07_decisions/0014-nine-core-family-taxonomy.md: Piano,
   Guitar, Bass/Sub, Brass, Flute, Synth, Pad, Pluck, Drums/Perc). This is
   the SAME technique and the SAME `msclap` library already used in
   production for auto-tagging training data
   (data-engineering/main_pipeline/CLAP_tagger/new_classification.py) — reused
   here, not reinvented, so a seed found "good" by this proxy is graded by a
   tool the team already trusts for a related job.

2. Sound-quality proxy (SQ_proxy): cheap DSP heuristics from librosa —
   clipping ratio, silence ratio, and spectral flatness — that catch the
   obviously-broken end of the distribution (silence, hard clipping,
   noise-only output) fast. It is deliberately NOT a claim of perceptual
   quality; it exists to cut the human-review set down before Label Studio.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

INSTRUMENT_FAMILIES = [
    "Piano",
    "Guitar",
    "Bass/Sub",
    "Brass",
    "Flute",
    "Synth",
    "Pad",
    "Pluck",
    "Drums/Perc",
]


@dataclass
class SeedScore:
    wav_path: str
    prompt: str
    instrument_family: str
    seed: int
    ia_proxy_target_score: float
    ia_proxy_top1_family: str
    ia_proxy_top1_score: float
    ia_proxy_correct: bool
    sq_clipping_ratio: float
    sq_silence_ratio: float
    sq_spectral_flatness: float


def _load_clap(use_cuda: bool = False):
    """Loads the same CLAP wrapper as the production auto-tagger.

    Deferred import: `msclap` is only needed for scoring, not for
    generation, and pulls ~1.7 GB of pretrained weights on first use.
    """
    from msclap import CLAP

    return CLAP(version="2023", use_cuda=use_cuda)


def score_instrument_accuracy_proxy(model, wav_path: Path, target_family: str) -> dict:
    families = INSTRUMENT_FAMILIES
    text_emb = model.get_text_embeddings(families)
    audio_emb = model.get_audio_embeddings([str(wav_path)])
    raw_scores = model.compute_similarity(audio_emb, text_emb)[0]
    probs = F.softmax(torch.tensor(raw_scores), dim=0).tolist()

    ranked = sorted(zip(families, probs), key=lambda x: x[1], reverse=True)
    target_score = dict(zip(families, probs)).get(target_family, float("nan"))
    top1_family, top1_score = ranked[0]

    return {
        "ia_proxy_target_score": target_score,
        "ia_proxy_top1_family": top1_family,
        "ia_proxy_top1_score": top1_score,
        "ia_proxy_correct": top1_family == target_family,
    }


def score_sound_quality_proxy(wav_path: Path) -> dict:
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0:
        return {"sq_clipping_ratio": 1.0, "sq_silence_ratio": 1.0, "sq_spectral_flatness": float("nan")}

    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))
    silence_ratio = float(np.mean(np.abs(y) < 1e-4))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    return {
        "sq_clipping_ratio": clipping_ratio,
        "sq_silence_ratio": silence_ratio,
        "sq_spectral_flatness": flatness,
    }


def score_sweep_dir(sweep_dir: Path, out_csv: Path, use_cuda: bool = False) -> list[SeedScore]:
    """Scores every (wav, json-sidecar) pair written by generate_seed_sweep.py."""
    clap_model = _load_clap(use_cuda=use_cuda)
    rows: list[SeedScore] = []

    for wav_path in sorted(sweep_dir.glob("*.wav")):
        sidecar_path = wav_path.with_suffix(".json")
        if not sidecar_path.exists():
            continue
        meta = json.loads(sidecar_path.read_text())

        ia = score_instrument_accuracy_proxy(clap_model, wav_path, meta["instrument_family"])
        sq = score_sound_quality_proxy(wav_path)

        rows.append(
            SeedScore(
                wav_path=str(wav_path),
                prompt=meta["prompt"],
                instrument_family=meta["instrument_family"],
                seed=meta["seed"],
                **ia,
                **sq,
            )
        )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field for field in SeedScore.__dataclass_fields__])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="Directory of WAV+JSON pairs from generate_seed_sweep.py")
    parser.add_argument("out_csv", type=Path, help="Where to write the scored CSV")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for CLAP (falls back to CPU by default)")
    args = parser.parse_args()

    scored = score_sweep_dir(args.sweep_dir, args.out_csv, use_cuda=args.cuda)
    print(f"Scored {len(scored)} takes -> {args.out_csv}")
