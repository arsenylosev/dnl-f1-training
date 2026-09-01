"""Tests whether a "good seed" is a real, prompt-independent property.

This is the load-bearing question behind the 2026-09-01 decision
("strict good-seed boundaries for acoustics... in production"): a seed
allowlist/exclusion policy is only safe to ship if a seed that scores well
for one piano prompt also tends to score well for OTHER piano prompts. If
seed quality is actually dominated by prompt-specific interaction (i.e. a
seed that is great for "warm felt piano, soft attack" is unremarkable for
"bright grand piano, staccato"), then a fixed seed range calibrated on a
handful of prompts will silently misfire on prompts outside that set —
the opposite of the stability the decision is meant to buy.

Method: given a scored CSV covering >= 2 DISTINCT prompts in the same
instrument family (each swept across the same seed list — run
generate_seed_sweep.py once per prompt, then score_seed_sweep.py, then
concatenate the CSVs), split prompts into two folds. Rank seeds by mean
ia_proxy_target_score within fold A; check whether that ranking predicts
seed performance in fold B (Spearman correlation). A near-zero or negative
correlation is evidence AGAINST a global good-seed policy for this family;
a strong positive correlation supports one.

This does not replace a human Label Studio read — ia_proxy_target_score is
a CLAP proxy (see score_seed_sweep.py docstring), so treat the output as a
prioritised shortlist for human review, not a shipped policy.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


@dataclass
class SeedGeneralizationReport:
    instrument_family: str
    n_prompts: int
    n_seeds: int
    fold_a_prompts: list[str]
    fold_b_prompts: list[str]
    spearman_rho: float
    spearman_p: float
    verdict: str
    per_seed_mean_score: dict[int, float]


def load_scored_csvs(csv_paths: list[Path]) -> list[dict]:
    rows = []
    for path in csv_paths:
        with open(path, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def analyze_generalization(rows: list[dict], instrument_family: str, metric: str = "ia_proxy_target_score") -> SeedGeneralizationReport:
    family_rows = [r for r in rows if r["instrument_family"] == instrument_family]
    prompts = sorted(set(r["prompt"] for r in family_rows))
    seeds = sorted(set(int(r["seed"]) for r in family_rows))

    if len(prompts) < 2:
        raise ValueError(
            f"Need >= 2 distinct prompts in family {instrument_family!r} to test "
            f"generalization; got {len(prompts)}. Run generate_seed_sweep.py on more "
            f"prompts within this family first."
        )

    # Deterministic, order-preserving split: first half of sorted prompts is
    # fold A, the rest is fold B. For a real study, prefer a random split
    # repeated across many resamples rather than trusting one split.
    midpoint = len(prompts) // 2 or 1
    fold_a_prompts, fold_b_prompts = prompts[:midpoint], prompts[midpoint:]
    if not fold_b_prompts:
        fold_b_prompts = fold_a_prompts

    def per_seed_mean(fold_prompts: list[str]) -> dict[int, float]:
        by_seed: dict[int, list[float]] = defaultdict(list)
        for r in family_rows:
            if r["prompt"] in fold_prompts:
                by_seed[int(r["seed"])].append(float(r[metric]))
        return {seed: float(np.mean(vals)) for seed, vals in by_seed.items() if vals}

    scores_a = per_seed_mean(fold_a_prompts)
    scores_b = per_seed_mean(fold_b_prompts)
    common_seeds = sorted(set(scores_a) & set(scores_b))

    if len(common_seeds) < 3:
        raise ValueError(
            f"Only {len(common_seeds)} seeds have data in both folds; need >= 3 "
            f"for a meaningful Spearman correlation. Sweep the same seed list "
            f"across all prompts in this family."
        )

    a_vals = [scores_a[s] for s in common_seeds]
    b_vals = [scores_b[s] for s in common_seeds]
    rho, p = spearmanr(a_vals, b_vals)

    if np.isnan(rho):
        verdict = "INCONCLUSIVE (degenerate scores — check for ties/zero variance)"
    elif rho >= 0.5 and p < 0.05:
        verdict = "SUPPORTS a global good-seed policy for this family (strong, significant positive correlation)"
    elif rho <= 0.0:
        verdict = "CONTRADICTS a global good-seed policy — seed quality looks prompt-specific, not a seed property"
    else:
        verdict = "WEAK/INCONCLUSIVE — gather more prompts before committing to a fixed seed range"

    per_seed_mean_all = per_seed_mean(prompts)

    return SeedGeneralizationReport(
        instrument_family=instrument_family,
        n_prompts=len(prompts),
        n_seeds=len(seeds),
        fold_a_prompts=fold_a_prompts,
        fold_b_prompts=fold_b_prompts,
        spearman_rho=float(rho),
        spearman_p=float(p),
        verdict=verdict,
        per_seed_mean_score=per_seed_mean_all,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored_csvs", type=Path, nargs="+", help="One or more CSVs from score_seed_sweep.py")
    parser.add_argument("--family", required=True, help="Instrument family to analyze, e.g. Piano")
    parser.add_argument("--metric", default="ia_proxy_target_score")
    args = parser.parse_args()

    rows = load_scored_csvs(args.scored_csvs)
    report = analyze_generalization(rows, args.family, metric=args.metric)
    print(json.dumps(report.__dict__, indent=2))
