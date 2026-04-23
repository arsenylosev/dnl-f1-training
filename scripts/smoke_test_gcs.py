#!/usr/bin/env python3
"""
GCS Data Pipeline Smoke Test
=============================
Validates the Foundation-1 GCS dataset before committing to a full
pre-encoding run.  Loads N WAV+JSON pairs from each configured GCS
prefix and checks:

  1. Object listing   — bucket/prefix is reachable and contains WAV files
  2. JSON sidecar     — sibling .json exists and contains a "text" field
  3. Audio loading    — WAV decodes without error via torchaudio
  4. Sample rate      — audio is 44100 Hz (or will be resampled)
  5. Channel count    — audio is stereo (2 channels)
  6. Duration         — clip is ≥ 3 seconds (132300 samples at 44100 Hz)
  7. Prompt format    — "text" value is a non-empty string

Usage:
    # Authenticate first (on local machine):
    gcloud auth application-default login

    # Run against your bucket (replace placeholders):
    python3 scripts/smoke_test_gcs.py \
        --bucket your-gcs-bucket \
        --prefixes training_datasets/f1_initial/train/ \
                   training_datasets/f1_initial/valid/ \
                   training_datasets/f1_initial/test/ \
        --n-samples 10 \
        --target-sr 44100 \
        --min-duration-s 3.0

    # Verbose mode prints the full prompt for each sample:
    python3 scripts/smoke_test_gcs.py --bucket your-bucket --verbose

Exit codes:
    0  All checks passed
    1  One or more checks failed (details printed to stderr)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — fail with a helpful message if not installed
# ---------------------------------------------------------------------------
try:
    from google.cloud import storage as gcs
except ImportError:
    print(
        "ERROR: google-cloud-storage is not installed.\n"
        "       Run: pip install google-cloud-storage",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import torch
except Exception as _e:
    # find_spec only checks whether the package directory exists on disk;
    # it does NOT attempt to load the C-extension.  A missing CUDA .so
    # (libcuda.so.1, libcublas.so, etc.) will therefore pass find_spec but
    # still raise ImportError / OSError here.  Always print the real error.
    import importlib.util as _ilu
    _present = _ilu.find_spec("torch") is not None
    if not _present:
        print(
            "ERROR: torch is not installed.\n"
            "       Run: pip install torch",
            file=sys.stderr,
        )
    else:
        print(
            f"ERROR: torch is installed but failed to import.\n"
            f"       This is usually a CUDA library mismatch or a missing shared\n"
            f"       library (e.g. libcuda.so.1, libcublas.so).\n"
            f"       Exception type : {type(_e).__name__}\n"
            f"       Underlying error: {_e}",
            file=sys.stderr,
        )
    sys.exit(1)

try:
    import torchaudio
except Exception as _e:
    import importlib.util as _ilu
    _present = _ilu.find_spec("torchaudio") is not None
    if not _present:
        print(
            "ERROR: torchaudio is not installed.\n"
            "       Run: pip install torchaudio",
            file=sys.stderr,
        )
    else:
        print(
            f"ERROR: torchaudio is installed but failed to import.\n"
            f"       Exception type : {type(_e).__name__}\n"
            f"       Underlying error: {_e}",
            file=sys.stderr,
        )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    wav_key: str
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Populated on success
    shape: Optional[tuple] = None
    sample_rate: Optional[int] = None
    duration_s: Optional[float] = None
    prompt: Optional[str] = None
    load_time_s: Optional[float] = None


@dataclass
class PrefixResult:
    prefix: str
    n_found: int = 0
    n_tested: int = 0
    n_passed: int = 0
    samples: list[SampleResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _make_client() -> gcs.Client:
    """Return a GCS client using Application Default Credentials."""
    return gcs.Client()


def _list_wav_keys(client: gcs.Client, bucket_name: str, prefix: str, limit: int) -> list[str]:
    """Return up to *limit* WAV object keys under *prefix*."""
    blobs = client.list_blobs(bucket_name, prefix=prefix, max_results=limit * 10)
    keys = []
    for blob in blobs:
        if blob.name.lower().endswith(".wav"):
            keys.append(blob.name)
            if len(keys) >= limit:
                break
    return keys


def _download_bytes(client: gcs.Client, bucket_name: str, key: str) -> bytes:
    bucket = client.bucket(bucket_name)
    return bucket.blob(key).download_as_bytes()


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def _validate_sample(
    client: gcs.Client,
    bucket_name: str,
    wav_key: str,
    target_sr: int,
    min_samples: int,
    verbose: bool,
) -> SampleResult:
    result = SampleResult(wav_key=wav_key)
    t0 = time.perf_counter()

    # ---- 1. Download JSON sidecar ----------------------------------------
    json_key = wav_key[: -len(".wav")] + ".json"
    try:
        raw_json = _download_bytes(client, bucket_name, json_key)
        meta = json.loads(raw_json.decode("utf-8"))
    except Exception as exc:
        result.errors.append(f"JSON sidecar missing or unreadable: {exc}")
        result.passed = False
        meta = {}

    # ---- 2. Check "text" field -------------------------------------------
    text = meta.get("text") or meta.get("prompt")
    if not text:
        result.errors.append(
            f"JSON has no 'text' or 'prompt' field. Keys present: {list(meta.keys())}"
        )
        result.passed = False
    elif not isinstance(text, str) or len(text.strip()) == 0:
        result.errors.append(f"'text' field is empty or not a string: {text!r}")
        result.passed = False
    else:
        result.prompt = text.strip()

    # ---- 3. Download and decode WAV --------------------------------------
    try:
        raw_wav = _download_bytes(client, bucket_name, wav_key)
        audio, sr = torchaudio.load(io.BytesIO(raw_wav))
    except Exception as exc:
        result.errors.append(f"WAV decode failed: {exc}")
        result.passed = False
        result.load_time_s = time.perf_counter() - t0
        return result

    result.shape = tuple(audio.shape)
    result.sample_rate = sr
    result.duration_s = audio.shape[-1] / sr
    result.load_time_s = time.perf_counter() - t0

    # ---- 4. Sample rate check -------------------------------------------
    if sr != target_sr:
        result.warnings.append(
            f"Sample rate is {sr} Hz (expected {target_sr} Hz). "
            "Will be resampled automatically by GCSDataset."
        )

    # ---- 5. Channel count -----------------------------------------------
    n_channels = audio.shape[0]
    if n_channels == 1:
        result.warnings.append(
            "Audio is mono (1 channel). GCSDataset will up-mix to stereo."
        )
    elif n_channels != 2:
        result.errors.append(
            f"Unexpected channel count: {n_channels}. Expected 1 (mono) or 2 (stereo)."
        )
        result.passed = False

    # ---- 6. Duration check ----------------------------------------------
    n_samples = audio.shape[-1]
    if n_samples < min_samples:
        actual_s = n_samples / sr
        min_s = min_samples / target_sr
        result.errors.append(
            f"Clip too short: {actual_s:.2f}s ({n_samples} samples). "
            f"Minimum required: {min_s:.1f}s ({min_samples} samples at {target_sr} Hz)."
        )
        result.passed = False

    # ---- 7. Silence check -----------------------------------------------
    rms = audio.float().pow(2).mean().sqrt().item()
    if rms < 1e-6:
        result.warnings.append(f"Audio appears to be silent (RMS={rms:.2e}).")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_smoke_test(
    bucket_name: str,
    prefixes: list[str],
    n_samples: int,
    target_sr: int,
    min_duration_s: float,
    verbose: bool,
) -> bool:
    """Run the full smoke test. Returns True if all checks passed."""

    min_samples = int(min_duration_s * target_sr)
    print(f"\n{'='*70}")
    print(f"  Foundation-1 GCS Data Pipeline Smoke Test")
    print(f"{'='*70}")
    print(f"  Bucket       : gs://{bucket_name}")
    print(f"  Prefixes     : {prefixes}")
    print(f"  Samples/prefix: {n_samples}")
    print(f"  Target SR    : {target_sr} Hz")
    print(f"  Min duration : {min_duration_s}s ({min_samples} samples)")
    print(f"{'='*70}\n")

    try:
        client = _make_client()
        # Quick connectivity check
        client.get_bucket(bucket_name)
        print(f"  ✓ GCS bucket gs://{bucket_name} is accessible\n")
    except Exception as exc:
        print(f"  ✗ Cannot access GCS bucket gs://{bucket_name}: {exc}", file=sys.stderr)
        print(
            "\n  Hint: Run 'gcloud auth application-default login' on a local machine,\n"
            "        or ensure the Vertex AI service account has 'Storage Object Viewer' role.",
            file=sys.stderr,
        )
        return False

    all_passed = True
    prefix_results: list[PrefixResult] = []

    for prefix in prefixes:
        pr = PrefixResult(prefix=prefix)
        prefix_results.append(pr)

        print(f"  Prefix: gs://{bucket_name}/{prefix}")

        # List WAV files
        try:
            wav_keys = _list_wav_keys(client, bucket_name, prefix, n_samples)
        except Exception as exc:
            print(f"    ✗ Listing failed: {exc}", file=sys.stderr)
            all_passed = False
            continue

        pr.n_found = len(wav_keys)
        if pr.n_found == 0:
            print(f"    ✗ No WAV files found under this prefix.", file=sys.stderr)
            all_passed = False
            continue

        print(f"    Found {pr.n_found} WAV file(s) (testing first {min(pr.n_found, n_samples)})")

        for wav_key in wav_keys[:n_samples]:
            pr.n_tested += 1
            result = _validate_sample(
                client, bucket_name, wav_key, target_sr, min_samples, verbose
            )
            pr.samples.append(result)

            status = "✓" if result.passed else "✗"
            duration_str = f"{result.duration_s:.2f}s" if result.duration_s else "?"
            shape_str = str(result.shape) if result.shape else "?"
            sr_str = str(result.sample_rate) if result.sample_rate else "?"
            load_str = f"{result.load_time_s:.2f}s" if result.load_time_s else "?"

            print(
                f"    {status} {wav_key.split('/')[-1]:<40} "
                f"shape={shape_str:<14} sr={sr_str:<6} dur={duration_str:<8} load={load_str}"
            )

            if verbose and result.prompt:
                print(f"       prompt: {result.prompt[:120]}")

            for w in result.warnings:
                print(f"       ⚠  {w}")
            for e in result.errors:
                print(f"       ✗  {e}", file=sys.stderr)

            if result.passed:
                pr.n_passed += 1
            else:
                all_passed = False

        print()

    # ---- Summary -----------------------------------------------------------
    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for pr in prefix_results:
        status = "PASS" if pr.n_passed == pr.n_tested else "FAIL"
        print(
            f"  [{status}] gs://{bucket_name}/{pr.prefix}\n"
            f"         {pr.n_passed}/{pr.n_tested} samples passed "
            f"({pr.n_found} total WAV files found)"
        )

    print(f"{'='*70}")
    if all_passed:
        print("  ✓ All checks passed. Safe to proceed with pre-encoding.\n")
    else:
        print(
            "  ✗ Some checks failed. Fix the issues above before running pre_encode.py.\n",
            file=sys.stderr,
        )

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Foundation-1 GCS data pipeline before pre-encoding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name (without gs:// prefix)",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=[
            "training_datasets/f1_initial/train/",
            "training_datasets/f1_initial/valid/",
            "training_datasets/f1_initial/test/",
        ],
        help="One or more GCS object key prefixes to test (space-separated)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to test per prefix (default: 10)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=44100,
        help="Expected sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=3.0,
        help="Minimum clip duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full prompt string for each sample",
    )
    args = parser.parse_args()

    passed = run_smoke_test(
        bucket_name=args.bucket,
        prefixes=args.prefixes,
        n_samples=args.n_samples,
        target_sr=args.target_sr,
        min_duration_s=args.min_duration_s,
        verbose=args.verbose,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
