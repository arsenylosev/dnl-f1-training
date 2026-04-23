"""
GCS-native dataset loader for stable-audio-tools
=================================================
Streams WAV + JSON pairs directly from a Google Cloud Storage bucket
without requiring gcsfuse or a local copy of the data.

Design decisions:
- Uses google-cloud-storage for object listing and streaming.
- Reads audio via torchaudio from an in-memory BytesIO buffer.
- Mirrors the SampleDataset interface so it plugs into
  create_dataloader_from_config with dataset_type = "gcs".
- Each worker opens its own GCS client to avoid thread-safety issues.

Source: https://cloud.google.com/python/docs/reference/storage/latest
Source: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import time
from typing import Callable, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T

from .utils import PadCrop_Normalized_T, PhaseFlipper, Stereo, Mono

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gcs_client():
    """Return a GCS client.

    Uses Application Default Credentials (ADC) when running on GCP
    (Vertex AI, GCE, Cloud Run) or when GOOGLE_APPLICATION_CREDENTIALS
    is set locally.

    Source: https://cloud.google.com/docs/authentication/application-default-credentials
    """
    from google.cloud import storage  # lazy import — only needed for gcs type
    return storage.Client()


def _list_wav_keys(bucket_name: str, prefix: str, max_items: Optional[int] = None) -> list[str]:
    """Return all .wav object keys under *prefix* in *bucket_name*."""
    client = _make_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    keys = []
    for blob in blobs:
        if blob.name.endswith(".wav"):
            keys.append(blob.name)
            if max_items is not None and len(keys) >= max_items:
                break
    return keys


def _read_bytes(bucket_name: str, key: str) -> bytes:
    """Download an object from GCS and return its raw bytes."""
    client = _make_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    return blob.download_as_bytes()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GCSDataset(Dataset):
    """
    PyTorch Dataset that streams WAV + JSON pairs from a GCS bucket.

    Each WAV file is expected to have a sibling JSON file with the same
    base name (e.g. ``abc123.wav`` / ``abc123.json``).  The JSON must
    contain a ``text`` field which is used as the conditioning prompt.

    Parameters
    ----------
    bucket_name:
        GCS bucket name (without ``gs://`` prefix).
    prefix:
        Object key prefix to search under (e.g. ``datasets/f1_initial/train/``).
    sample_size:
        Number of audio samples to pad/crop each clip to.
    sample_rate:
        Target sample rate in Hz.  Audio is resampled if it differs.
    random_crop:
        If True, the crop position within the audio is randomised.
    force_channels:
        ``"stereo"`` or ``"mono"``.
    custom_metadata_fn:
        Optional callable ``(info: dict, audio: Tensor) -> dict`` that
        can inject extra conditioning keys into the info dict.
    max_items:
        Limit the number of files loaded (useful for smoke-testing).
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str,
        sample_size: int = 132300,
        sample_rate: int = 44100,
        random_crop: bool = False,
        force_channels: str = "stereo",
        custom_metadata_fn: Optional[Callable] = None,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.custom_metadata_fn = custom_metadata_fn

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)
        self.augs = torch.nn.Sequential(PhaseFlipper())
        self.encoding = torch.nn.Sequential(
            Stereo() if force_channels == "stereo" else torch.nn.Identity(),
            Mono() if force_channels == "mono" else torch.nn.Identity(),
        )

        log.info("Listing GCS objects at gs://%s/%s …", bucket_name, self.prefix)
        self.wav_keys = _list_wav_keys(bucket_name, self.prefix, max_items)
        log.info("Found %d WAV files.", len(self.wav_keys))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio(self, wav_key: str) -> torch.Tensor:
        """Download a WAV from GCS and return a float32 tensor [C, T]."""
        raw = _read_bytes(self.bucket_name, wav_key)
        audio, in_sr = torchaudio.load(io.BytesIO(raw))
        if in_sr != self.sample_rate:
            audio = T.Resample(in_sr, self.sample_rate)(audio)
        return audio

    def _load_json(self, wav_key: str) -> dict:
        """Download the sibling JSON for *wav_key*.  Returns {} on failure."""
        json_key = wav_key[: -len(".wav")] + ".json"
        try:
            raw = _read_bytes(self.bucket_name, json_key)
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            log.warning("Could not load JSON for %s: %s", wav_key, exc)
            return {}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.wav_keys)

    def __getitem__(self, idx: int):
        wav_key = self.wav_keys[idx]
        try:
            t0 = time.time()
            audio = self._load_audio(wav_key)
            meta = self._load_json(wav_key)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            if self.augs is not None:
                audio = self.augs(audio)
            audio = audio.clamp(-1, 1)
            if self.encoding is not None:
                audio = self.encoding(audio)

            info: dict = {
                "path": f"gs://{self.bucket_name}/{wav_key}",
                "timestamps": (t_start, t_end),
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
                "padding_mask": padding_mask,
                "load_time": time.time() - t0,
            }

            # Merge JSON sidecar — the "text" field becomes "prompt"
            if "text" in meta:
                meta["prompt"] = meta["text"]
            info.update(meta)

            if self.custom_metadata_fn is not None:
                info.update(self.custom_metadata_fn(info, audio))

            if info.get("__reject__", False):
                return self[random.randrange(len(self))]

            return audio, info

        except Exception as exc:
            log.error("Could not load %s: %s", wav_key, exc)
            return self[random.randrange(len(self))]
