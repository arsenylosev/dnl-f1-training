# DNL Foundation-1 Training Fork
## Fine-Tuning Foundation-1 on 3-Second One-Shot Samples — Google Cloud Platform

This repository is a fork of [RoyalCities/RC-stable-audio-tools](https://github.com/RoyalCities/RC-stable-audio-tools) with additions specific to **Deep Noise Labs** training workflows:

- **GCS-native dataset loader** (`gcs` dataset type) — streams WAV+JSON pairs directly from Google Cloud Storage without requiring a local copy of the data.
- **Foundation-1 3-second model config** — adjusted `sample_size`, `latent_crop_length`, learning rate, and demo prompts for 3-second one-shot instrument samples.
- **GCP launch scripts** — Vertex AI custom job YAML, `gcp_train_f1_3s.sh`, and a Dockerfile for the training container.

---

## Architecture Overview

Foundation-1 is a Diffusion Transformer (DiT) model for text-conditioned audio generation:

```
Text Prompt ──► T5-Base Encoder (109M params, 768-dim, max 128 tokens)
                         │
Timing (seconds_start,   │  Cross-Attention + Global Conditioning
        seconds_total) ──┤
                         ▼
Raw Audio ──► OOBLECK VAE Encoder ──► Latent (64-dim, 2048× downsample)
             (frozen during training)         │
                                              ▼
                                   DiT (24 blocks, 1536-dim)
                                              │
                                              ▼
                                   OOBLECK VAE Decoder ──► Generated Audio
                                   (frozen during training)  Stereo 44.1kHz
```

### 3-Second Latent Math

| Parameter | 20-second (original) | 3-second (this fork) |
|---|---|---|
| `sample_size` | 882,000 | **132,300** |
| Latent tokens | 430 | **64** |
| Pre-encoded `.npy` size per sample | ~220 KB | **~33 KB** |
| Approx. training step speedup | baseline | **~4×** |

---

## Data Format

Your GCS bucket should contain WAV+JSON pairs in the following structure:

```
gs://YOUR_BUCKET/
  training_datasets/
    f1_initial/
      train/
        sample_001.wav
        sample_001.json
        sample_002.wav
        sample_002.json
        ...
      valid/
        ...
      test/
        ...
```

Each JSON file must contain a `text` field with the Foundation-1 conditioning prompt:

```json
{
  "text": "Synth Bass, distorted, layered, dynamic, An analog synth bass dry fat minimalist., C",
  "seconds_start": 0,
  "seconds_total": 3
}
```

The prompt format follows the Foundation-1 tagging schema:
`{Instrument}, {timbre_tags}, {description}, {key}`

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/arsenylosev/dnl-f1-training.git
cd dnl-f1-training
```

### 2. Install dependencies

```bash
pip install .
pip install google-cloud-storage  # required for the 'gcs' dataset type
```

For a minimal smoke test of GCS connectivity, see `scripts/setup_python_env.sh` (`--step smoke`). The Vertex training image does not use editable installs.

### 3. Authenticate with Google Cloud

On a local machine:
```bash
gcloud auth application-default login
```

On Vertex AI / GCE, Application Default Credentials are active automatically.

### 4. Download the Foundation-1 checkpoint

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="RoyalCities/Foundation-1",
    filename="Foundation_1.safetensors",
    local_dir="models/foundation1_3s/",
)
```

---

## Training Pipeline

Production work for this project runs on **Vertex AI** (pre-encoding, then training). The steps below are ordered for that path.

### Step 1 — Pre-encode latents on Vertex (run once)

For production, **pre-encoding is done only on Google Cloud** via the custom job in `scripts/vertex_job_pre_encode.yaml`. The container runs the same `pre_encode.py` as the repo, streams inputs from GCS, and writes `gs://$BUCKET/pre_encoded_3s/`.

1. **Build the training image** (name `f1-trainer` in Artifact Registry):

   ```bash
   gcloud builds submit \
     --tag europe-west4-docker.pkg.dev/winter-quanta-457022-s0/dnl-training/f1-trainer:latest \
     --machine-type e2-highcpu-32 --timeout 3600 --async .
   ```

2. **Configure dataset + job env** — In `stable_audio_tools/configs/dataset_configs/gcs_f1_3s_pre_encode.json`, set `bucket` to your GCS bucket and correct `train`/`valid` prefixes. In the YAML, set `GCS_BUCKET` and, if needed, **inject secrets (e.g. `HUGGINGFACE_TOKEN`, ClearML) at submit time** via a private config copy or by patching the `env` block before `gcloud ai custom-jobs create` (do not commit tokens).

3. **Submit the pre-encode job** — See comments at the top of `scripts/vertex_job_pre_encode.yaml` (`europe-west4`).

*Optional (debug only):* you can run `pre_encode.py` with `gcs_f1_3s_pre_encode.json` and ADC on a GPU machine. That is not the supported production path for this project.

### Step 2 — DiT fine-tune on Vertex (or local GPU)

- **On Vertex:** After `gs://$BUCKET/pre_encoded_3s/` exists, submit `scripts/vertex_job_f1_3s.yaml` in `europe-west4` (same `f1-trainer` image). The job `gsutil rsync`s latents to the VM disk, runs `torchrun`, then syncs checkpoints back to GCS.

- **On a self-managed machine** (GCE, bare metal) with a GPU, use `pre_encoded_f1_3s.json` with a **local** path to latents. Vertex containers **do not** use `gcsfuse` (FUSE is unavailable unprivileged), so a VM-only flow may use `gcsfuse` with `gcp_train_f1_3s.sh` or copy latents to disk.

### Step 3 — (Optional) Fine-tune locally

```bash
python3 train.py \
    --model-config models/foundation1_3s/model_config_3s.json \
    --dataset-config stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json \
    --pretrained-ckpt-path models/foundation1_3s/Foundation_1.safetensors \
    --name foundation1-3s-finetune \
    --batch-size 16 \
    --num-gpus 2 \
    --precision 16-mixed \
    --checkpoint-every 5000 \
    --save-dir /tmp/f1_checkpoints \
    --strategy ddp \
    --num-workers 8
```

Point `pre_encoded_f1_3s.json` at a local directory that mirrors the layout produced by the Vertex pre-encode job.

### Step 4 — Unwrap the checkpoint

```bash
bash scripts/unwrap_checkpoint.sh \
    /tmp/f1_checkpoints/foundation1-3s-finetune/RUN_ID/checkpoints/epoch=0-step=50000.ckpt \
    foundation1_3s_step50k
```

Output: `models/foundation1_3s/foundation1_3s_step50k.safetensors`

---

## Vertex AI launch

**Image:** all jobs use the same Artifact Registry name: `europe-west4-docker.pkg.dev/winter-quanta-457022-s0/dnl-training/f1-trainer:latest` (build with `gcloud builds submit`, not local `docker push` to `gcr.io`).

**Region:** `europe-west4` (see the comments in each `scripts/vertex_job_*.yaml` file).

**Secrets / env at submit time:** set `GCS_BUCKET`, `HUGGINGFACE_TOKEN`, ClearML keys, and the like in a **private** YAML (or patch the `env` block) immediately before you run `gcloud ai custom-jobs create` — the checked-in files use placeholders and empty token fields on purpose.

### Pre-encode, then training

```bash
gcloud config set project winter-quanta-457022-s0

# 1) Pre-encode (after image build; edit YAML for bucket + env first)
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=dnl-f1-pre-encode-$(date +%Y%m%d) \
  --config=scripts/vertex_job_pre_encode.yaml

# 2) DiT fine-tuning (after gs://$BUCKET/pre_encoded_3s/ exists)
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=foundation1-3s-finetune \
  --config=scripts/vertex_job_f1_3s.yaml
```

### Option — VM helper script (not Vertex)

`scripts/gcp_train_f1_3s.sh` is aimed at a **FUSE-capable** GPU host (e.g. self-managed GCE), not Vertex. Prefer the YAML custom jobs for cloud training.

---

## Recommended Hyperparameters for 3-Second Fine-Tuning

| Parameter | Value | Notes |
|---|---|---|
| `sample_size` | 132,300 | 44100 Hz × 3 s |
| `latent_crop_length` | 64 | 132300 / 2048 = 64.6 → 64 |
| `learning_rate` | 2e-5 | Lower than original (5e-5) for fine-tuning |
| `batch_size` | 16 | Per GPU on A100 80GB with pre-encoded latents |
| `precision` | 16-mixed | BF16 also valid on A100/H100 |
| `checkpoint_every` | 5,000 | ~2 h on 2× A100 |
| `num_gpus` | 2 | a3-highgpu-2g on GCP |
| `strategy` | ddp | DDP for 2-GPU, DeepSpeed for 4+ |
| `accum_batches` | 1 | Increase to 2 if OOM |

---

## Dataset Size Recommendations

| Tier | Unique source phrases | With 72× augmentation | Storage |
|---|---|---|---|
| Proof of concept | 500–2,000 | 36k–144k files | 18–71 GB |
| Minimal viable | 2,000–10,000 | 144k–720k files | 71–355 GB |
| **Recommended** | **15,000–75,000** | **1.1M–5.4M files** | **540 GB–2.6 TiB** |
| Full domain | 50,000–200,000 | 3.6M–14.4M files | 1.7–6.9 TiB |

Start at the **Minimal viable** tier to validate the pipeline and config, then scale to **Recommended** once checkpoint quality is confirmed at ~50k steps.

---

## New Files in This Fork

```
stable_audio_tools/data/gcs_dataset.py          ← GCS-native streaming dataset
stable_audio_tools/configs/dataset_configs/
    gcs_f1_3s.json                               ← GCS dataset config (training)
    gcs_f1_3s_pre_encode.json                    ← GCS dataset config (pre-encoding)
    pre_encoded_f1_3s.json                       ← Pre-encoded latents dataset config
models/foundation1_3s/
    model_config_3s.json                         ← Foundation-1 3s model config
scripts/
    gcp_train_f1_3s.sh                           ← All-in-one GCP training script
    vertex_job_f1_3s.yaml                        ← Vertex AI custom job spec
    unwrap_checkpoint.sh                         ← Checkpoint → .safetensors
Dockerfile                                       ← Training container
README_DNL.md                                    ← This file
```

---

## License

This fork inherits the license of the upstream repository. See `LICENSE` for details.

Foundation-1 model weights are subject to the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE).
