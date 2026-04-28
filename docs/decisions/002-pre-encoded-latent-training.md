# ADR-002: Pre-encoded latent datasets for Foundation-1 DiT fine-tuning

## Status

Accepted (2026-04-28)

## Context

- **Goal:** After `pre_encode.py` writes VAE latents (`.npy` + sidecar `.json`) under a directory tree, DiT fine-tuning (`train.py`) must consume those tensors **without** running the frozen OOBLECK encoder each step — see `DiffusionCondTrainingWrapper` (`pre_encoded=True` skips `pretransform.encode`, applies scale when needed).
- **Gap:** The docs (`docs/pre_encoding.md`) described `dataset_type: "pre_encoded"` and `stable_audio_tools.data.dataset.PreEncodedDataset`, but **`create_dataloader_from_config` had no `pre_encoded` branch**, so `scripts/vertex_job_f1_3s.yaml` pointed at a dataset JSON that could not load data.
- **Vertex constraints:** Custom Job containers are unprivileged — **no FUSE / gcsfuse** for training data. The supported pattern is **`gsutil rsync`** (or equivalent) to local SSD, same as the pre-encode job.
- **Path consistency:** Training configs must agree on the local directory where rsync lands. This fork standardizes **`/workspace/pre_encoded_3s`** for the Docker/Vertex layout (mirrors job working dir and pre-encode upload prefix `pre_encoded_3s/`).
- **CI regression:** Earlier failures (`FileNotFoundError` for `models/foundation1_3s/model_config_3s.json`) showed that **pytest-only PR checks** did not prove files existed **inside the Docker image**. Guards were added separately (`ci/Dockerfile.context-verify`, contract tests); this ADR records the **data-plane** decision.

## Decision

1. **Implement `PreEncodedLatentsDataset`** in `stable_audio_tools/data/dataset.py`: discover all `.npy` files under configured root paths that have a sibling `.json`, load latents as `float32` `[channels, time]`, normalize metadata (`text` → `prompt`, tensor `padding_mask`), match the metadata produced by `pre_encode.py`.
2. **Wire `dataset_type: "pre_encoded"`** in `create_dataloader_from_config` with the same `collation_fn` as other training loaders; support multiple `datasets[]` entries via `ConcatDataset` when needed.
3. **Ship `stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json`** with `"path": "/workspace/pre_encoded_3s"` as the canonical Vertex/container path.
4. **Set `"training": { "pre_encoded": true, ... }`** in `models/foundation1_3s/model_config_3s.json` so the training wrapper matches the dataset mode (must stay aligned with `train.py` → `create_training_wrapper_from_config`).
5. **Vertex job YAML:** define `DS_PRE=stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json`, verify the file exists before `torchrun`, pass `--dataset-config "${DS_PRE}"` — one definition, no drift between comment and CLI.
6. **GCE shell workflow (`scripts/gcp_train_f1_3s.sh`):** replace gcsfuse-based mounting with **`gsutil rsync` of `gs://…/pre_encoded_3s/` to `/workspace/pre_encoded_3s`**, matching Vertex and the dataset JSON ( privileged fuse no longer required for that script path).

## Alternatives Considered

| Approach | Pros | Cons | Outcome |
|----------|------|------|---------|
| **Stream latents from GCS inside `Dataset.__getitem__`** | No full local copy | High latency / cost per step; complex listing; not how existing audio loaders work | Rejected |
| **gcsfuse mount for training** | Lazy reads | **Fails on Vertex** (no `/dev/fuse`); fragile on VMs | Rejected for Vertex; removed from the default `gcp_train_f1_3s.sh` path |
| **Keep latents only under `/tmp/...` with a dataset JSON committed to `/tmp/...`** | Matches old encode output dir | Diverges from Docker `/workspace` and Vertex | Rejected — one canonical path |
| **Separate dataset JSON per environment** (Vertex vs local) | Flexible paths | Duplication and confusion | Deferred — operators can copy JSON and change `path`; `_comment` documents this |

## Consequences

- DiT fine-tuning jobs **require** pre-encoded output layout compatible with `PreEncodedLatentsDataset` (paired `.npy`/`.json`, metadata keys usable by conditioners).
- **`training.pre_encoded`** in the model JSON is mandatory for correct behavior when training on latents; mismatch would either re-encode incorrectly or shape-mismatch.
- **Disk:** Training VMs need enough SSD for a full rsync of latents (same assumption as Vertex boot disk sizing).
- **Documentation:** `README_DNL.md` remains the operator-facing overview; this ADR is the durable rationale for agents and future refactors.

## References

- `stable_audio_tools/data/dataset.py` — `PreEncodedLatentsDataset`, `create_dataloader_from_config` (`pre_encoded` branch).
- `stable_audio_tools/training/diffusion.py` — `DiffusionCondTrainingWrapper.training_step` (`pre_encoded` vs encode path).
- `stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json` — canonical dataset config.
- `models/foundation1_3s/model_config_3s.json` — `training.pre_encoded`.
- `scripts/vertex_job_f1_3s.yaml` — `DS_PRE`, rsync step, `torchrun` invocation.
- `scripts/gcp_train_f1_3s.sh` — Step 3 `gsutil rsync` to `/workspace/pre_encoded_3s`.
- `.github/workflows/ci.yml` — optional smoke checks for artifact paths inside the built image (related guardrail).

## Verification (repeatable)

```bash
# Contract tests (paths + JSON shape + model flag)
pip install pytest pyyaml
pytest tests/test_vertex_job_contract.py -v

# Import-only sanity: dataset type registers (requires deps / repo install)
python3 -c "
import json
from pathlib import Path
from stable_audio_tools.data.dataset import create_dataloader_from_config
cfg = json.loads(Path('stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json').read_text())
mc = json.loads(Path('models/foundation1_3s/model_config_3s.json').read_text())
assert cfg['dataset_type'] == 'pre_encoded'
assert mc['training'].get('pre_encoded') is True
# Skip full DataLoader if no latents on disk — constructor asserts non-empty pairs
"
```

Full integration requires a populated `/workspace/pre_encoded_3s` (or a copied JSON pointing at a test tree) and GPU training stack; the assertions above validate configuration wiring without a multi-hour run.
