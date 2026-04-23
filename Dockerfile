# =============================================================================
# Foundation-1 Training Container
# =============================================================================
# Base: NVIDIA PyTorch 24.01 (CUDA 12.3, PyTorch 2.2, Python 3.10)
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# Build:
#   docker build -t gcr.io/YOUR_PROJECT/f1-training:latest .
#   docker push gcr.io/YOUR_PROJECT/f1-training:latest
#
# Or using Artifact Registry:
#   docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/f1-training/train:latest .
#   docker push us-central1-docker.pkg.dev/YOUR_PROJECT/f1-training/train:latest
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

# System dependencies: gcsfuse for GCS mounting
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    fuse \
    && echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" \
       | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository
COPY . /workspace/

# Install Python dependencies
# Note: torch and torchaudio are already in the base image; we skip them here.
RUN pip install --no-cache-dir \
    google-cloud-storage \
    huggingface_hub \
    safetensors \
    wandb \
    clearml \
    && pip install --no-cache-dir -e ".[train]" \
    || pip install --no-cache-dir \
        aeiou \
        alias-free-torch \
        auraloss \
        einops \
        einops-exts \
        ema-pytorch \
        encodec \
        gradio \
        importlib-resources \
        k-diffusion \
        laion-clap \
        librosa \
        local-attention \
        pandas \
        pedalboard \
        prefigure \
        pretty_midi \
        pytorch_lightning \
        pydub \
        PyWavelets \
        safetensors \
        sentencepiece \
        s3fs \
        soxr \
        tqdm \
        transformers \
        torchmetrics \
        v-diffusion-pytorch \
        vector-quantize-pytorch \
        wandb \
        clearml \
        webdataset \
        x-transformers \
        basic_pitch \
        hffs \
        google-cloud-storage

# Make scripts executable
RUN chmod +x scripts/*.sh

# Default command
CMD ["bash", "scripts/gcp_train_f1_3s.sh"]
