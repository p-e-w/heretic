# CUDA 12.6 + cuDNN on Ubuntu 24.04
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Speed up Hugging Face model downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.6)
RUN pip install --no-cache-dir --break-system-packages \
    torch \
    --index-url https://download.pytorch.org/whl/cu124

# Install Heretic
ARG HERETIC_VERSION=1.3.0
RUN pip install --no-cache-dir --break-system-packages heretic-llm==${HERETIC_VERSION}

WORKDIR /workspace

# Web UI port (used by heretic-webui)
EXPOSE 7860

ENTRYPOINT ["heretic"]
