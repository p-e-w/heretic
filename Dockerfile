# CUDA 12.6 + cuDNN on Ubuntu 24.04 (ships with Python 3.12)
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
RUN pip install --no-cache-dir --break-system-packages heretic-llm

WORKDIR /workspace

ENTRYPOINT ["heretic"]
