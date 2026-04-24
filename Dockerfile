FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install fashn-vton package from GitHub
RUN pip3 install --no-cache-dir \
    git+https://github.com/fashn-AI/fashn-vton-1.5.git \
    runpod requests

# Download model weights
RUN python3 -c "from fashn_vton.scripts.download_weights import download_weights; download_weights('./weights')"

COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
