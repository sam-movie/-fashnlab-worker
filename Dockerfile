FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone and install fashn-vton
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git /app/fashn-vton && \
    pip3 install --no-cache-dir -e /app/fashn-vton

# Install runpod and requests
RUN pip3 install --no-cache-dir runpod requests

# Download model weights
RUN python3 /app/fashn-vton/scripts/download_weights.py --weights-dir /app/weights

COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
