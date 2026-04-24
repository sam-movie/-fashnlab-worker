FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone fashn-vton repo
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git /app/fashn-vton

# Install fashn-vton package
RUN pip3 install --no-cache-dir /app/fashn-vton

# Install additional dependencies
RUN pip3 install --no-cache-dir runpod requests huggingface_hub

# Download model weights
RUN python3 /app/fashn-vton/scripts/download_weights.py --weights-dir /app/weights

COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
