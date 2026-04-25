FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git /app/fashn-vton

WORKDIR /app/fashn-vton
RUN pip3 install --no-cache-dir -e .

RUN pip3 install --no-cache-dir runpod requests huggingface_hub

RUN python3 scripts/download_weights.py --weights-dir /app/weights

RUN python3 -c "from fashn_vton import TryOnPipeline; print('fashn_vton OK')"

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
