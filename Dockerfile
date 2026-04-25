FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repo (we'll use it directly without pip install)
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git /app/fashn-vton

# DIAG: see actual repo structure
RUN ls -la /app/fashn-vton/ && echo "---" && find /app/fashn-vton -name "*.py" -path "*/fashn_vton/*" | head -10

# Install all dependencies manually (from pyproject.toml)
RUN pip3 install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    "safetensors>=0.3.0" \
    "huggingface_hub>=0.20.0" \
    "pillow>=9.0.0" \
    "numpy>=1.21.0" \
    "opencv-python>=4.5.0" \
    "tqdm>=4.65.0" \
    "einops>=0.6.0" \
    "onnxruntime-gpu>=1.14.0" \
    "matplotlib>=3.5.0" \
    "fashn-human-parser>=0.1.1" \
    "runpod" \
    "requests"

# Add fashn-vton source to Python path
ENV PYTHONPATH=/app/fashn-vton/src:/app/fashn-vton:$PYTHONPATH

# Verify import works
RUN python3 -c "from fashn_vton import TryOnPipeline; print('fashn_vton OK')"

# Download model weights
RUN python3 /app/fashn-vton/scripts/download_weights.py --weights-dir /app/weights

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
