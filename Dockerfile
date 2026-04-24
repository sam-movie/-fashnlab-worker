FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    diffusers transformers accelerate safetensors \
    runpod requests Pillow

# Download model at build time (baked into image)
RUN python3 -c "from diffusers import AutoPipelineForInpainting; AutoPipelineForInpainting.from_pretrained('fashn-ai/fashn-vton-v1.5')"

COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
