# Use gonka's custom vLLM image with PoC v2 endpoints support
# Note: -poc-v2-post2 includes group_id/n_groups support for nonce distribution
FROM ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-post3

# Set working directory
WORKDIR /app

# Install RunPod SDK and HTTP client
# Use vLLM's Python directly
RUN /usr/bin/python3.12 -m pip install --no-cache-dir runpod requests httpx

# Copy handler code
COPY handler.py /app/
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# PoC v2 settings
ENV POC_VERSION=v2
ENV MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
ENV K_DIM=12
ENV SEQ_LEN=1024

# vLLM settings
ENV VLLM_PORT=8000
ENV VLLM_HOST=127.0.0.1
# Increase PoC request timeout (default 10000ms = 10s)
ENV VLLM_RPC_TIMEOUT=120000

# Tensor parallel size — auto-detected from GPU count in startup.sh
# Note: TP=8 causes issues with Qwen3-235B FP8 (MoE gate/up weights not divisible by FP8 block_n)
# TP=2 and TP=4 work fine

# NCCL settings for multi-GPU communication
# Disable NVLS (NVLink SHARP) - causes "invalid argument" errors in containers
ENV NCCL_NVLS_ENABLE=0
# Disable P2P if GPUs can't communicate directly
ENV NCCL_P2P_DISABLE=1
# Disable InfiniBand (not available in most cloud setups)
ENV NCCL_IB_DISABLE=1
# Use shared memory for communication
ENV NCCL_SHM_DISABLE=0
# Debug level (set to INFO for troubleshooting)
ENV NCCL_DEBUG=WARN

# HuggingFace cache location (RunPod caches models here)
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub
ENV VLLM_USE_V1=0

# Clear any default entrypoint from vLLM image
ENTRYPOINT []

# Run startup script which starts vLLM and handler
CMD ["/app/startup.sh"]
