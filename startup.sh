#!/bin/bash
set -e

echo "=== RunPod2 vLLM Startup ==="
echo "Model: ${MODEL_NAME}"
echo "K_DIM: ${K_DIM}"
echo "SEQ_LEN: ${SEQ_LEN}"

# Detect number of GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "Detected GPUs: ${GPU_COUNT}"

if [ "$GPU_COUNT" -eq "0" ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

# Use actual GPU count (auto-detect)
TP_SIZE=${GPU_COUNT}
echo "Tensor Parallel Size: ${TP_SIZE}"

# vLLM server settings
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}

echo ""
echo "=== Starting vLLM Server ==="
echo "Port: ${VLLM_PORT}"
echo "Host: ${VLLM_HOST}"

# Start vLLM server in background (PoC endpoints included in v0.9.1-poc-v2-post2)
# --load-format runai_streamer: parallel safetensor loading (faster model load)
/usr/bin/python3.12 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1025 \
    --enforce-eager \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":16}' \
    2>&1 | tee /tmp/vllm.log &

VLLM_PID=$!
echo "vLLM started with PID: ${VLLM_PID}"

# Start RunPod handler immediately (handler.py will wait for vLLM health
# while polling orchestrator for shutdown commands, so the worker can be
# stopped even during model loading)
echo ""
echo "=== Starting RunPod Handler ==="
echo "Handler will wait for vLLM readiness while polling orchestrator"
exec /usr/bin/python3.12 /app/handler.py
