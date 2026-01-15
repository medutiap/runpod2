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

# Calculate tensor parallel size (use all GPUs)
TP_SIZE=${TENSOR_PARALLEL_SIZE:-$GPU_COUNT}
echo "Tensor Parallel Size: ${TP_SIZE}"

# vLLM server settings
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}

echo ""
echo "=== Starting vLLM Server ==="
echo "Port: ${VLLM_PORT}"
echo "Host: ${VLLM_HOST}"

# Start vLLM server in background with PoC endpoints enabled
/usr/bin/python3.12 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --enable-poc \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager \
    2>&1 | tee /tmp/vllm.log &

VLLM_PID=$!
echo "vLLM started with PID: ${VLLM_PID}"

# Wait for vLLM to be ready
echo ""
echo "=== Waiting for vLLM to be ready ==="
MAX_WAIT=1200  # 20 minutes for large model
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if vLLM is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died"
        echo "=== vLLM logs ==="
        tail -100 /tmp/vllm.log
        exit 1
    fi

    # Check health endpoint
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM is healthy after ${ELAPSED}s"
        break
    fi

    echo "Waiting for vLLM... (${ELAPSED}s/${MAX_WAIT}s)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM failed to start within ${MAX_WAIT}s"
    echo "=== vLLM logs ==="
    tail -100 /tmp/vllm.log
    exit 1
fi

# Verify PoC endpoints are available
echo ""
echo "=== Verifying PoC endpoints ==="
if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/api/v1/pow/status" >/dev/null 2>&1; then
    echo "PoC endpoints available"
else
    echo "WARNING: PoC endpoints may not be available (this might be normal before first init)"
fi

# Start RunPod handler
echo ""
echo "=== Starting RunPod Handler ==="
exec /usr/bin/python3.12 /app/handler.py
