#!/usr/bin/env python3
"""
RunPod Worker Handler for PoC v2 using vLLM.

Supports two modes:
1. Serverless mode: RunPod jobs with input parameters
2. Warmup mode: Pre-load model, connect to orchestrator, wait for jobs

This handler communicates with vLLM's PoC endpoints via HTTP.
vLLM is started by startup.sh before this handler runs.
"""

import os
import sys
import time
import logging
import requests
from typing import Generator, Any, Dict, List, Optional

import runpod

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)-16s:%(lineno)-4d %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# vLLM server settings
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "MiniMaxAI/MiniMax-M2.7")
K_DIM = int(os.getenv("K_DIM", "12"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "1024"))

# Warmup mode settings
WARMUP_MODE = os.getenv("WARMUP_MODE", "false").lower() == "true"
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")

# HTTP timeouts
REQUEST_TIMEOUT = 660  # 11 minutes for compute


# =============================================================================
# vLLM Client
# =============================================================================

class VLLMPoCClient:
    """Client for vLLM PoC v2 endpoints."""

    def __init__(self, base_url: str = VLLM_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if vLLM is healthy."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict:
        """Get PoC status."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/pow/status",
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "detail": resp.text}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def init_generate(
        self,
        block_hash: str,
        block_height: int,
        public_key: str,
        node_id: int,
        node_count: int,
        batch_size: int = 32,
        callback_url: Optional[str] = None,
        group_id: int = 0,
        n_groups: int = 1,
    ) -> Dict:
        """
        Start continuous artifact generation.

        Results are sent to callback_url if provided.
        Uses node_id/node_count for nonce interleaving in PoC v2.
        """
        payload = {
            "block_hash": block_hash,
            "block_height": block_height,
            "public_key": public_key,
            "node_id": node_id,
            "node_count": node_count,
            "batch_size": batch_size,
            "group_id": group_id,
            "n_groups": n_groups,
            "params": {
                "model": MODEL_NAME,
                "seq_len": SEQ_LEN,
                "k_dim": K_DIM,
            },
        }

        if callback_url:
            payload["url"] = callback_url

        logger.info(f"init_generate: {payload}")

        resp = self.session.post(
            f"{self.base_url}/api/v1/pow/init/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"init_generate failed: {resp.status_code} {resp.text}")

        return resp.json()

    def generate(
        self,
        block_hash: str,
        block_height: int,
        public_key: str,
        node_id: int,
        node_count: int,
        nonces: List[int],
        batch_size: int = 32,
        wait: bool = True,
    ) -> Dict:
        """
        Generate artifacts for specific nonces.

        Args:
            wait: If True, wait for results synchronously
        """
        payload = {
            "block_hash": block_hash,
            "block_height": block_height,
            "public_key": public_key,
            "node_id": node_id,
            "node_count": node_count,
            "nonces": nonces,
            "batch_size": batch_size,
            "wait": wait,
            "params": {
                "model": MODEL_NAME,
                "seq_len": SEQ_LEN,
                "k_dim": K_DIM,
            },
        }

        logger.info(f"generate: nonces={len(nonces)}, wait={wait}")

        resp = self.session.post(
            f"{self.base_url}/api/v1/pow/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"generate failed: {resp.status_code} {resp.text}")

        return resp.json()

    def stop(self) -> Dict:
        """Stop artifact generation."""
        resp = self.session.post(
            f"{self.base_url}/api/v1/pow/stop",
            json={},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"stop failed: {resp.status_code} {resp.text}")
        return resp.json() if resp.status_code == 200 else {"status": "error"}


# Global client
vllm_client = VLLMPoCClient()


# =============================================================================
# Handler Modes
# =============================================================================

def single_handler_v2(input_data: Dict) -> Generator[Dict, None, None]:
    """
    Single mode for testing - generate artifacts for specific nonces.

    Input:
        block_hash: str
        block_height: int
        public_key: str
        batch_size: int (default 32)
        start_nonce: int (default 0)
        max_batches: int (default 10)
    """
    block_hash = input_data["block_hash"]
    block_height = input_data["block_height"]
    public_key = input_data["public_key"]
    batch_size = input_data.get("batch_size", 32)
    start_nonce = input_data.get("start_nonce", 0)
    max_batches = input_data.get("max_batches", 10)

    logger.info(f"SINGLE V2 MODE: block_hash={block_hash[:16]}..., public_key={public_key[:16]}...")
    logger.info(f"  batch_size={batch_size}, start_nonce={start_nonce}, max_batches={max_batches}")

    # Check vLLM health
    if not vllm_client.health_check():
        yield {"error": "vLLM server not healthy", "error_type": "ServerError", "fatal": True}
        return

    yield {"status": "vllm_ready", "poc_version": "v2"}

    # Generate artifacts batch by batch
    all_artifacts = []
    total_nonces = batch_size * max_batches
    start_time = time.time()

    for batch_idx in range(max_batches):
        batch_start = start_nonce + batch_idx * batch_size
        nonces = list(range(batch_start, batch_start + batch_size))

        yield {
            "status": "computing",
            "batch": batch_idx + 1,
            "total_batches": max_batches,
            "nonces_range": f"{nonces[0]}-{nonces[-1]}",
        }

        try:
            result = vllm_client.generate(
                block_hash=block_hash,
                block_height=block_height,
                public_key=public_key,
                node_id=0,
                node_count=1,
                nonces=nonces,
                batch_size=batch_size,
                wait=True,
            )

            if result.get("status") == "completed":
                artifacts = result.get("artifacts", [])
                all_artifacts.extend(artifacts)
                logger.info(f"Batch {batch_idx + 1}: got {len(artifacts)} artifacts")
            else:
                logger.warning(f"Batch {batch_idx + 1} status: {result.get('status')}")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} error: {e}")
            yield {"status": "batch_error", "batch": batch_idx + 1, "error": str(e)}

    elapsed = time.time() - start_time

    # Final result
    encoding = {
        "dtype": "f16",
        "k_dim": K_DIM,
        "endian": "le",
    }

    yield {
        "status": "completed",
        "artifacts": all_artifacts,
        "encoding": encoding,
        "stats": {
            "total_artifacts": len(all_artifacts),
            "elapsed_seconds": int(elapsed),
            "artifacts_per_second": round(len(all_artifacts) / elapsed, 2) if elapsed > 0 else 0,
        },
    }


def pooled_handler_v2(input_data: Dict) -> Generator[Dict, None, None]:
    """
    Pooled mode for orchestrator - warmup and continuous artifact generation.

    Workflow:
    1. Worker starts, vLLM loads model (warmup)
    2. Connect to orchestrator and register as ready
    3. Wait for orchestrator to assign block_hash and public_key (up to 10 minutes)
    4. Start continuous generation with callbacks
    5. Handle job switching and shutdown commands

    Input (minimal):
        orchestrator_url: str - Orchestrator base URL (e.g., https://orch.com/e/runpod|main)

    Orchestrator assigns dynamically:
        - worker_id (auto-generated)
        - block_hash, block_height (when session starts)
        - public_key, node_id, node_count (when job assigned)
    """
    import uuid
    import subprocess

    orchestrator_url = input_data.get("orchestrator_url")
    if not orchestrator_url:
        yield {"error": "orchestrator_url required", "error_type": "ValidationError", "fatal": True}
        return

    # Generate worker ID
    worker_id = str(uuid.uuid4())
    batch_size = input_data.get("batch_size", 32)

    # Detect GPU count
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_count = len([line for line in result.stdout.strip().split("\n") if line])
    except Exception as e:
        logger.warning(f"Could not detect GPU count: {e}, defaulting to 1")
        gpu_count = 1

    logger.info("=" * 60)
    logger.info(f"POOLED V2 MODE (Warmup)")
    logger.info(f"  Worker ID: {worker_id[:8]}...")
    logger.info(f"  Orchestrator: {orchestrator_url}")
    logger.info(f"  GPU Count: {gpu_count}")
    logger.info("=" * 60)

    # Connect to orchestrator FIRST (before waiting for vLLM)
    # This allows us to receive shutdown commands during model loading
    logger.info("Connecting to orchestrator...")
    try:
        response = requests.post(
            f"{orchestrator_url}/api/workers/connect",
            json={
                "worker_id": worker_id,
                "gpu_count": gpu_count,
                "gpu_info": [],
            },
            timeout=30,
        )
        response.raise_for_status()
        connect_data = response.json()
        logger.info(f"Connected: {connect_data}")

        yield {
            "status": "connected",
            "worker_id": worker_id,
            "connect_response": connect_data,
        }

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        yield {"error": f"Connection failed: {e}", "error_type": "ConnectionError", "fatal": True}
        return

    # Wait for vLLM to be ready while polling orchestrator for shutdown
    logger.info("Waiting for vLLM health while polling orchestrator for commands...")
    vllm_wait_start = time.time()
    vllm_max_wait = 1200  # 20 minutes (model loading can take a while)
    vllm_ready = False

    while time.time() - vllm_wait_start < vllm_max_wait:
        # Check for shutdown from orchestrator (every iteration)
        try:
            response = requests.get(
                f"{orchestrator_url}/api/workers/{worker_id}/config",
                timeout=5,
            )
            if response.status_code == 200:
                cmd = response.json()
                if cmd and cmd.get("type") == "shutdown":
                    logger.info("Received shutdown command during vLLM loading - exiting")
                    yield {"status": "shutdown", "reason": "orchestrator_shutdown_during_loading"}
                    return
        except Exception:
            pass  # Orchestrator might be temporarily unavailable

        # Check vLLM health
        if vllm_client.health_check():
            vllm_ready = True
            elapsed = int(time.time() - vllm_wait_start)
            logger.info(f"vLLM is healthy after {elapsed}s")
            break

        time.sleep(2)

    if not vllm_ready:
        elapsed = int(time.time() - vllm_wait_start)
        logger.error(f"vLLM not ready after {elapsed}s")
        yield {"error": f"vLLM not ready after {elapsed}s", "error_type": "ServerError", "fatal": True}
        return

    yield {"status": "vllm_ready", "worker_id": worker_id, "poc_version": "v2"}

    # Poll for config (block_hash) - wait up to 10 minutes
    logger.info("Waiting for block_hash from orchestrator (max 10 minutes)...")
    poll_start = time.time()
    max_wait = 600  # 10 minutes
    poll_interval = 0.5  # 500ms
    block_hash = None
    block_height = None

    while time.time() - poll_start < max_wait:
        try:
            response = requests.get(
                f"{orchestrator_url}/api/workers/{worker_id}/config",
                timeout=10,
            )
            if response.status_code == 200:
                config = response.json()
                if config and config.get("type") == "config":
                    block_hash = config.get("block_hash")
                    block_height = config.get("block_height")
                    if block_hash:
                        logger.info(f"Received block_hash: {block_hash[:16]}...")
                        break

                elif config and config.get("type") == "shutdown":
                    logger.info("Received shutdown command while waiting for config")
                    yield {"status": "shutdown", "reason": "orchestrator_command"}
                    return

        except Exception as e:
            logger.warning(f"Poll config error: {e}")

        time.sleep(poll_interval)

    if not block_hash:
        elapsed = int(time.time() - poll_start)
        logger.error(f"Timeout waiting for block_hash ({elapsed}s)")
        yield {"error": f"Timeout waiting for config ({elapsed}s)", "error_type": "TimeoutError", "fatal": True}
        return

    yield {
        "status": "config_received",
        "block_hash": block_hash[:16] + "...",
        "block_height": block_height,
    }

    # Check for shutdown before proceeding to /ready
    # (session may have been cancelled while we were waiting for config)
    try:
        response = requests.get(
            f"{orchestrator_url}/api/workers/{worker_id}/config",
            timeout=10,
        )
        if response.status_code == 200:
            cmd = response.json()
            if cmd and cmd.get("type") == "shutdown":
                logger.info("Received shutdown command before ready - session cancelled")
                yield {"status": "shutdown", "reason": "session_cancelled_before_ready"}
                return
    except Exception:
        pass

    # Model already loaded (vLLM warmup), notify ready and retry until assigned or shutdown
    logger.info("Notifying orchestrator: ready for job assignment")
    ready_start = time.time()
    ready_max_wait = 600  # 10 minutes

    public_key = None
    node_id = 0
    node_count = 1
    group_id = 0
    n_groups = 1

    while time.time() - ready_start < ready_max_wait:
        try:
            response = requests.post(
                f"{orchestrator_url}/api/workers/{worker_id}/ready",
                json={"gpu_count": gpu_count},
                timeout=30,
            )

            # Worker not found (404) means session was cleared - shutdown
            if response.status_code == 404:
                logger.info("Worker not found (session ended) - shutting down")
                yield {"status": "shutdown", "reason": "worker_not_found"}
                return

            response.raise_for_status()
            ready_data = response.json()

            if ready_data.get("status") == "wait":
                # Node not assigned yet - poll /config for shutdown, then retry
                try:
                    cfg_resp = requests.get(
                        f"{orchestrator_url}/api/workers/{worker_id}/config",
                        timeout=5,
                    )
                    if cfg_resp.status_code == 200:
                        cmd = cfg_resp.json()
                        if cmd and cmd.get("type") == "shutdown":
                            logger.info("Received shutdown while waiting for job")
                            yield {"status": "shutdown", "reason": "orchestrator_command"}
                            return
                except Exception:
                    pass
                time.sleep(2)
                continue

            public_key = ready_data.get("public_key")
            node_id = ready_data.get("node_id", 0)
            node_count = ready_data.get("node_count", 1)
            group_id = ready_data.get("group_id", 0)
            n_groups = ready_data.get("n_groups", 1)

            if not public_key:
                time.sleep(2)
                continue

            logger.info(f"Ready response: {ready_data}")
            break

        except requests.exceptions.HTTPError as e:
            logger.error(f"Ready notification failed: {e}")
            yield {"error": f"Ready notification failed: {e}", "error_type": "ReadyError", "fatal": True}
            return
        except Exception as e:
            logger.warning(f"Ready request error: {e}")
            time.sleep(2)

    if not public_key:
        elapsed = int(time.time() - ready_start)
        logger.error(f"Timeout waiting for job assignment ({elapsed}s)")
        yield {"error": f"Timeout waiting for job assignment ({elapsed}s)", "error_type": "TimeoutError", "fatal": True}
        return

    logger.info(f"Job assigned: pk={public_key[:16]}..., node_id={node_id}/{node_count}, group_id={group_id}, n_groups={n_groups}")
    yield {
        "status": "job_assigned",
        "public_key": public_key[:16] + "...",
        "node_id": node_id,
        "node_count": node_count,
        "group_id": group_id,
        "n_groups": n_groups,
    }

    # Build callback URL (vLLM will append /generated automatically)
    callback_url = f"{orchestrator_url}/callback"

    # Start continuous generation
    logger.info("Starting artifact generation...")
    try:
        result = vllm_client.init_generate(
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            node_id=node_id,
            node_count=node_count,
            batch_size=batch_size,
            callback_url=callback_url,
            group_id=group_id,
            n_groups=n_groups,
        )

        yield {
            "status": "generating",
            "poc_version": "v2",
            "init_result": result,
        }

    except Exception as e:
        logger.error(f"init_generate error: {e}")
        yield {"error": str(e), "error_type": "InitError", "fatal": True}
        return

    # Monitor generation and handle commands
    start_time = time.time()
    last_status_time = 0
    last_heartbeat = 0

    while True:
        elapsed = time.time() - start_time

        # Send heartbeat every 10 seconds
        if elapsed - last_heartbeat >= 10:
            try:
                requests.post(
                    f"{orchestrator_url}/api/workers/{worker_id}/heartbeat",
                    json={"pending_results": 0},
                    timeout=10,
                )
            except Exception:
                pass
            last_heartbeat = elapsed

        # Send status every 30 seconds
        if elapsed - last_status_time >= 30:
            status = vllm_client.get_status()
            yield {
                "status": "progress",
                "elapsed_seconds": int(elapsed),
                "vllm_status": status,
            }
            last_status_time = elapsed

            # Check if generation stopped
            if status.get("status") in ("IDLE", "STOPPED", "ERROR"):
                logger.info(f"Generation ended: {status.get('status')}")
                break

        # Poll for commands (switch_job, shutdown)
        try:
            response = requests.get(
                f"{orchestrator_url}/api/workers/{worker_id}/config",
                timeout=5,
            )
            if response.status_code == 200:
                cmd = response.json()
                if cmd and cmd.get("type") == "shutdown":
                    logger.info("Received shutdown command")
                    vllm_client.stop()
                    break
                elif cmd and cmd.get("type") == "switch_job":
                    new_public_key = cmd.get("public_key")
                    new_node_count = cmd.get("node_count", node_count)
                    logger.info(f"switch_job: stopping current generation, switching to pk={new_public_key[:16] if new_public_key else 'None'}...")

                    # Stop current generation
                    vllm_client.stop()

                    # Restart with new public_key
                    try:
                        result = vllm_client.init_generate(
                            block_hash=block_hash,
                            block_height=block_height,
                            public_key=new_public_key,
                            node_id=node_id,
                            node_count=new_node_count,
                            batch_size=batch_size,
                            callback_url=callback_url,
                            group_id=group_id,
                            n_groups=n_groups,
                        )
                        public_key = new_public_key
                        node_count = new_node_count
                        logger.info(f"switch_job: generation restarted for pk={new_public_key[:16]}..., result={result}")
                        yield {
                            "status": "switched_job",
                            "public_key": new_public_key[:16] + "...",
                        }
                    except Exception as e:
                        logger.error(f"switch_job: failed to restart generation: {e}")
                        yield {"status": "switch_job_error", "error": str(e)}

        except Exception:
            pass

        time.sleep(5)

    # Final status
    final_status = vllm_client.get_status()

    # Notify shutdown
    try:
        requests.post(
            f"{orchestrator_url}/api/workers/{worker_id}/shutdown",
            json={"stats": {}},
            timeout=10,
        )
    except Exception:
        pass

    yield {
        "status": "completed",
        "elapsed_seconds": int(time.time() - start_time),
        "final_status": final_status,
    }


# =============================================================================
# Main Handler
# =============================================================================

def handler(job: Dict) -> Generator[Dict, None, None]:
    """
    Main RunPod handler.

    Modes:
        - single_v2: Generate artifacts for testing (returns artifacts directly)
        - pooled_v2: Continuous generation with orchestrator callbacks
    """
    input_data = job.get("input", {})
    mode = input_data.get("mode", "single_v2")

    logger.info(f"Job received: mode={mode}")

    try:
        if mode == "single_v2":
            yield from single_handler_v2(input_data)

        elif mode == "pooled_v2":
            yield from pooled_handler_v2(input_data)

        else:
            yield {
                "error": f"Unknown mode: {mode}",
                "error_type": "ValidationError",
                "supported_modes": ["single_v2", "pooled_v2"],
            }

    except Exception as e:
        logger.exception(f"Handler error: {e}")
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
            "fatal": True,
        }


# =============================================================================
# Entry Point
# =============================================================================

logger.info("=" * 60)
logger.info("RunPod2 vLLM Handler Starting")
logger.info("=" * 60)
logger.info(f"vLLM URL: {VLLM_BASE_URL}")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"K_DIM: {K_DIM}, SEQ_LEN: {SEQ_LEN}")
logger.info("Starting RunPod serverless worker (vLLM may still be loading)...")

runpod.serverless.start({"handler": handler})
