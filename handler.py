import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional

import runpod

# NOTE: 'requests' imported inside functions to avoid import order issues with CUDA

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pow.compute.gpu_group import create_gpu_groups, GpuGroup, NotEnoughGPUResources
from pow.compute.autobs_v2 import get_batch_size_for_gpu_group
from pow.compute.worker import ParallelWorkerManager, PooledWorkerManager, PooledWorkerManagerV2
from pow.compute.model_init import ModelWrapper, QwenModelWrapper
from pow.compute.orchestrator_client import OrchestratorClient
from pow.compute.gpu_arch import (
    get_gpu_architecture,
    get_architecture_config,
    GPUArchitecture,
    should_use_fallback_mode,
)
from pow.models.utils import Params, get_params_with_fp8

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PoC version: "v1" (custom LLaMA with random weights) or "v2" (Qwen with pretrained weights)
POC_VERSION = os.getenv("POC_VERSION", "v2")

# Maximum job duration: 7 minutes
MAX_JOB_DURATION = 7 * 60

# Warmup polling interval
WARMUP_POLL_INTERVAL = 5  # seconds
WARMUP_MAX_DURATION = 10 * 60  # 10 minutes max warmup time

# Pooled mode settings
POOLED_POLL_INTERVAL = 0.5  # 500ms - how often to poll orchestrator for commands
POOLED_MAX_DURATION = 10 * 60  # 10 minutes max session duration
POOLED_MODEL_LOAD_TIMEOUT = 300  # 5 minutes max for model loading

# Global flag to track if CUDA is broken - prevents repeated failed attempts
_cuda_broken = False


def fetch_warmup_params(callback_url: str, job_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch generation params from MLNode callback URL.
    Returns params dict if available, None if still waiting.
    """
    import requests
    try:
        url = f"{callback_url}/warmup/params"
        response = requests.get(
            url,
            params={"job_id": job_id},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ready":
            return data.get("params")
        return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch warmup params: {e}")
        return None


def send_batch_to_callback(callback_url: str, batch: Dict[str, Any], node_id: int) -> bool:
    """
    Send a generated batch to the callback URL.
    Used in warmup mode where DelegationController is not polling.
    """
    import requests
    try:
        url = f"{callback_url}/generated"
        payload = {
            "public_key": batch.get("public_key", ""),
            "block_hash": batch.get("block_hash", ""),
            "block_height": batch.get("block_height", 0),
            "nonces": batch.get("nonces", []),
            "dist": batch.get("dist", []),
            "node_id": node_id,
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to send batch to callback: {e}")
        return False


def notify_generation_complete(callback_url: str, job_id: str, stats: Dict[str, Any]) -> bool:
    """
    Notify MLNode that generation is complete so it can stop the warmup job.
    """
    import requests
    try:
        url = f"{callback_url}/warmup/complete"
        payload = {
            "job_id": job_id,
            "total_batches": stats.get("total_batches", 0),
            "total_computed": stats.get("total_computed", 0),
            "total_valid": stats.get("total_valid", 0),
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Notified MLNode of generation complete: {stats}")
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to notify generation complete: {e}")
        return False


def warmup_handler(event: Dict[str, Any], callback_url: str, job_id: str):
    """
    Warmup mode handler - reserves worker and waits for params.

    IMPORTANT: Does NOT initialize CUDA early - that causes 5min slowdown!
    Just polls for params, then runs normal generation (which initializes CUDA).

    1. Poll callback_url for generation params
    2. When params received, run normal generation
    """
    logger.info(f"WARMUP MODE: job_id={job_id}, callback_url={callback_url}")
    logger.info("Worker reserved, waiting for generation params (no CUDA init)")

    yield {
        "status": "warmup_ready",
        "job_id": job_id,
    }

    # Poll for params
    warmup_start = time.time()
    poll_count = 0

    while True:
        elapsed = time.time() - warmup_start

        # Check warmup timeout
        if elapsed > WARMUP_MAX_DURATION:
            logger.info(f"WARMUP TIMEOUT: {elapsed:.0f}s exceeded {WARMUP_MAX_DURATION}s limit")
            yield {"status": "warmup_timeout", "elapsed": int(elapsed)}
            return

        poll_count += 1
        if poll_count % 12 == 1:  # Log every minute (12 * 5s = 60s)
            logger.info(f"Warmup poll #{poll_count}: waiting for params... ({int(elapsed)}s)")

        params = fetch_warmup_params(callback_url, job_id)

        if params:
            logger.info(f"Warmup got params after {elapsed:.0f}s, starting generation")
            yield {"status": "warmup_params_received", "elapsed": int(elapsed)}

            # Create event with received params and run generation
            # NOTE: Don't use HTTP callback (send_to_callback=False) because:
            # 1. Runpod cannot access internal Docker URLs (http://api:9100)
            # 2. DelegationController polls our /stream for results instead
            generation_event = {"input": params}
            yield from generation_handler(
                generation_event,
                send_to_callback=False,  # DelegationController polls us via /stream
                callback_url="",
                node_id=0,
                warmup_job_id=job_id,
                warmup_callback_url=callback_url,
            )
            return

        # Yield heartbeat
        yield {
            "status": "warmup_waiting",
            "poll_count": poll_count,
            "elapsed": int(elapsed),
        }

        time.sleep(WARMUP_POLL_INTERVAL)


def generation_handler(
    event: Dict[str, Any],
    send_to_callback: bool = False,
    callback_url: str = "",
    node_id: int = 0,
    warmup_job_id: str = "",
    warmup_callback_url: str = "",
):
    """
    Parallel streaming nonce generator using multiple GPU groups.

    Each GPU group runs as an independent worker process,
    processing different nonce ranges in parallel.

    Stops when:
    1. Client calls POST /cancel/{job_id}
    2. Timeout after 7 minutes (MAX_JOB_DURATION)

    Input from client (ALL REQUIRED):
    {
        "block_hash": str,
        "block_height": int,
        "public_key": str,
        "r_target": float,
        "batch_size": int,  # This is now total batch size, will be distributed
        "start_nonce": int,
        "params": dict,
    }

    Args:
        event: Runpod event with input data
        send_to_callback: If True, send results via HTTP to callback_url
        callback_url: URL to send results to (used in warmup mode)
        node_id: Node ID to include in batches sent to callback
        warmup_job_id: Job ID for warmup mode (to notify completion)
        warmup_callback_url: Callback URL for warmup completion notification

    Yields:
    {
        "nonces": [...],
        "dist": [...],
        "batch_number": int,
        "worker_id": int,
        "elapsed_seconds": int,
        ...
    }
    """
    global _cuda_broken

    # Fast exit if CUDA already known to be broken - don't waste time retrying
    if _cuda_broken:
        logger.error("CUDA already marked as broken - killing worker immediately")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)  # Hard exit - cannot be caught

    aggregated_batch_count = 0
    total_computed = 0
    total_valid = 0

    try:
        input_data = event.get("input", {})

        # Get ALL parameters from client - NO DEFAULTS
        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        r_target = input_data["r_target"]
        client_batch_size = input_data["batch_size"]
        start_nonce = input_data["start_nonce"]
        params_dict = input_data["params"]

        params = Params(**params_dict)

        # Auto-detect GPUs and architecture
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        # Detect GPU architecture for optimizations
        recommended_dtype = torch.float16  # default
        try:
            gpu_caps = get_gpu_architecture(0)
            arch_config = get_architecture_config(0)
            recommended_dtype = gpu_caps.recommended_dtype
            logger.info(
                f"GPU Architecture: {gpu_caps.device_name} "
                f"({gpu_caps.architecture.value}, SM{gpu_caps.compute_capability[0]}{gpu_caps.compute_capability[1]})"
            )
            logger.info(
                f"  Memory: {gpu_caps.total_memory_gb:.1f}GB, "
                f"FP8: {gpu_caps.supports_fp8}, BF16: {gpu_caps.supports_bfloat16}, "
                f"dtype: {recommended_dtype}"
            )

            # Enable FP8 for Blackwell GPUs
            use_fp8 = arch_config.get("use_fp8", False)
            if use_fp8 and gpu_caps.architecture == GPUArchitecture.BLACKWELL:
                logger.info("Blackwell GPU detected - enabling FP8 optimizations")
                params = get_params_with_fp8(params, enable_fp8=True)

            # Check for fallback mode
            if should_use_fallback_mode():
                logger.warning("Fallback mode enabled - using conservative settings")

        except Exception as e:
            logger.warning(f"Could not detect GPU architecture: {e}, using defaults")

        # Create GPU groups based on VRAM requirements
        gpu_groups = create_gpu_groups(params=params)
        n_workers = len(gpu_groups)

        logger.info(f"Created {n_workers} GPU groups for parallel processing:")
        for i, group in enumerate(gpu_groups):
            logger.info(f"  Worker {i}: {group} (VRAM: {group.get_total_vram_gb():.1f}GB)")

        # Calculate batch size per worker
        # Use auto-calculated batch size based on GPU memory
        batch_sizes = []
        for group in gpu_groups:
            bs = get_batch_size_for_gpu_group(group, params)
            batch_sizes.append(bs)
            logger.info(f"  Worker batch size for {group.devices}: {bs}")

        # Use minimum batch size across all groups to ensure consistency
        batch_size_per_worker = min(batch_sizes)
        total_batch_size = batch_size_per_worker * n_workers

        logger.info(f"START: block={block_height}, workers={n_workers}, "
                   f"batch_per_worker={batch_size_per_worker}, total_batch={total_batch_size}, "
                   f"start={start_nonce}")

        # Convert GPU groups to device string lists
        gpu_group_devices = [group.get_device_strings() for group in gpu_groups]

        # Build base model ONCE on GPU 0 (this is the slow part)
        logger.info(f"Building base model on GPU 0 with dtype={recommended_dtype}...")
        base_model_start = time.time()
        base_model_data = ModelWrapper.build_base_model(
            hash_=block_hash,
            params=params,
            max_seq_len=params.seq_len,
            dtype=recommended_dtype,
        )
        logger.info(f"Base model built in {time.time() - base_model_start:.1f}s")

        # Create and start parallel worker manager with pre-built model
        manager = ParallelWorkerManager(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            r_target=r_target,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            start_nonce=start_nonce,
            max_duration=MAX_JOB_DURATION,
            base_model_data=base_model_data,  # Pass pre-built model to all workers
        )

        manager.start()

        # Wait for all workers to initialize models
        if not manager.wait_for_ready(timeout=180):
            logger.error("Workers failed to initialize within timeout")
            yield {
                "error": "Worker initialization timeout",
                "error_type": "TimeoutError",
            }
            manager.stop()
            return

        logger.info("All workers ready, starting streaming")

        start_time = time.time()
        last_result_time = start_time

        # Streaming results from all workers
        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > MAX_JOB_DURATION:
                logger.info(f"TIMEOUT: {elapsed:.0f}s exceeded {MAX_JOB_DURATION}s limit")
                break

            # Check if all workers have stopped
            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            # Get results from workers
            results = manager.get_results(timeout=0.5)

            if not results:
                # No results available, check for stall
                if time.time() - last_result_time > 60:
                    logger.warning("No results for 60s, workers may be stuck")
                continue

            last_result_time = time.time()

            # Yield each result
            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                aggregated_batch_count += 1
                total_computed += result.get("batch_computed", 0)
                total_valid += result.get("batch_valid", 0)

                # Add aggregated stats and fix batch_number for deduplication
                result["aggregated_batch_number"] = aggregated_batch_count
                result["aggregated_total_computed"] = total_computed
                result["aggregated_total_valid"] = total_valid
                result["n_workers"] = n_workers
                # Override batch_number with globally unique value
                # (delegation_controller deduplicates by batch_number)
                result["batch_number"] = aggregated_batch_count

                # Send to callback URL if in warmup mode
                if send_to_callback and callback_url:
                    send_batch_to_callback(callback_url, result, node_id)

                logger.info(f"Batch #{aggregated_batch_count} from worker {result['worker_id']}: "
                           f"{result.get('batch_valid', 0)} valid, elapsed={int(elapsed)}s")

                yield result

        logger.info(f"STOPPED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        manager.stop()

        # Notify MLNode that generation is complete (for warmup mode auto-stop)
        if warmup_job_id and warmup_callback_url:
            notify_generation_complete(warmup_callback_url, warmup_job_id, {
                "total_batches": aggregated_batch_count,
                "total_computed": total_computed,
                "total_valid": total_valid,
            })

        # Free GPU 0 memory - delete base model data after all workers done
        logger.info("Freeing GPU 0 memory...")
        del base_model_data
        torch.cuda.empty_cache()

    except GeneratorExit:
        logger.info(f"CANCELLED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        try:
            manager.stop()
        except:
            pass
        try:
            del base_model_data
            torch.cuda.empty_cache()
        except:
            pass
    except NotEnoughGPUResources as e:
        # CUDA/GPU initialization failed - mark as broken and kill worker
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        logger.error("CUDA broken - killing worker with os._exit(1)")
        yield {
            "error": str(e),
            "error_type": "NotEnoughGPUResources",
            "fatal": True,
        }
        # os._exit(1) cannot be caught - forces immediate process termination
        os._exit(1)

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        try:
            manager.stop()
        except:
            pass
        try:
            del base_model_data
            torch.cuda.empty_cache()
        except:
            pass


def pooled_handler(event: Dict[str, Any]):
    """
    Pooled mode handler - worker managed by external orchestrator.

    This mode supports:
    1. Registration with orchestrator (reports GPU count)
    2. Waiting for block_hash to load model
    3. Dynamic public_key switching without model reload
    4. Range-based nonce distribution
    5. Results tagged with public_key for routing

    Input:
    {
        "mode": "pooled",
        "orchestrator_url": "https://orchestrator.example.com"
    }

    Flow:
    1. Connect to orchestrator, register with GPU info
    2. Wait for block_hash from orchestrator
    3. Load model (slow, ~30-60s)
    4. Notify orchestrator that we're ready
    5. Receive public_key and nonce range, start computing
    6. On switch_job command: flush results, switch public_key, reset nonces
    7. On shutdown command: flush results, cleanup, exit
    """
    global _cuda_broken

    if _cuda_broken:
        logger.error("CUDA already marked as broken - killing worker immediately")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    input_data = event.get("input", {})
    orchestrator_url = input_data.get("orchestrator_url", "")

    if not orchestrator_url:
        yield {"error": "orchestrator_url is required for pooled mode", "error_type": "ValueError"}
        return

    # Initialize orchestrator client (generates UUID for worker_id)
    client = OrchestratorClient(orchestrator_url)
    worker_id = client.worker_id

    logger.info(f"POOLED MODE: worker_id={worker_id}, orchestrator={orchestrator_url}")

    try:
        import torch

        # Step 1: Detect GPUs
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        if gpu_count == 0:
            yield {"error": "No GPUs detected", "error_type": "NotEnoughGPUResources", "fatal": True}
            return

        # Step 2: Register with orchestrator
        yield {"status": "registering", "worker_id": worker_id, "gpu_count": gpu_count}

        reg_response = client.register(gpu_count)
        if reg_response.get("status") == "error":
            yield {"error": "Failed to register with orchestrator", "error_type": "ConnectionError"}
            return

        yield {"status": "registered", "worker_id": worker_id}

        # Step 3: Wait for block_hash
        logger.info("Waiting for block_hash from orchestrator...")
        yield {"status": "waiting_block_hash"}

        config = None
        wait_start = time.time()

        while not client.current_config.block_hash:
            if time.time() - wait_start > POOLED_MAX_DURATION:
                logger.warning("Timeout waiting for block_hash")
                yield {"status": "timeout", "message": "No block_hash received"}
                client.notify_shutdown({"reason": "timeout_waiting_block_hash"})
                return

            config = client.poll_config()
            if config and config.get("type") == "shutdown":
                logger.info("Received shutdown before block_hash")
                yield {"status": "shutdown"}
                return

            time.sleep(1)

        block_hash = client.current_config.block_hash
        params_dict = client.current_config.params
        logger.info(f"Received block_hash: {block_hash[:16]}...")

        yield {"status": "received_block_hash", "block_hash": block_hash[:16]}

        # Step 4: Create GPU groups and load model
        yield {"status": "loading_model"}

        params = Params(**params_dict)
        gpu_groups = create_gpu_groups(params=params)
        n_workers = len(gpu_groups)

        logger.info(f"Created {n_workers} GPU groups")
        for i, group in enumerate(gpu_groups):
            logger.info(f"  Worker {i}: {group} (VRAM: {group.get_total_vram_gb():.1f}GB)")

        # Calculate batch size
        batch_sizes = [get_batch_size_for_gpu_group(group, params) for group in gpu_groups]
        batch_size_per_worker = min(batch_sizes)
        logger.info(f"Batch size per worker: {batch_size_per_worker}")

        # Build base model on GPU 0
        logger.info("Building base model on GPU 0...")
        model_start = time.time()
        base_model_data = ModelWrapper.build_base_model(
            hash_=block_hash,
            params=params,
            max_seq_len=params.seq_len,
        )
        logger.info(f"Base model built in {time.time() - model_start:.1f}s")

        yield {"status": "model_loaded", "load_time": int(time.time() - model_start)}

        # Step 5: Notify orchestrator we're ready and get job config
        ready_response = client.notify_model_loaded()

        if ready_response.get("status") == "error":
            logger.error("Failed to notify ready")
            yield {"error": "Failed to notify ready", "error_type": "ConnectionError"}
            return

        # Get initial job configuration
        public_key = client.current_config.public_key
        range_start = client.current_config.nonce_range_start
        range_end = client.current_config.nonce_range_end
        r_target = client.current_config.r_target
        block_height = client.current_config.block_height

        logger.info(f"Ready to compute: public_key={public_key[:16] if public_key else 'None'}..., range={range_start}-{range_end}")

        yield {"status": "ready", "public_key": public_key[:16] if public_key else None}

        # Step 6: Create pooled worker manager
        gpu_group_devices = [group.get_device_strings() for group in gpu_groups]

        manager = PooledWorkerManager(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            r_target=r_target,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            range_start=range_start,
            range_end=range_end,
            max_duration=POOLED_MAX_DURATION,
            base_model_data=base_model_data,
        )

        manager.start()

        if not manager.wait_for_ready(timeout=POOLED_MODEL_LOAD_TIMEOUT):
            logger.error("Pooled workers failed to initialize")
            yield {"error": "Worker initialization timeout", "error_type": "TimeoutError"}
            manager.stop()
            return

        # Set initial public_key if we have one
        if public_key:
            manager.set_public_key(public_key)

        logger.info("All pooled workers ready, starting compute loop")
        yield {"status": "computing"}

        # Step 7: Main compute loop
        session_start = time.time()
        total_batches_sent = 0
        last_poll_time = time.time()

        while True:
            elapsed = time.time() - session_start

            # Check session timeout
            if elapsed > POOLED_MAX_DURATION:
                logger.info(f"Session timeout after {elapsed:.0f}s")
                break

            # Check if all workers stopped
            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            # Poll orchestrator for commands (every POOLED_POLL_INTERVAL)
            if time.time() - last_poll_time >= POOLED_POLL_INTERVAL:
                last_poll_time = time.time()
                command = client.poll_config()

                if command:
                    cmd_type = command.get("type")

                    if cmd_type == "switch_job":
                        # Flush pending results before switching
                        pending_results = manager.get_all_pending_results()
                        for result in pending_results:
                            client.send_result(result)
                            total_batches_sent += 1

                        # Switch to new public_key
                        new_public_key = command.get("public_key")
                        if new_public_key:
                            manager.switch_public_key(new_public_key)
                            logger.info(f"Switched to public_key: {new_public_key[:16]}...")
                            yield {
                                "status": "switched_public_key",
                                "public_key": new_public_key[:16],
                            }

                    elif cmd_type == "shutdown":
                        logger.info("Received shutdown command")
                        break

                    elif cmd_type == "compute" and command.get("public_key"):
                        # New job assignment (initial or after waiting)
                        new_public_key = command.get("public_key")
                        manager.set_public_key(new_public_key)
                        logger.info(f"Set public_key: {new_public_key[:16]}...")

            # Collect and send results
            results = manager.get_results(timeout=0.1)

            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                # Send to orchestrator (buffered with retry)
                client.send_result(result)
                total_batches_sent += 1

                # Yield for RunPod streaming
                yield result

            # Small sleep to prevent busy loop
            time.sleep(0.01)

        # Step 8: Cleanup
        logger.info(f"Session ended: {total_batches_sent} batches sent, pending={client.get_pending_count()}")

        # Flush any remaining results
        pending_results = manager.get_all_pending_results()
        for result in pending_results:
            client.send_result(result)
            total_batches_sent += 1

        manager.stop()

        # Notify orchestrator of shutdown
        client.notify_shutdown({
            "total_batches_sent": total_batches_sent,
            "session_duration": int(time.time() - session_start),
        })

        yield {
            "status": "shutdown",
            "total_batches_sent": total_batches_sent,
            "session_duration": int(time.time() - session_start),
        }

        # Free GPU memory
        del base_model_data
        torch.cuda.empty_cache()

    except NotEnoughGPUResources as e:
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        yield {"error": str(e), "error_type": "NotEnoughGPUResources", "fatal": True}
        client.notify_shutdown({"error": str(e)})
        os._exit(1)

    except Exception as e:
        logger.error(f"POOLED ERROR: {str(e)}", exc_info=True)
        yield {"error": str(e), "error_type": type(e).__name__}
        try:
            client.notify_shutdown({"error": str(e)})
        except:
            pass

    finally:
        try:
            client.close()
        except:
            pass


def single_handler_v2(event: Dict[str, Any]):
    """
    PoC v2 single mode handler - uses Qwen model with pretrained weights.

    Single job mode for testing without orchestrator.
    Loads Qwen model, runs computation, returns artifacts.

    Input (ALL REQUIRED except max_batches):
    {
        "mode": "single_v2",
        "block_hash": str,        # 64 char hex string
        "block_height": int,
        "public_key": str,        # e.g., "cosmos1..."
        "batch_size": int,        # e.g., 32
        "start_nonce": int,       # e.g., 0
        "max_batches": int,       # optional, default 100
    }

    Returns:
    {
        "artifacts": [{"nonce": int, "vector_b64": str}, ...],
        "encoding": {"dtype": "f16", "k_dim": 12, "endian": "le"},
        "stats": {...}
    }
    """
    global _cuda_broken

    if _cuda_broken:
        logger.error("CUDA already marked as broken - killing worker immediately")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    try:
        import torch
        from pow.compute.compute import ComputeV2
        from pow.data import ArtifactBatch

        input_data = event.get("input", {})

        # Get ALL parameters from client
        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        batch_size = input_data.get("batch_size", 32)
        start_nonce = input_data.get("start_nonce", 0)
        max_batches = input_data.get("max_batches", 100)

        logger.info(f"SINGLE V2 MODE: block_hash={block_hash[:16]}..., public_key={public_key[:16]}...")
        logger.info(f"  batch_size={batch_size}, start_nonce={start_nonce}, max_batches={max_batches}")

        # Step 1: Detect GPUs
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        if gpu_count == 0:
            yield {"error": "No GPUs detected", "error_type": "NotEnoughGPUResources", "fatal": True}
            return

        yield {"status": "loading_model", "poc_version": "v2", "gpu_count": gpu_count}

        # Step 2: Load Qwen model from HuggingFace
        logger.info("Loading Qwen model from HuggingFace...")
        model_start = time.time()

        from pow.models.qwen_loader import MODEL_NAME, K_DIM

        base_model_data = QwenModelWrapper.build_base_model_qwen(
            model_name=MODEL_NAME,
            k_dim=K_DIM,
        )

        model_load_time = time.time() - model_start
        logger.info(f"Qwen model loaded in {model_load_time:.1f}s")

        yield {
            "status": "model_loaded",
            "load_time": int(model_load_time),
            "model_name": MODEL_NAME,
            "k_dim": K_DIM,
            "hidden_size": base_model_data["hidden_size"],
            "vocab_size": base_model_data["vocab_size"],
        }

        # Step 3: Initialize ComputeV2
        seq_len = input_data.get("seq_len", 16)

        compute = ComputeV2(
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            node_id=0,
            base_model_data=base_model_data,
            seq_len=seq_len,
        )

        yield {"status": "computing", "poc_version": "v2"}

        # Step 4: Run computation loop
        start_time = time.time()
        batch_count = 0
        total_artifacts = 0
        all_artifacts = []
        encoding_info = None
        current_nonce = start_nonce

        while batch_count < max_batches:
            elapsed = time.time() - start_time

            # Check timeout (7 minutes)
            if elapsed > MAX_JOB_DURATION:
                logger.info(f"TIMEOUT: {elapsed:.0f}s exceeded {MAX_JOB_DURATION}s limit")
                break

            # Generate nonce batch
            nonces = list(range(current_nonce, current_nonce + batch_size))
            current_nonce += batch_size

            # Process batch through Qwen model
            future = compute(
                nonces=nonces,
                public_key=public_key,
                next_nonces=None,
            )

            # Wait for result
            artifact_batch: ArtifactBatch = future.result()

            batch_count += 1
            batch_artifacts = [a.to_dict() for a in artifact_batch.artifacts]
            all_artifacts.extend(batch_artifacts)
            total_artifacts += len(batch_artifacts)

            if encoding_info is None:
                encoding_info = artifact_batch.encoding.to_dict()

            # Yield progress
            yield {
                "status": "batch_complete",
                "batch_number": batch_count,
                "batch_size": len(batch_artifacts),
                "total_artifacts": total_artifacts,
                "elapsed_seconds": int(elapsed),
            }

            if batch_count % 10 == 0:
                logger.info(f"Batch #{batch_count}: {len(batch_artifacts)} artifacts, total={total_artifacts}")

        # Step 5: Final result
        elapsed = time.time() - start_time
        logger.info(f"COMPLETED: {batch_count} batches, {total_artifacts} artifacts in {elapsed:.1f}s")

        compute.shutdown()

        yield {
            "status": "completed",
            "artifacts": all_artifacts,
            "encoding": encoding_info or {"dtype": "f16", "k_dim": K_DIM, "endian": "le"},
            "stats": {
                "total_batches": batch_count,
                "total_artifacts": total_artifacts,
                "elapsed_seconds": int(elapsed),
                "artifacts_per_second": total_artifacts / elapsed if elapsed > 0 else 0,
            },
        }

        # Free GPU memory
        del base_model_data
        torch.cuda.empty_cache()

    except NotEnoughGPUResources as e:
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        yield {"error": str(e), "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        yield {"error": f"Missing required parameter: {e}", "error_type": "KeyError"}

    except Exception as e:
        logger.error(f"SINGLE V2 ERROR: {str(e)}", exc_info=True)
        yield {"error": str(e), "error_type": type(e).__name__}


def pooled_handler_v2(event: Dict[str, Any]):
    """
    PoC v2 pooled mode handler - uses Qwen model with pretrained weights.

    Key differences from v1:
    - Uses QwenModelWrapper (pretrained weights from HuggingFace)
    - Returns artifacts (base64-encoded vectors) instead of distances
    - No r_target parameter (all artifacts returned)
    - Model loaded via MODEL_NAME env var (cached by RunPod)

    Input:
    {
        "mode": "pooled_v2",
        "orchestrator_url": "https://orchestrator.example.com"
    }
    """
    global _cuda_broken

    if _cuda_broken:
        logger.error("CUDA already marked as broken - killing worker immediately")
        yield {"error": "Worker CUDA is broken", "error_type": "NotEnoughGPUResources", "fatal": True}
        os._exit(1)

    input_data = event.get("input", {})
    orchestrator_url = input_data.get("orchestrator_url", "")

    if not orchestrator_url:
        yield {"error": "orchestrator_url is required for pooled_v2 mode", "error_type": "ValueError"}
        return

    # Initialize orchestrator client
    client = OrchestratorClient(orchestrator_url)
    worker_id = client.worker_id

    logger.info(f"POOLED V2 MODE: worker_id={worker_id}, orchestrator={orchestrator_url}")

    try:
        import torch

        # Step 1: Detect GPUs
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        if gpu_count == 0:
            yield {"error": "No GPUs detected", "error_type": "NotEnoughGPUResources", "fatal": True}
            return

        # Step 2: Register with orchestrator
        yield {"status": "registering", "worker_id": worker_id, "gpu_count": gpu_count, "poc_version": "v2"}

        reg_response = client.register(gpu_count)
        if reg_response.get("status") == "error":
            yield {"error": "Failed to register with orchestrator", "error_type": "ConnectionError"}
            return

        yield {"status": "registered", "worker_id": worker_id}

        # Step 3: Load Qwen model from HuggingFace (pretrained weights)
        # Unlike v1, we don't need to wait for block_hash - model weights are fixed
        yield {"status": "loading_model", "poc_version": "v2"}

        logger.info("Loading Qwen model from HuggingFace...")
        model_start = time.time()

        # Get model config from env or orchestrator
        from pow.models.qwen_loader import MODEL_NAME, K_DIM

        base_model_data = QwenModelWrapper.build_base_model_qwen(
            model_name=MODEL_NAME,
            k_dim=K_DIM,
        )

        model_load_time = time.time() - model_start
        logger.info(f"Qwen model loaded in {model_load_time:.1f}s")

        yield {
            "status": "model_loaded",
            "load_time": int(model_load_time),
            "model_name": MODEL_NAME,
            "k_dim": K_DIM,
            "hidden_size": base_model_data["hidden_size"],
            "vocab_size": base_model_data["vocab_size"],
        }

        # Step 4: Wait for block_hash from orchestrator
        # (only used for input generation, not model weights)
        logger.info("Waiting for block_hash from orchestrator...")
        yield {"status": "waiting_block_hash"}

        wait_start = time.time()

        while not client.current_config.block_hash:
            if time.time() - wait_start > POOLED_MAX_DURATION:
                logger.warning("Timeout waiting for block_hash")
                yield {"status": "timeout", "message": "No block_hash received"}
                client.notify_shutdown({"reason": "timeout_waiting_block_hash"})
                return

            config = client.poll_config()
            if config and config.get("type") == "shutdown":
                logger.info("Received shutdown before block_hash")
                yield {"status": "shutdown"}
                return

            time.sleep(1)

        block_hash = client.current_config.block_hash
        block_height = client.current_config.block_height
        seq_len = client.current_config.params.get("seq_len", 16) if client.current_config.params else 16

        logger.info(f"Received block_hash: {block_hash[:16]}...")
        yield {"status": "received_block_hash", "block_hash": block_hash[:16]}

        # Step 5: Notify orchestrator we're ready
        ready_response = client.notify_model_loaded()

        if ready_response.get("status") == "error":
            logger.error("Failed to notify ready")
            yield {"error": "Failed to notify ready", "error_type": "ConnectionError"}
            return

        # Get initial job configuration
        public_key = client.current_config.public_key
        range_start = client.current_config.nonce_range_start
        range_end = client.current_config.nonce_range_end

        logger.info(f"Ready to compute: public_key={public_key[:16] if public_key else 'None'}..., range={range_start}-{range_end}")
        yield {"status": "ready", "public_key": public_key[:16] if public_key else None}

        # Step 6: Create V2 worker manager
        # For Qwen with device_map="auto", we use a single worker (model spans all GPUs)
        gpu_group_devices = [[f"cuda:{i}" for i in range(gpu_count)]]

        # Calculate batch size based on available memory
        # For 32B model on multi-GPU, start with conservative batch size
        batch_size_per_worker = input_data.get("batch_size", 32)

        manager = PooledWorkerManagerV2(
            block_hash=block_hash,
            block_height=block_height,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            range_start=range_start,
            range_end=range_end,
            base_model_data=base_model_data,
            seq_len=seq_len,
            max_duration=POOLED_MAX_DURATION,
        )

        manager.start()

        if not manager.wait_for_ready(timeout=POOLED_MODEL_LOAD_TIMEOUT):
            logger.error("V2 workers failed to initialize")
            yield {"error": "Worker initialization timeout", "error_type": "TimeoutError"}
            manager.stop()
            return

        # Set initial public_key if we have one
        if public_key:
            manager.set_public_key(public_key)

        logger.info("All V2 workers ready, starting compute loop")
        yield {"status": "computing", "poc_version": "v2"}

        # Step 7: Main compute loop
        session_start = time.time()
        total_batches_sent = 0
        last_poll_time = time.time()

        while True:
            elapsed = time.time() - session_start

            if elapsed > POOLED_MAX_DURATION:
                logger.info(f"Session timeout after {elapsed:.0f}s")
                break

            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            # Poll orchestrator for commands
            if time.time() - last_poll_time >= POOLED_POLL_INTERVAL:
                last_poll_time = time.time()
                command = client.poll_config()

                if command:
                    cmd_type = command.get("type")

                    if cmd_type == "switch_job":
                        # Flush pending results before switching
                        pending_results = manager.get_all_pending_results()
                        for result in pending_results:
                            client.send_result(result)
                            total_batches_sent += 1

                        new_public_key = command.get("public_key")
                        if new_public_key:
                            manager.switch_public_key(new_public_key)
                            logger.info(f"Switched to public_key: {new_public_key[:16]}...")
                            yield {
                                "status": "switched_public_key",
                                "public_key": new_public_key[:16],
                            }

                    elif cmd_type == "shutdown":
                        logger.info("Received shutdown command")
                        break

                    elif cmd_type == "compute" and command.get("public_key"):
                        new_public_key = command.get("public_key")
                        manager.set_public_key(new_public_key)
                        logger.info(f"Set public_key: {new_public_key[:16]}...")

            # Collect and send results (artifacts)
            results = manager.get_results(timeout=0.1)

            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                # Send artifacts to orchestrator
                client.send_result(result)
                total_batches_sent += 1

                # Yield for RunPod streaming
                yield result

            time.sleep(0.01)

        # Step 8: Cleanup
        logger.info(f"V2 session ended: {total_batches_sent} batches sent")

        pending_results = manager.get_all_pending_results()
        for result in pending_results:
            client.send_result(result)
            total_batches_sent += 1

        manager.stop()

        client.notify_shutdown({
            "total_batches_sent": total_batches_sent,
            "session_duration": int(time.time() - session_start),
            "poc_version": "v2",
        })

        yield {
            "status": "shutdown",
            "total_batches_sent": total_batches_sent,
            "session_duration": int(time.time() - session_start),
            "poc_version": "v2",
        }

        # Free GPU memory
        del base_model_data
        torch.cuda.empty_cache()

    except NotEnoughGPUResources as e:
        _cuda_broken = True
        logger.error(f"GPU INIT FAILED: {str(e)}")
        yield {"error": str(e), "error_type": "NotEnoughGPUResources", "fatal": True}
        client.notify_shutdown({"error": str(e)})
        os._exit(1)

    except Exception as e:
        logger.error(f"POOLED V2 ERROR: {str(e)}", exc_info=True)
        yield {"error": str(e), "error_type": type(e).__name__}
        try:
            client.notify_shutdown({"error": str(e)})
        except:
            pass

    finally:
        try:
            client.close()
        except:
            pass


def handler(event: Dict[str, Any]):
    """
    Main handler - dispatches to pooled, warmup, or generation mode.

    Single V2 mode input (PoC v2 with Qwen, for testing):
    {
        "mode": "single_v2",
        "block_hash": str,
        "block_height": int,
        "public_key": str,
        "batch_size": int,        # optional, default 32
        "start_nonce": int,       # optional, default 0
        "max_batches": int,       # optional, default 100
    }

    Pooled V2 mode input (PoC v2 with Qwen, recommended):
    {
        "mode": "pooled_v2",
        "orchestrator_url": "https://orchestrator.example.com"
    }

    Pooled mode input (PoC v1, orchestrator-managed):
    {
        "mode": "pooled",
        "orchestrator_url": "https://orchestrator.example.com"
    }

    Warmup mode input:
    {
        "warmup_mode": True,
        "callback_url": "http://mlnode:9090"
    }

    Generation mode input (normal):
    {
        "block_hash": str,
        "block_height": int,
        ...
    }
    """
    input_data = event.get("input", {})
    mode = input_data.get("mode", "")

    # Check for single v2 mode (PoC v2 with Qwen, for testing)
    if mode == "single_v2":
        yield from single_handler_v2(event)
        return

    # Check for pooled v2 mode (PoC v2 with Qwen)
    if mode == "pooled_v2":
        yield from pooled_handler_v2(event)
        return

    # Check for pooled mode (PoC v1)
    if mode == "pooled":
        # Auto-upgrade to v2 if POC_VERSION is v2
        if POC_VERSION == "v2":
            logger.info("Auto-upgrading pooled mode to v2 (POC_VERSION=v2)")
            yield from pooled_handler_v2(event)
        else:
            yield from pooled_handler(event)
        return

    # Legacy modes
    warmup_mode = input_data.get("warmup_mode", False)

    if warmup_mode:
        callback_url = input_data.get("callback_url", "")
        job_id = event.get("id", "unknown")

        if not callback_url:
            yield {"error": "callback_url is required for warmup mode", "error_type": "ValueError"}
            return

        yield from warmup_handler(event, callback_url, job_id)
    else:
        yield from generation_handler(event)


# Start serverless handler with streaming support
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
