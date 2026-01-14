import time
import os
from typing import Any, List

import torch

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from pow.compute.autobs import get_total_GPU_memory
from pow.compute.utils import TimeStats
from pow.models.llama31 import ModelArgs, Transformer
from pow.models.utils import Params, count_params, set_default_dtype
from pow.random_pool_optimized import initialize_model_with_pool
from common.logger import create_logger


logger = create_logger(__name__)


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        devices: List[str],
        output_device: int = None,
        stats: TimeStats = None,
    ):
        super().__init__()
        self.output_device = output_device
        self.stats = stats
        self.module = module

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        with torch.no_grad():
            with self.stats.time_infer():
                # Get device and dtype from model weights
                device = self.module.layers[0].attention.wq.weight.device
                dtype = self.module.layers[0].attention.wq.weight.dtype
                # Convert inputs to match model's device and dtype
                inputs = inputs.to(device=device, dtype=dtype)
                return self.module(inputs, **kwargs)

    @staticmethod
    def build_base_model(
        hash_: str,
        params: Params = Params(),
        seed: int = 42,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> dict:
        """Build model on CPU, load to GPU 0, return GPU state_dict for fast GPU→GPU cloning."""
        torch.manual_seed(seed)
        start_time = time.time()

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            flash=False,
            **(params.__dict__),
        )

        logger.info("Creating base model on CPU...")
        with torch.device("meta"):
            model = Transformer(model_args)
        model.to_empty(device="cpu")
        logger.info(f"Model structure created in {time.time() - start_time:.2f}s")

        model.eval()
        model.requires_grad_(False)

        if dtype == torch.float16:
            model = model.half()
            logger.info("Model converted to float16")
        elif dtype == torch.bfloat16:
            model = model.bfloat16()
        elif dtype == torch.float32:
            model = model.float()

        from pow.random_pool_optimized import initialize_model_with_pool
        initialize_model_with_pool(model, str(hash_), dtype=dtype, pool_fraction=0.05)
        model.recompute_freqs_cis()

        cpu_init_time = time.time() - start_time
        logger.info(f"Base model initialized on CPU in {cpu_init_time:.2f}s | {count_params(model)} params")

        # Load model to GPU 0
        gpu0_start = time.time()
        logger.info("Loading model to GPU 0...")
        model = model.to("cuda:0")
        torch.cuda.synchronize(0)
        logger.info(f"Model loaded to GPU 0 in {time.time() - gpu0_start:.2f}s")

        # Get state_dict from GPU 0 (tensors are on GPU 0)
        gpu_state_dict = model.state_dict()

        total_time = time.time() - start_time
        logger.info(f"Base model ready on GPU 0 in {total_time:.2f}s")

        return {
            "gpu_state_dict": gpu_state_dict,  # Tensors on GPU 0
            "model": model,  # Keep model for Worker 0 to use directly
            "model_args": model_args,
            "dtype": dtype,
        }

    @staticmethod
    def build_from_gpu_state_dict(
        base_model_data: dict,
        stats: TimeStats,
        target_device: str,
    ) -> "ModelWrapper":
        """Clone model from GPU 0 to target GPU using fast GPU→GPU copy."""
        start_time = time.time()

        target_device = torch.device(target_device)
        target_idx = target_device.index

        model_args = base_model_data["model_args"]
        gpu_state_dict = base_model_data["gpu_state_dict"]
        dtype = base_model_data["dtype"]
        base_model = base_model_data.get("model")  # Pre-loaded model on GPU 0

        # Check if target is same as source (GPU 0)
        source_device = next(iter(gpu_state_dict.values())).device
        is_same_device = (target_device == source_device)

        if is_same_device and base_model is not None:
            # Worker 0: Use existing model directly (no copy, no new allocation)
            logger.info(f"Using existing model on {target_device} (no copy needed)...")
            model = base_model
            set_default_dtype(device=target_device, dtype=dtype)
            logger.info(f"Model ready on {target_device} in {time.time() - start_time:.2f}s")
            return ModelWrapper(model, devices=[target_device], stats=stats)

        # Workers 1-N: Clone from GPU 0
        logger.info(f"Cloning model to {target_device} via GPU→GPU copy...")

        # Copy state_dict tensors from GPU 0 to target GPU
        copy_start = time.time()
        target_state_dict = {}
        for name, tensor in gpu_state_dict.items():
            # GPU→GPU copy (uses NVLink if available)
            target_state_dict[name] = tensor.to(target_device, non_blocking=True)

        # Synchronize to ensure all copies complete
        torch.cuda.synchronize(target_idx)
        logger.info(f"State dict copied to {target_device} in {time.time() - copy_start:.2f}s")

        # Create model directly on target device
        load_start = time.time()
        with torch.device(target_device):
            model = Transformer(model_args)

        # Load state_dict
        model.load_state_dict(target_state_dict)
        model.eval()
        model.requires_grad_(False)

        # Free target_state_dict after loading
        del target_state_dict
        torch.cuda.empty_cache()

        # Recompute freqs_cis on target device
        model.recompute_freqs_cis()

        set_default_dtype(device=target_device, dtype=dtype)
        logger.info(f"Model structure loaded in {time.time() - load_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Model cloned to {target_device} in {total_time:.2f}s")

        return ModelWrapper(model, devices=[target_device], stats=stats)

    @staticmethod
    def build(
        hash_: str,
        stats: TimeStats,
        params: Params = Params(),
        seed: int = 42,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        devices: List[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "ModelWrapper":
        with stats.time_model_load():
            devices = [torch.device(device) for device in devices]
            primary_device = devices[0]

            torch.manual_seed(seed)
            start_time = time.time()

            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                flash=False,
                **(params.__dict__),
            )

            logger.info("Creating model...")
            with torch.device("meta"):
                model = Transformer(model_args)
            model.to_empty(device="cpu")
            logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")

            model.eval()
            model.requires_grad_(False)
            
            # Convert model to specified dtype before moving to GPUs
            if dtype == torch.float16:
                model = model.half()
                logger.info("Model converted to float16")
            elif dtype == torch.bfloat16:
                model = model.bfloat16()
                logger.info("Model converted to bfloat16")
            elif dtype == torch.float32:
                model = model.float()
                logger.info("Model converted to float32")

            initialize_model_with_pool(model, str(hash_), dtype=dtype, pool_fraction=0.05)
            # Recompute freqs_cis after model is on CPU and properly initialized
            model.recompute_freqs_cis()

            init_time = time.time() - start_time
            logger.info(f"Model initialized in {init_time:.2f}s | {count_params(model)} params")

            try:
                max_memory = {}
                for device in devices:
                    device_id = device.index
                    max_memory[device_id] = f"{get_total_GPU_memory(device_id)}MB"
                max_memory = get_balanced_memory(model, max_memory=max_memory)
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=["TransformerBlock"],
                    dtype=dtype
                )
                logger.info(f"Inferred device map: {device_map}")
                model = dispatch_model(model, device_map=device_map)
                logger.info("Multi-GPU distribution successful")
            except Exception as e:
                logger.error(f"Multi-GPU distribution failed: {e}")
                logger.error("Falling back to single GPU")
                raise e
            
            model.eval()
            model.requires_grad_(False)

            set_default_dtype(device=primary_device, dtype=dtype)
            
            logger.info("Wrapping model in ModelWrapper")
            model_wrapper = ModelWrapper(model, devices=devices, stats=stats)
            logger.info(f"ModelWrapper created in {stats.model_load_time:.2f}s")

            return model_wrapper


# ============================================================================
# PoC v2 - Qwen Model Support (pretrained weights from HuggingFace)
# ============================================================================

class QwenModelWrapper(torch.nn.Module):
    """Wrapper for Qwen models loaded from HuggingFace."""

    def __init__(
        self,
        module: torch.nn.Module,
        stats: TimeStats = None,
        k_dim: int = 12,
    ):
        super().__init__()
        self.module = module
        self.stats = stats
        self.k_dim = k_dim
        self.vocab_size = module.config.vocab_size
        self.hidden_size = module.config.hidden_size

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass that returns normalized logits for artifacts.

        Args:
            inputs: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Normalized logits of shape (batch, k_dim) - first k dimensions
        """
        with torch.no_grad():
            with self.stats.time_infer():
                # Get device from model (use first non-meta device)
                device = None
                for param in self.module.parameters():
                    if param.device.type != 'meta':
                        device = param.device
                        break
                if device is None:
                    device = torch.device("cuda:0")

                # IMPORTANT: Keep inputs in float16/bfloat16 for FP8 models
                # FP8 models internally handle mixed precision - inputs should NOT be FP8
                # The model will use FP8 for weights but higher precision for activations
                inputs = inputs.to(device=device, dtype=torch.bfloat16)

                logger.info(f"[DEBUG] Input shape: {inputs.shape}, dtype: {inputs.dtype}, device: {inputs.device}")
                logger.info(f"[DEBUG] Input stats: min={inputs.min().item():.4f}, max={inputs.max().item():.4f}, has_nan={torch.isnan(inputs).any().item()}")

                # Qwen expects inputs_embeds for direct embedding input
                outputs = self.module(inputs_embeds=inputs, return_dict=True)

                # Get logits from last token position
                logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

                logger.info(f"[DEBUG] Logits shape: {logits.shape}, dtype: {logits.dtype}")
                logger.info(f"[DEBUG] Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, has_nan={torch.isnan(logits).any().item()}")

                # Normalize logits (L2 norm)
                logits_norm = torch.nn.functional.normalize(logits, p=2, dim=-1)

                logger.info(f"[DEBUG] Normalized logits: has_nan={torch.isnan(logits_norm).any().item()}")

                # Return first k_dim dimensions
                return logits_norm[:, :self.k_dim]

    @staticmethod
    def build_base_model_qwen(
        model_name: str = None,
        dtype: torch.dtype = torch.float16,
        k_dim: int = 12,
    ) -> dict:
        """
        Load Qwen model from HuggingFace with pretrained weights.

        Unlike v1 (build_base_model), this does NOT use block_hash for weights.
        Model weights come from HuggingFace pretrained checkpoint.

        Args:
            model_name: HuggingFace model ID (default from MODEL_NAME env)
            dtype: Model dtype
            k_dim: Number of logit dimensions for artifacts

        Returns:
            Dict with model data for cloning to workers
        """
        from pow.models.qwen_loader import load_qwen_model, MODEL_NAME, K_DIM

        if model_name is None:
            model_name = MODEL_NAME
        if k_dim is None:
            k_dim = K_DIM

        start_time = time.time()
        logger.info(f"Loading Qwen model from HuggingFace: {model_name}")

        # Load model with auto device map (multi-GPU support)
        model = load_qwen_model(
            model_name=model_name,
            dtype=dtype,
            device_map="auto",
        )

        load_time = time.time() - start_time

        # Log full model config for debugging
        logger.info(
            f"Qwen model loaded in {load_time:.2f}s | "
            f"hidden_size={model.config.hidden_size}, "
            f"vocab_size={model.config.vocab_size}, "
            f"num_hidden_layers={model.config.num_hidden_layers}, "
            f"num_attention_heads={model.config.num_attention_heads}"
        )

        # For MoE models, also log MoE-specific info
        if hasattr(model.config, 'num_experts'):
            logger.info(f"MoE config: num_experts={model.config.num_experts}")
        if hasattr(model.config, 'num_experts_per_tok'):
            logger.info(f"MoE config: num_experts_per_tok={model.config.num_experts_per_tok}")

        # Log actual parameter dtype
        first_param = next(model.parameters())
        logger.info(f"Model parameter dtype: {first_param.dtype}, device: {first_param.device}")

        return {
            "model": model,
            "model_name": model_name,
            "dtype": dtype,
            "k_dim": k_dim,
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
        }

    @staticmethod
    def build_from_qwen_model(
        base_model_data: dict,
        stats: TimeStats,
        target_device: str = None,  # Ignored for Qwen - uses device_map="auto"
    ) -> "QwenModelWrapper":
        """
        Create QwenModelWrapper from loaded model.

        For Qwen with device_map="auto", the model is already distributed.
        Workers share the same model instance (no GPU→GPU cloning needed).

        Args:
            base_model_data: Dict from build_base_model_qwen()
            stats: TimeStats for inference timing
            target_device: Ignored (Qwen uses auto device mapping)

        Returns:
            QwenModelWrapper ready for inference
        """
        start_time = time.time()

        model = base_model_data["model"]
        k_dim = base_model_data["k_dim"]

        logger.info(f"Creating QwenModelWrapper (k_dim={k_dim})...")

        wrapper = QwenModelWrapper(
            module=model,
            stats=stats,
            k_dim=k_dim,
        )

        logger.info(f"QwenModelWrapper created in {time.time() - start_time:.2f}s")
        return wrapper
