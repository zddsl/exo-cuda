"""
TinyGrad CUDA Backend for Interweave

Wraps the existing TinygradDynamicShardInferenceEngine to provide
the InterweaveBackend interface for heterogeneous clustering.

Features:
- Native CUDA tensor support via tinygrad
- f16/f32/bf16 dtype support
- Efficient GPU memory management
- JIT compilation for optimized kernels
"""

import asyncio
import logging
import os
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..backend import InterweaveBackend, BackendRegistry
from ..tensor_format import UniversalTensor, DType
from ..shard import InterweaveShard
from ..state import InterweaveState

logger = logging.getLogger(__name__)


@BackendRegistry.register('tinygrad_cuda')
class TinygradCudaBackend(InterweaveBackend):
    """
    TinyGrad CUDA backend using the existing exo inference engine.

    This backend wraps TinygradDynamicShardInferenceEngine to provide
    the Interweave interface while maintaining compatibility with
    the existing exo infrastructure.

    Optimal for:
    - NVIDIA GPUs (V100, A100, RTX series)
    - Dense matrix operations (attention, FFN)
    - Compute-bound layers
    """

    def __init__(
        self,
        device: str = 'CUDA',
        jit: bool = True,
        beam: int = 0,
    ):
        """
        Initialize TinyGrad CUDA backend.

        Args:
            device: TinyGrad device string ('CUDA', 'GPU', 'CPU')
            jit: Enable JIT compilation for kernels
            beam: Beam search width for kernel optimization (0=disabled)
        """
        self.device = device
        self.jit = jit
        self.beam = beam

        self._model = None
        self._shard: Optional[InterweaveShard] = None
        self._loaded_model: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._states: Dict[str, Any] = {}  # request_id -> state

        # Try to import tinygrad and set device
        self._tinygrad_available = self._check_tinygrad()

    @property
    def name(self) -> str:
        return 'tinygrad_cuda'

    @property
    def device_type(self) -> str:
        return 'cuda'

    @property
    def supported_dtypes(self) -> List[str]:
        return ['f32', 'f16', 'bf16']

    @property
    def preferred_dtype(self) -> str:
        return 'f16'

    def _check_tinygrad(self) -> bool:
        """Check if tinygrad with CUDA is available"""
        try:
            from tinygrad import Tensor, Device

            # Try to set CUDA device
            if self.device == 'CUDA':
                if 'CUDA' in Device._devices or os.getenv('GPU', '1') == '1':
                    logger.info("TinyGrad CUDA backend available")
                    return True
                else:
                    logger.warning("CUDA not available, falling back to GPU/CPU")
                    self.device = 'GPU'
                    return True
            return True
        except ImportError as e:
            logger.error(f"TinyGrad not available: {e}")
            return False

    async def load_shard(
        self,
        model_id: str,
        shard: InterweaveShard,
        model_path: Optional[str] = None
    ) -> None:
        """
        Load model weights for the specified shard.

        This uses the existing exo infrastructure to download and load
        model weights, then wraps them for Interweave compatibility.
        """
        if not self._tinygrad_available:
            raise RuntimeError("TinyGrad not available")

        self._shard = shard
        self._loaded_model = model_id

        # Convert InterweaveShard to exo Shard for compatibility
        from exo.inference.shard import Shard as ExoShard
        exo_shard = ExoShard(
            model_id=shard.model_id,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            n_layers=shard.n_layers,
        )

        # Load using existing tinygrad infrastructure
        loop = asyncio.get_running_loop()

        def _load_model():
            from exo.inference.tinygrad.inference import build_transformer
            from pathlib import Path

            # Determine model size
            model_lower = model_id.lower()
            if '1b' in model_lower:
                size = '1B'
            elif '3b' in model_lower:
                size = '3B'
            elif '8b' in model_lower:
                size = '8B'
            elif '70b' in model_lower:
                size = '70B'
            else:
                size = '8B'  # Default

            if model_path:
                return build_transformer(Path(model_path), exo_shard, size)
            else:
                # Use HF cache path
                cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
                # Try to find model in cache
                for entry in os.listdir(cache_dir):
                    if model_id.replace('/', '--') in entry:
                        model_dir = os.path.join(cache_dir, entry, 'snapshots')
                        if os.path.isdir(model_dir):
                            snapshots = os.listdir(model_dir)
                            if snapshots:
                                return build_transformer(
                                    Path(os.path.join(model_dir, snapshots[0])),
                                    exo_shard,
                                    size
                                )
                raise RuntimeError(f"Model {model_id} not found in cache")

        try:
            self._model = await loop.run_in_executor(self._executor, _load_model)
            logger.info(f"Loaded shard {shard} with TinyGrad CUDA")
        except Exception as e:
            logger.error(f"Failed to load shard: {e}")
            raise

    async def forward(
        self,
        input_tensor: UniversalTensor,
        shard: InterweaveShard,
        state: Optional[InterweaveState] = None
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """
        Execute forward pass through TinyGrad CUDA backend.

        Converts universal tensors to tinygrad tensors, runs inference,
        and converts outputs back to universal format.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_shard first.")

        loop = asyncio.get_running_loop()

        def _forward():
            from tinygrad import Tensor

            # Convert input to tinygrad tensor
            input_np = input_tensor.to_numpy()
            x = Tensor(input_np)

            # Get or create state for this request
            request_id = state.request_id if state else "default"

            if request_id not in self._states:
                # Initialize state
                h = self._model.embed(x) if hasattr(self._model, 'embed') else x
                from exo.inference.tinygrad.stateful_model import make_prompt_state
                self._states[request_id] = make_prompt_state(h, self._model)

            model_state = self._states[request_id]

            # Run forward pass
            if hasattr(self._model, 'embed'):
                h = self._model.embed(x)
                out = self._model.forward(h, start_pos=model_state.start, cache=model_state.cache)
                model_state.start += x.shape[1]
            else:
                out = self._model(x)

            # Convert output to numpy
            output_np = out.numpy()

            return output_np, model_state.start

        output_np, new_pos = await loop.run_in_executor(self._executor, _forward)

        # Create output tensor
        output_tensor = UniversalTensor.from_numpy(output_np)

        # Update state
        new_state = state.clone() if state else InterweaveState(request_id="tinygrad_cuda")
        new_state.sequence_position = new_pos

        return output_tensor, new_state

    async def infer_tensor(
        self,
        request_id: str,
        input_data: np.ndarray,
        shard: InterweaveShard,
        inference_state: Optional[dict] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Raw tensor inference matching exo's InferenceEngine interface.

        This method provides compatibility with the existing exo infrastructure.
        """
        input_tensor = UniversalTensor.from_numpy(input_data)

        # Convert inference_state to InterweaveState if needed
        state = None
        if inference_state:
            state = InterweaveState(
                request_id=request_id,
                sequence_position=inference_state.get('position', 0)
            )
        else:
            state = InterweaveState(request_id=request_id)

        output_tensor, new_state = await self.forward(input_tensor, shard, state)

        # Convert back to dict-based state
        new_inference_state = {
            'position': new_state.sequence_position,
        }

        return output_tensor.to_numpy(), new_inference_state

    async def sample(
        self,
        logits: np.ndarray,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 25
    ) -> np.ndarray:
        """
        Sample from logits using tinygrad.

        Args:
            logits: Shape [batch, seq, vocab]
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit

        Returns:
            Sampled token IDs
        """
        loop = asyncio.get_running_loop()

        def _sample():
            from tinygrad import Tensor
            from exo.inference.tinygrad.models.llama import sample_logits

            # Get last position logits
            last_logits = logits[:, -1, :]

            # Sample
            token_ids = sample_logits(
                Tensor(last_logits).flatten(),
                temperature,
                top_k,
                top_p,
                0.0,  # alpha_f
                0.0   # alpha_p
            ).realize().numpy().astype(int)

            return token_ids

        return await loop.run_in_executor(self._executor, _sample)

    async def get_memory_available(self) -> int:
        """Get available GPU VRAM in bytes"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Get first GPU's free memory (in MB)
                free_mb = int(result.stdout.strip().split('\n')[0])
                return free_mb * 1024 * 1024  # Convert to bytes
        except Exception as e:
            logger.warning(f"Could not query GPU memory: {e}")

        # Fallback: estimate based on device
        return 16 * 1024**3  # Assume 16GB default

    async def warmup(self, shard: InterweaveShard) -> None:
        """
        Warmup JIT compilation by running a small forward pass.

        TinyGrad's JIT needs warmup to compile optimized kernels.
        """
        if self._model is None:
            return

        logger.info("Warming up TinyGrad JIT...")

        # Create small test input
        test_input = np.zeros((1, 1), dtype=np.int32)
        test_tensor = UniversalTensor.from_numpy(test_input)
        test_state = InterweaveState(request_id="warmup")

        try:
            await self.forward(test_tensor, shard, test_state)
            logger.info("TinyGrad JIT warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (may be normal): {e}")

    async def cleanup(self) -> None:
        """Release GPU resources"""
        self._model = None
        self._states.clear()

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

        logger.info("TinyGrad CUDA backend cleaned up")


@BackendRegistry.register('tinygrad_opencl')
class TinygradOpenCLBackend(TinygradCudaBackend):
    """
    TinyGrad OpenCL backend for AMD GPUs.

    Uses tinygrad's GPU device which leverages OpenCL on systems
    without CUDA (like Mac Pro with AMD FirePro D500).

    Optimal for:
    - AMD GPUs (FirePro, Radeon series)
    - Intel integrated GPUs with OpenCL
    - Any GPU with OpenCL 1.2+ support
    """

    def __init__(self, jit: bool = True, beam: int = 0):
        # Force GPU device which uses OpenCL
        super().__init__(device='GPU', jit=jit, beam=beam)

    @property
    def name(self) -> str:
        return 'tinygrad_opencl'

    @property
    def device_type(self) -> str:
        return 'opencl'

    def _check_tinygrad(self) -> bool:
        """Check if tinygrad with OpenCL/GPU is available"""
        try:
            from tinygrad import Tensor, Device

            # Check if GPU (OpenCL) device is available
            try:
                gpu = Device['GPU']
                if gpu is not None:
                    logger.info(f"TinyGrad OpenCL backend available: {type(gpu).__name__}")
                    return True
            except Exception as e:
                logger.warning(f"GPU device not available: {e}")
                return False

            return True
        except ImportError as e:
            logger.error(f"TinyGrad not available: {e}")
            return False

    async def get_memory_available(self) -> int:
        """Get available GPU VRAM via OpenCL query or estimate"""
        try:
            from tinygrad import Device
            gpu = Device['GPU']

            # Try to get OpenCL device info
            if hasattr(gpu, 'dev') and hasattr(gpu.dev, 'global_mem_size'):
                return gpu.dev.global_mem_size
        except Exception as e:
            logger.debug(f"Could not query OpenCL memory: {e}")

        # For AMD FirePro D500: 3GB VRAM per GPU
        # Default estimate for unknown GPUs
        return 3 * 1024**3  # 3GB default


@BackendRegistry.register('tinygrad_cpu')
class TinygradCpuBackend(TinygradCudaBackend):
    """
    TinyGrad CPU backend fallback.

    Uses the same implementation as CUDA but runs on CPU.
    Useful for testing or when CUDA is unavailable.
    """

    def __init__(self, jit: bool = True, beam: int = 0):
        super().__init__(device='CPU', jit=jit, beam=beam)

    @property
    def name(self) -> str:
        return 'tinygrad_cpu'

    @property
    def device_type(self) -> str:
        return 'cpu'

    async def get_memory_available(self) -> int:
        """Get available RAM in bytes"""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # Fallback: read from /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            parts = line.split()
                            return int(parts[1]) * 1024
            except:
                pass
        return 32 * 1024**3  # Assume 32GB default


class TinygradInterweaveAdapter:
    """
    Adapter to use existing TinygradDynamicShardInferenceEngine
    with Interweave protocol.

    This provides a bridge between the existing exo inference API
    and the new Interweave backend interface.
    """

    def __init__(self, engine: 'TinygradDynamicShardInferenceEngine'):
        """
        Wrap an existing inference engine.

        Args:
            engine: Existing TinygradDynamicShardInferenceEngine instance
        """
        self.engine = engine

    async def forward(
        self,
        input_tensor: UniversalTensor,
        shard: InterweaveShard,
        state: Optional[InterweaveState] = None
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """
        Execute forward pass using the wrapped engine.
        """
        # Convert InterweaveShard to exo Shard
        from exo.inference.shard import Shard as ExoShard
        exo_shard = ExoShard(
            model_id=shard.model_id,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            n_layers=shard.n_layers,
        )

        # Get request ID and convert state
        request_id = state.request_id if state else "adapter"
        inference_state = None
        if state:
            inference_state = {'position': state.sequence_position}

        # Run inference through existing engine
        input_np = input_tensor.to_numpy()
        output_np, new_state = await self.engine.infer_tensor(
            request_id,
            exo_shard,
            input_np,
            inference_state
        )

        # Convert back to Interweave format
        output_tensor = UniversalTensor.from_numpy(output_np)

        new_interweave_state = state.clone() if state else InterweaveState(request_id=request_id)
        if new_state:
            new_interweave_state.sequence_position = new_state.get('position', 0)

        return output_tensor, new_interweave_state

    @classmethod
    def from_shard_downloader(cls, shard_downloader) -> 'TinygradInterweaveAdapter':
        """
        Create adapter from shard downloader.

        This is the typical way to create the adapter when starting fresh.
        """
        from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
        engine = TinygradDynamicShardInferenceEngine(shard_downloader)
        return cls(engine)
