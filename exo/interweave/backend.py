"""
Interweave Backend Interface

Abstract base class for inference backends and registry for backend discovery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Type, Tuple, Any
import logging

from .tensor_format import UniversalTensor, DType
from .shard import InterweaveShard
from .state import InterweaveState

logger = logging.getLogger(__name__)


@dataclass
class BackendCapabilities:
    """Describes a backend's capabilities"""
    name: str
    device_type: str  # 'cuda', 'metal', 'cpu'
    supported_dtypes: List[str]
    preferred_dtype: str
    max_batch_size: int = 32
    max_sequence_length: int = 8192
    supports_kv_cache: bool = True
    supports_streaming: bool = True

    # Performance hints
    memory_available: int = 0  # Bytes
    estimated_flops: float = 0.0  # TFLOPS


class InterweaveBackend(ABC):
    """
    Abstract base class for Interweave inference backends.

    Each backend wraps a specific inference engine (tinygrad, llama.cpp, mlx, etc.)
    and provides a unified interface for:
    - Loading model shards
    - Running forward passes
    - Converting tensors to/from universal format
    - Managing inference state

    Implementations must be registered with BackendRegistry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique backend identifier.

        Examples: 'tinygrad_cuda', 'tinygrad_cpu', 'llama_cpp', 'mlx'
        """
        ...

    @property
    @abstractmethod
    def device_type(self) -> str:
        """
        Device type this backend uses.

        One of: 'cuda', 'metal', 'cpu', 'rocm'
        """
        ...

    @property
    @abstractmethod
    def supported_dtypes(self) -> List[str]:
        """
        List of dtypes this backend can handle natively.

        Examples: ['f32', 'f16', 'bf16'] for GPU backends
                  ['f32', 'f16', 'i8', 'i4'] for quantized CPU backends
        """
        ...

    @property
    @abstractmethod
    def preferred_dtype(self) -> str:
        """
        Optimal dtype for this backend.

        This is the dtype the backend runs most efficiently with.
        """
        ...

    @abstractmethod
    async def load_shard(
        self,
        model_id: str,
        shard: InterweaveShard,
        model_path: Optional[str] = None
    ) -> None:
        """
        Load model weights for the specified shard.

        Args:
            model_id: Model identifier (e.g., 'llama-3.1-70b')
            shard: Shard defining layer range to load
            model_path: Optional path to model weights

        Raises:
            RuntimeError: If model cannot be loaded
        """
        ...

    @abstractmethod
    async def forward(
        self,
        input_tensor: UniversalTensor,
        shard: InterweaveShard,
        state: Optional[InterweaveState] = None
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """
        Execute forward pass through the loaded shard.

        Args:
            input_tensor: Input activations in universal format
            shard: Shard defining which layers to run
            state: Optional inference state (KV-cache, position, etc.)

        Returns:
            Tuple of:
            - Output activations in universal format
            - Updated inference state (or None if no state tracking)

        Raises:
            RuntimeError: If inference fails
        """
        ...

    @abstractmethod
    async def get_memory_available(self) -> int:
        """
        Get available memory in bytes.

        For GPU backends: available VRAM
        For CPU backends: available RAM
        """
        ...

    async def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities"""
        return BackendCapabilities(
            name=self.name,
            device_type=self.device_type,
            supported_dtypes=self.supported_dtypes,
            preferred_dtype=self.preferred_dtype,
            memory_available=await self.get_memory_available(),
        )

    def can_handle_shard(self, shard: InterweaveShard) -> bool:
        """
        Check if this backend can handle a shard.

        Default implementation checks dtype compatibility.
        Override for more sophisticated checks.
        """
        return shard.compatible_with(self)

    async def warmup(self, shard: InterweaveShard) -> None:
        """
        Optional warmup/compilation step.

        Some backends (like tinygrad with JIT) benefit from a warmup
        pass to compile kernels.
        """
        pass

    async def cleanup(self) -> None:
        """
        Release resources held by this backend.

        Called when backend is no longer needed.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, device={self.device_type})"


class BackendRegistry:
    """
    Registry for discovering and instantiating Interweave backends.

    Backends self-register using the @BackendRegistry.register decorator.
    The registry can probe the system to detect available backends.
    """

    _backends: Dict[str, Type[InterweaveBackend]] = {}
    _instances: Dict[str, InterweaveBackend] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a backend class.

        Usage:
            @BackendRegistry.register('tinygrad_cuda')
            class TinygradCudaBackend(InterweaveBackend):
                ...
        """
        def decorator(backend_class: Type[InterweaveBackend]):
            cls._backends[name] = backend_class
            logger.debug(f"Registered backend: {name}")
            return backend_class
        return decorator

    @classmethod
    def get_registered(cls) -> Dict[str, Type[InterweaveBackend]]:
        """Get all registered backend classes"""
        return cls._backends.copy()

    @classmethod
    def detect_available(cls) -> List[str]:
        """
        Probe system and return list of available backend names.

        Checks for:
        - CUDA availability (nvidia-smi, torch.cuda)
        - OpenCL availability (AMD GPUs, Intel GPUs)
        - MLX availability (Apple Silicon check)
        - llama.cpp binary
        - CPU is always available
        """
        available = []

        # Check CUDA
        if cls._check_cuda():
            available.append('tinygrad_cuda')

        # Check OpenCL (AMD/Intel GPUs)
        if cls._check_opencl():
            available.append('tinygrad_opencl')

        # Check MLX (Apple Silicon)
        if cls._check_mlx():
            available.append('mlx')

        # Check llama.cpp
        if cls._check_llamacpp():
            available.append('llama_cpp')

        # CPU backends always available
        available.append('tinygrad_cpu')

        return available

    @classmethod
    def _check_cuda(cls) -> bool:
        """Check if CUDA is available"""
        try:
            # Try nvidia-smi
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            # Try tinygrad
            from tinygrad import Device
            if 'CUDA' in Device.DEFAULT or 'GPU' in Device.DEFAULT:
                return True
        except ImportError:
            pass

        return False

    @classmethod
    def _check_opencl(cls) -> bool:
        """Check if OpenCL GPU is available (AMD, Intel GPUs)"""
        # Skip if CUDA is available (prefer CUDA over OpenCL for NVIDIA)
        if cls._check_cuda():
            return False

        try:
            from tinygrad import Device
            # Check if GPU device is available (uses OpenCL on non-CUDA systems)
            try:
                gpu = Device['GPU']
                if gpu is not None:
                    # Verify it's actually OpenCL (not CUDA/Metal)
                    device_type = type(gpu).__name__
                    if 'CL' in device_type or 'OpenCL' in device_type:
                        return True
                    # Also accept generic GPU on non-Apple platforms
                    import platform
                    if platform.system() == 'Darwin':
                        # On macOS, GPU might be Metal or OpenCL
                        return True
            except Exception:
                pass
        except ImportError:
            pass

        return False

    @classmethod
    def _check_mlx(cls) -> bool:
        """Check if MLX is available (Apple Silicon)"""
        try:
            import mlx.core as mx
            return True
        except ImportError:
            return False

    @classmethod
    def _check_llamacpp(cls) -> bool:
        """Check if llama.cpp is available"""
        import shutil

        # Check for llama-server or llama-cli
        if shutil.which('llama-server') or shutil.which('llama-cli'):
            return True

        # Check common installation paths
        import os
        common_paths = [
            os.path.expanduser('~/llama.cpp/build/bin/llama-server'),
            os.path.expanduser('~/llama.cpp/build-pse-collapse/bin/llama-server'),
            '/usr/local/bin/llama-server',
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return True

        return False

    @classmethod
    def create_backend(cls, name: str, **kwargs) -> InterweaveBackend:
        """
        Create an instance of a registered backend.

        Args:
            name: Backend name (e.g., 'tinygrad_cuda')
            **kwargs: Arguments passed to backend constructor

        Returns:
            Instantiated backend

        Raises:
            ValueError: If backend not registered
        """
        if name not in cls._backends:
            raise ValueError(
                f"Unknown backend: {name}. "
                f"Available: {list(cls._backends.keys())}"
            )

        backend_class = cls._backends[name]
        return backend_class(**kwargs)

    @classmethod
    def get_or_create(cls, name: str, **kwargs) -> InterweaveBackend:
        """
        Get cached backend instance or create new one.

        Backends are cached to avoid redundant initialization.
        """
        if name not in cls._instances:
            cls._instances[name] = cls.create_backend(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def select_optimal(
        cls,
        shard: InterweaveShard,
        available: Optional[List[str]] = None,
        memory_required: int = 0
    ) -> str:
        """
        Select the optimal backend for a shard.

        Selection criteria (in order):
        1. Must be available on this system
        2. Must be in shard's preferred_backends (or 'any')
        3. Must have enough memory
        4. Prefer GPU over CPU
        5. Prefer backend with matching preferred dtype

        Args:
            shard: The shard to run
            available: Available backends (detected if None)
            memory_required: Required memory in bytes

        Returns:
            Selected backend name

        Raises:
            RuntimeError: If no suitable backend found
        """
        if available is None:
            available = cls.detect_available()

        # Filter by shard preferences
        candidates = []
        for backend_name in available:
            if shard.compatible_with_backend_name(backend_name):
                candidates.append(backend_name)

        if not candidates:
            raise RuntimeError(
                f"No available backend compatible with shard preferences: "
                f"{shard.preferred_backends}. Available: {available}"
            )

        # Sort by preference
        def score_backend(name: str) -> tuple:
            # GPU preferred over CPU
            is_gpu = 'cuda' in name or 'mlx' in name or 'opencl' in name
            # Exact match in preferences
            exact_match = name in shard.preferred_backends
            return (is_gpu, exact_match, name)

        candidates.sort(key=score_backend, reverse=True)

        return candidates[0]

    @classmethod
    async def cleanup_all(cls) -> None:
        """Cleanup all cached backend instances"""
        for backend in cls._instances.values():
            await backend.cleanup()
        cls._instances.clear()


# Import and register default backends
def _register_default_backends():
    """Register built-in backends"""
    try:
        from .backends import llamacpp, tinygrad_cuda
    except ImportError as e:
        logger.debug(f"Could not import some backends: {e}")


# Auto-register on module load
_register_default_backends()
