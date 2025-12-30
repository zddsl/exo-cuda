"""
Interweave Protocol - Backend-agnostic distributed inference

Enables heterogeneous compute clusters (CUDA GPUs, Power8 CPUs, Apple Silicon)
to participate in the same inference pipeline through a universal tensor format.

Usage:
    from exo.interweave import InterweaveRouter, UniversalTensor, InterweaveShard

    # Create router with auto-detected backend
    router = InterweaveRouter(node_id="my-node")

    # Add peers
    await router.add_peer("power8-node", "192.168.0.50:50051")

    # Route computation
    output, state, backend = await router.route_forward(shard, input_tensor)
"""

from .tensor_format import UniversalTensor, DType
from .shard import InterweaveShard, create_model_shards, estimate_layer_memory
from .state import InterweaveState, StateManager
from .backend import InterweaveBackend, BackendRegistry, BackendCapabilities

# Router requires gRPC which may not be available on all platforms (e.g., Power8)
try:
    from .router import InterweaveRouter, InterweaveServiceImpl, PeerNode, GRPC_AVAILABLE
except ImportError:
    InterweaveRouter = None
    InterweaveServiceImpl = None
    PeerNode = None
    GRPC_AVAILABLE = False

__all__ = [
    # Core types
    'UniversalTensor',
    'DType',
    'InterweaveShard',
    'InterweaveState',
    'StateManager',
    # Backend interface
    'InterweaveBackend',
    'BackendRegistry',
    'BackendCapabilities',
    # Routing (may be None if gRPC unavailable)
    'InterweaveRouter',
    'InterweaveServiceImpl',
    'PeerNode',
    'GRPC_AVAILABLE',
    # Helpers
    'create_model_shards',
    'estimate_layer_memory',
]

__version__ = '0.1.0'
