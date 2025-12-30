"""
Interweave Shard Definition

Extended shard representation with backend preferences and resource estimates
for heterogeneous compute clustering.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import InterweaveBackend
    from .tensor_format import DType


@dataclass(frozen=True)
class InterweaveShard:
    """
    Extended shard with backend preferences for heterogeneous inference.

    Compatible with existing exo Shard but adds:
    - Backend preference ordering
    - Memory/compute estimates
    - Dtype requirements
    - Layer type classification

    Attributes:
        model_id: Model identifier (e.g., 'llama-3.1-70b')
        start_layer: First layer index in this shard (inclusive)
        end_layer: Last layer index in this shard (inclusive)
        n_layers: Total layers in the full model

        preferred_backends: Priority-ordered list of backend names
        required_dtype: Force specific dtype (None = backend default)
        memory_estimate: Estimated bytes needed for this shard
        compute_intensity: Relative compute weight (1.0 = average)

        is_embedding: True if this shard contains the embedding layer
        is_output: True if this shard contains the output/lm_head layer
        requires_kv_cache: True if attention layers need KV-cache state
    """

    # Base shard info (compatible with existing exo)
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int

    # Interweave extensions
    preferred_backends: Tuple[str, ...] = ('any',)
    required_dtype: Optional[str] = None
    memory_estimate: int = 0
    compute_intensity: float = 1.0

    # Layer classification
    is_embedding: bool = False
    is_output: bool = False
    requires_kv_cache: bool = True

    def __post_init__(self):
        """Validate shard parameters"""
        if self.start_layer < 0:
            raise ValueError(f"start_layer must be >= 0, got {self.start_layer}")
        if self.end_layer >= self.n_layers:
            raise ValueError(f"end_layer must be < n_layers, got {self.end_layer} >= {self.n_layers}")
        if self.start_layer > self.end_layer:
            raise ValueError(f"start_layer must be <= end_layer")

    @property
    def layer_count(self) -> int:
        """Number of layers in this shard"""
        return self.end_layer - self.start_layer + 1

    @property
    def layer_fraction(self) -> float:
        """Fraction of total model layers"""
        return self.layer_count / self.n_layers

    def is_first_layer(self) -> bool:
        """True if this shard starts at layer 0"""
        return self.start_layer == 0

    def is_last_layer(self) -> bool:
        """True if this shard ends at the final layer"""
        return self.end_layer == self.n_layers - 1

    def compatible_with(self, backend: 'InterweaveBackend') -> bool:
        """
        Check if a backend can handle this shard.

        A backend is compatible if:
        1. 'any' is in preferred_backends, OR backend.name is in preferred_backends
        2. If required_dtype is set, backend must support it
        3. Backend has enough memory
        """
        # Check backend preference
        if 'any' not in self.preferred_backends:
            if backend.name not in self.preferred_backends:
                return False

        # Check dtype support
        if self.required_dtype:
            if self.required_dtype not in backend.supported_dtypes:
                return False

        return True

    def compatible_with_backend_name(self, backend_name: str) -> bool:
        """Quick check if backend name matches preferences"""
        if 'any' in self.preferred_backends:
            return True
        return backend_name in self.preferred_backends

    def with_backend_preference(self, *backends: str) -> 'InterweaveShard':
        """Create a new shard with updated backend preferences"""
        return InterweaveShard(
            model_id=self.model_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            n_layers=self.n_layers,
            preferred_backends=backends,
            required_dtype=self.required_dtype,
            memory_estimate=self.memory_estimate,
            compute_intensity=self.compute_intensity,
            is_embedding=self.is_embedding,
            is_output=self.is_output,
            requires_kv_cache=self.requires_kv_cache,
        )

    def split(self, at_layer: int) -> Tuple['InterweaveShard', 'InterweaveShard']:
        """
        Split this shard at a specific layer.

        Returns two shards: [start_layer, at_layer-1] and [at_layer, end_layer]
        """
        if at_layer <= self.start_layer or at_layer > self.end_layer:
            raise ValueError(f"Split layer {at_layer} must be in ({self.start_layer}, {self.end_layer}]")

        first = InterweaveShard(
            model_id=self.model_id,
            start_layer=self.start_layer,
            end_layer=at_layer - 1,
            n_layers=self.n_layers,
            preferred_backends=self.preferred_backends,
            required_dtype=self.required_dtype,
            memory_estimate=int(self.memory_estimate * (at_layer - self.start_layer) / self.layer_count),
            compute_intensity=self.compute_intensity,
            is_embedding=self.is_embedding,
            is_output=False,
            requires_kv_cache=self.requires_kv_cache,
        )

        second = InterweaveShard(
            model_id=self.model_id,
            start_layer=at_layer,
            end_layer=self.end_layer,
            n_layers=self.n_layers,
            preferred_backends=self.preferred_backends,
            required_dtype=self.required_dtype,
            memory_estimate=self.memory_estimate - first.memory_estimate,
            compute_intensity=self.compute_intensity,
            is_embedding=False,
            is_output=self.is_output,
            requires_kv_cache=self.requires_kv_cache,
        )

        return first, second

    @classmethod
    def from_exo_shard(cls, shard: 'Shard') -> 'InterweaveShard':
        """Convert from existing exo Shard to InterweaveShard"""
        return cls(
            model_id=shard.model_id,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            n_layers=shard.n_layers,
            is_embedding=shard.is_first_layer(),
            is_output=shard.is_last_layer(),
        )

    def to_exo_shard(self) -> 'Shard':
        """Convert to existing exo Shard format"""
        from exo.inference.shard import Shard
        return Shard(
            model_id=self.model_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            n_layers=self.n_layers,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/protobuf"""
        return {
            'model_id': self.model_id,
            'start_layer': self.start_layer,
            'end_layer': self.end_layer,
            'n_layers': self.n_layers,
            'preferred_backends': list(self.preferred_backends),
            'required_dtype': self.required_dtype,
            'memory_estimate': self.memory_estimate,
            'compute_intensity': self.compute_intensity,
            'is_embedding': self.is_embedding,
            'is_output': self.is_output,
            'requires_kv_cache': self.requires_kv_cache,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InterweaveShard':
        """Deserialize from dictionary"""
        return cls(
            model_id=data['model_id'],
            start_layer=data['start_layer'],
            end_layer=data['end_layer'],
            n_layers=data['n_layers'],
            preferred_backends=tuple(data.get('preferred_backends', ('any',))),
            required_dtype=data.get('required_dtype'),
            memory_estimate=data.get('memory_estimate', 0),
            compute_intensity=data.get('compute_intensity', 1.0),
            is_embedding=data.get('is_embedding', False),
            is_output=data.get('is_output', False),
            requires_kv_cache=data.get('requires_kv_cache', True),
        )

    def __repr__(self) -> str:
        backends = ','.join(self.preferred_backends)
        return (
            f"InterweaveShard({self.model_id}, layers={self.start_layer}-{self.end_layer}/{self.n_layers}, "
            f"backends=[{backends}])"
        )


# Type alias for compatibility
Shard = InterweaveShard


def estimate_layer_memory(model_id: str, layer_idx: int, dtype: str = 'f16') -> int:
    """
    Estimate memory for a single transformer layer.

    This is a rough estimate based on model architecture.
    Actual memory may vary based on batch size, sequence length, etc.
    """
    # Known model configs (hidden_dim, intermediate_dim, n_heads, head_dim)
    model_configs = {
        'llama-3.1-70b': (8192, 28672, 64, 128),
        'llama-3.1-8b': (4096, 14336, 32, 128),
        'llama-3.2-3b': (3072, 8192, 24, 128),
        'llama-3.2-1b': (2048, 5632, 16, 128),
        'deepseek-v3': (7168, 18432, 56, 128),
    }

    if model_id not in model_configs:
        # Default to 8B-scale estimate
        hidden, intermediate, heads, head_dim = 4096, 14336, 32, 128
    else:
        hidden, intermediate, heads, head_dim = model_configs[model_id]

    # Bytes per element
    bpe = 2 if dtype in ('f16', 'bf16') else 4 if dtype == 'f32' else 1

    # Attention weights: Q, K, V, O projections
    attn_size = 4 * hidden * hidden * bpe

    # FFN weights: gate, up, down projections
    ffn_size = 3 * hidden * intermediate * bpe

    # Layer norms, biases (small)
    other_size = 2 * hidden * bpe

    return attn_size + ffn_size + other_size


def create_model_shards(
    model_id: str,
    n_layers: int,
    splits: List[Tuple[int, str]],
    dtype: str = 'f16'
) -> List[InterweaveShard]:
    """
    Create shards for a model with specified split points and backends.

    Args:
        model_id: Model identifier
        n_layers: Total number of transformer layers
        splits: List of (layer_idx, backend_name) tuples defining split points
                Example: [(0, 'tinygrad_cuda'), (40, 'llama_cpp')]
                Creates: layers 0-39 on CUDA, layers 40-(n_layers-1) on llama.cpp
        dtype: Default dtype for memory estimation

    Returns:
        List of InterweaveShard objects
    """
    if not splits:
        return [InterweaveShard(
            model_id=model_id,
            start_layer=0,
            end_layer=n_layers - 1,
            n_layers=n_layers,
            is_embedding=True,
            is_output=True,
        )]

    shards = []
    splits = sorted(splits, key=lambda x: x[0])

    for i, (start, backend) in enumerate(splits):
        # Determine end layer
        if i + 1 < len(splits):
            end = splits[i + 1][0] - 1
        else:
            end = n_layers - 1

        # Estimate memory for this shard
        memory = sum(estimate_layer_memory(model_id, l, dtype) for l in range(start, end + 1))

        shard = InterweaveShard(
            model_id=model_id,
            start_layer=start,
            end_layer=end,
            n_layers=n_layers,
            preferred_backends=(backend,),
            memory_estimate=memory,
            is_embedding=(start == 0),
            is_output=(end == n_layers - 1),
        )
        shards.append(shard)

    return shards
