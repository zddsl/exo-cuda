"""
Interweave Router

Routes computation to optimal backends across the heterogeneous cluster.
Handles backend selection, load balancing, and fallback strategies.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

from .backend import BackendRegistry, InterweaveBackend, BackendCapabilities
from .tensor_format import UniversalTensor
from .shard import InterweaveShard
from .state import InterweaveState

# gRPC is optional (may not build on all platforms like Power8 ppc64le)
try:
    import grpc
    from .proto import (
        ForwardRequest,
        ForwardResponse,
        RouteQueryRequest,
        RouteQueryResponse,
        RouteCandidate,
        InterweaveServiceStub,
        UniversalTensorProto,
        InterweaveShardProto,
        InterweaveStateProto,
    )
    GRPC_AVAILABLE = True
except ImportError:
    grpc = None
    GRPC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PeerNode:
    """Represents a remote peer in the Interweave cluster"""
    node_id: str
    address: str  # host:port
    capabilities: BackendCapabilities
    last_seen: float = 0.0
    latency_ms: float = 0.0
    channel: Optional[grpc.aio.Channel] = None
    stub: Optional[InterweaveServiceStub] = None

    async def connect(self) -> bool:
        """Establish gRPC connection to peer"""
        try:
            self.channel = grpc.aio.insecure_channel(self.address)
            self.stub = InterweaveServiceStub(self.channel)
            self.last_seen = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to peer {self.node_id}: {e}")
            return False

    async def disconnect(self) -> None:
        """Close gRPC connection"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

    @property
    def is_connected(self) -> bool:
        return self.stub is not None


@dataclass
class RouteScore:
    """Score for a potential route"""
    node_id: str
    backend_name: str
    is_local: bool
    score: float
    latency_estimate_ms: float
    memory_available: int

    def __lt__(self, other: 'RouteScore') -> bool:
        return self.score > other.score  # Higher score = better


class InterweaveRouter:
    """
    Routes computation to optimal backends across the cluster.

    The router:
    1. Maintains connections to peer nodes
    2. Tracks backend capabilities across the cluster
    3. Selects optimal backend for each shard
    4. Handles failover if primary backend fails
    """

    def __init__(
        self,
        node_id: str,
        local_backend: Optional[InterweaveBackend] = None,
        prefer_local: bool = True,
        max_peers: int = 16,
    ):
        """
        Initialize router.

        Args:
            node_id: This node's identifier
            local_backend: Local inference backend (auto-detect if None)
            prefer_local: Prefer local execution when possible
            max_peers: Maximum number of peer connections
        """
        self.node_id = node_id
        self.local_backend = local_backend
        self.prefer_local = prefer_local
        self.max_peers = max_peers

        self.peers: Dict[str, PeerNode] = {}
        self._lock = asyncio.Lock()

        # Auto-detect local backend if not provided
        if self.local_backend is None:
            self._auto_detect_backend()

    def _auto_detect_backend(self) -> None:
        """Detect and initialize the best local backend"""
        available = BackendRegistry.detect_available()
        logger.info(f"Available backends: {available}")

        # Prefer CUDA > MLX > CPU
        for backend_name in ['tinygrad_cuda', 'mlx', 'llama_cpp', 'tinygrad_cpu']:
            if backend_name in available:
                try:
                    self.local_backend = BackendRegistry.create_backend(backend_name)
                    logger.info(f"Auto-selected local backend: {backend_name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to create {backend_name} backend: {e}")

        logger.warning("No local backend available")

    async def add_peer(
        self,
        node_id: str,
        address: str,
        capabilities: Optional[BackendCapabilities] = None
    ) -> bool:
        """
        Add a peer node to the routing table.

        Args:
            node_id: Peer's identifier
            address: gRPC endpoint (host:port)
            capabilities: Pre-fetched capabilities (will probe if None)

        Returns:
            True if peer was added successfully
        """
        async with self._lock:
            if len(self.peers) >= self.max_peers:
                logger.warning(f"Max peers ({self.max_peers}) reached")
                return False

            peer = PeerNode(
                node_id=node_id,
                address=address,
                capabilities=capabilities or BackendCapabilities(
                    name='unknown',
                    device_type='unknown',
                    supported_dtypes=[],
                    preferred_dtype='f32',
                ),
            )

            if await peer.connect():
                self.peers[node_id] = peer
                logger.info(f"Added peer {node_id} at {address}")
                return True

            return False

    async def remove_peer(self, node_id: str) -> None:
        """Remove a peer from the routing table"""
        async with self._lock:
            if node_id in self.peers:
                await self.peers[node_id].disconnect()
                del self.peers[node_id]
                logger.info(f"Removed peer {node_id}")

    async def _score_route(
        self,
        shard: InterweaveShard,
        node_id: str,
        backend_name: str,
        is_local: bool,
        latency_ms: float,
        memory_available: int,
    ) -> RouteScore:
        """
        Score a potential route.

        Higher score = better choice.
        """
        score = 0.0

        # Backend preference match (0-40 points)
        if 'any' in shard.preferred_backends:
            score += 20
        elif backend_name in shard.preferred_backends:
            idx = shard.preferred_backends.index(backend_name)
            score += 40 - (idx * 10)  # First preference gets 40, second 30, etc.

        # Device type preference (0-30 points)
        is_gpu = 'cuda' in backend_name or 'mlx' in backend_name
        if is_gpu:
            score += 30  # GPU preferred for compute-intensive shards

        # Local execution bonus (0-15 points)
        if is_local and self.prefer_local:
            score += 15

        # Latency penalty (0-10 points, lower latency = higher score)
        if latency_ms < 10:
            score += 10
        elif latency_ms < 50:
            score += 7
        elif latency_ms < 100:
            score += 4
        else:
            score += 1

        # Memory availability (0-5 points)
        if shard.memory_estimate > 0 and memory_available > 0:
            if memory_available >= shard.memory_estimate * 1.5:
                score += 5
            elif memory_available >= shard.memory_estimate:
                score += 3

        return RouteScore(
            node_id=node_id,
            backend_name=backend_name,
            is_local=is_local,
            score=score,
            latency_estimate_ms=latency_ms,
            memory_available=memory_available,
        )

    async def find_candidates(
        self,
        shard: InterweaveShard,
        memory_required: int = 0,
    ) -> List[RouteScore]:
        """
        Find all candidate routes for a shard.

        Returns list of scored routes, sorted by preference (best first).
        """
        candidates = []

        # Check local backend
        if self.local_backend:
            if shard.compatible_with(self.local_backend):
                memory_avail = await self.local_backend.get_memory_available()
                if memory_required == 0 or memory_avail >= memory_required:
                    score = await self._score_route(
                        shard=shard,
                        node_id=self.node_id,
                        backend_name=self.local_backend.name,
                        is_local=True,
                        latency_ms=0,
                        memory_available=memory_avail,
                    )
                    candidates.append(score)

        # Check peer backends
        async with self._lock:
            for peer_id, peer in self.peers.items():
                if not peer.is_connected:
                    continue

                # Check if peer's backend is compatible
                if shard.compatible_with_backend_name(peer.capabilities.name):
                    memory_avail = peer.capabilities.memory_available
                    if memory_required == 0 or memory_avail >= memory_required:
                        score = await self._score_route(
                            shard=shard,
                            node_id=peer_id,
                            backend_name=peer.capabilities.name,
                            is_local=False,
                            latency_ms=peer.latency_ms,
                            memory_available=memory_avail,
                        )
                        candidates.append(score)

        # Sort by score (highest first)
        candidates.sort()

        return candidates

    async def route_forward(
        self,
        shard: InterweaveShard,
        input_tensor: UniversalTensor,
        state: Optional[InterweaveState] = None,
    ) -> Tuple[UniversalTensor, Optional[InterweaveState], str]:
        """
        Route computation to the best available backend.

        Args:
            shard: Shard to execute
            input_tensor: Input in universal format
            state: Optional KV-cache state

        Returns:
            Tuple of (output_tensor, updated_state, backend_used)

        Raises:
            RuntimeError: If no suitable backend found
        """
        candidates = await self.find_candidates(shard, shard.memory_estimate)

        if not candidates:
            raise RuntimeError(
                f"No available backend for shard {shard}. "
                f"Preferences: {shard.preferred_backends}"
            )

        # Try candidates in order
        last_error = None
        for candidate in candidates:
            try:
                if candidate.is_local:
                    # Local execution
                    output, new_state = await self.local_backend.forward(
                        input_tensor, shard, state
                    )
                    return output, new_state, self.local_backend.name
                else:
                    # Remote execution
                    output, new_state = await self._remote_forward(
                        candidate.node_id,
                        shard,
                        input_tensor,
                        state,
                    )
                    return output, new_state, candidate.backend_name

            except Exception as e:
                logger.warning(
                    f"Failed to execute on {candidate.node_id}/{candidate.backend_name}: {e}"
                )
                last_error = e
                continue

        raise RuntimeError(f"All backends failed. Last error: {last_error}")

    async def _remote_forward(
        self,
        node_id: str,
        shard: InterweaveShard,
        input_tensor: UniversalTensor,
        state: Optional[InterweaveState] = None,
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """
        Execute forward pass on a remote peer.
        """
        peer = self.peers.get(node_id)
        if not peer or not peer.is_connected:
            raise RuntimeError(f"Peer {node_id} not connected")

        # Convert to proto format
        request = ForwardRequest(
            shard=self._shard_to_proto(shard),
            input=self._tensor_to_proto(input_tensor),
            request_id=state.request_id if state else "remote",
        )

        if state:
            request.state.CopyFrom(self._state_to_proto(state))

        # Execute RPC
        try:
            response = await peer.stub.Forward(request, timeout=300)

            # Convert response back
            output_tensor = self._tensor_from_proto(response.output)
            new_state = None
            if response.HasField('updated_state'):
                new_state = self._state_from_proto(response.updated_state)

            # Update peer latency
            peer.latency_ms = response.inference_time_ms

            return output_tensor, new_state

        except grpc.RpcError as e:
            logger.error(f"gRPC error from {node_id}: {e}")
            raise RuntimeError(f"Remote forward failed: {e}")

    # ========== Proto Conversion Helpers ==========

    def _tensor_to_proto(self, tensor: UniversalTensor) -> UniversalTensorProto:
        """Convert UniversalTensor to proto format"""
        return UniversalTensorProto(
            data=tensor.data,
            shape=list(tensor.shape),
            dtype=tensor.dtype.value,
            layout=tensor.layout,
            device_hint=tensor.device_hint,
            scale=tensor.scale,
            zero_point=tensor.zero_point,
        )

    def _tensor_from_proto(self, proto: UniversalTensorProto) -> UniversalTensor:
        """Convert proto to UniversalTensor"""
        from .tensor_format import DType
        return UniversalTensor(
            data=proto.data,
            shape=tuple(proto.shape),
            dtype=DType(proto.dtype),
            layout=proto.layout,
            device_hint=proto.device_hint,
            scale=proto.scale if proto.HasField('scale') else None,
            zero_point=proto.zero_point if proto.HasField('zero_point') else None,
        )

    def _shard_to_proto(self, shard: InterweaveShard) -> InterweaveShardProto:
        """Convert InterweaveShard to proto format"""
        return InterweaveShardProto(
            model_id=shard.model_id,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            n_layers=shard.n_layers,
            preferred_backends=list(shard.preferred_backends),
            required_dtype=shard.required_dtype or '',
            memory_estimate=shard.memory_estimate,
            compute_intensity=shard.compute_intensity,
            is_embedding=shard.is_embedding,
            is_output=shard.is_output,
            requires_kv_cache=shard.requires_kv_cache,
        )

    def _state_to_proto(self, state: InterweaveState) -> InterweaveStateProto:
        """Convert InterweaveState to proto format"""
        proto = InterweaveStateProto(
            request_id=state.request_id,
            sequence_position=state.sequence_position,
        )

        # Add KV-cache entries
        from .proto import KVCacheEntry
        for layer_idx, (k, v) in state.kv_cache.items():
            proto.kv_cache[layer_idx].CopyFrom(KVCacheEntry(
                key=self._tensor_to_proto(k),
                value=self._tensor_to_proto(v),
            ))

        if state.attention_mask:
            proto.attention_mask.CopyFrom(self._tensor_to_proto(state.attention_mask))

        if state.position_ids:
            proto.position_ids.CopyFrom(self._tensor_to_proto(state.position_ids))

        return proto

    def _state_from_proto(self, proto: InterweaveStateProto) -> InterweaveState:
        """Convert proto to InterweaveState"""
        state = InterweaveState(
            request_id=proto.request_id,
            sequence_position=proto.sequence_position,
        )

        # Convert KV-cache
        for layer_idx, entry in proto.kv_cache.items():
            state.kv_cache[layer_idx] = (
                self._tensor_from_proto(entry.key),
                self._tensor_from_proto(entry.value),
            )

        if proto.HasField('attention_mask'):
            state.attention_mask = self._tensor_from_proto(proto.attention_mask)

        if proto.HasField('position_ids'):
            state.position_ids = self._tensor_from_proto(proto.position_ids)

        return state

    # ========== Cluster Management ==========

    async def refresh_peer_capabilities(self) -> None:
        """Refresh capabilities from all connected peers"""
        from .proto import InterweaveHealthRequest

        async with self._lock:
            for peer_id, peer in self.peers.items():
                if not peer.is_connected:
                    continue

                try:
                    response = await peer.stub.HealthCheck(
                        InterweaveHealthRequest(include_capabilities=True),
                        timeout=10
                    )

                    if response.capabilities:
                        peer.capabilities = BackendCapabilities(
                            name=response.capabilities.backends[0].name if response.capabilities.backends else 'unknown',
                            device_type=response.capabilities.backends[0].device_type if response.capabilities.backends else 'cpu',
                            supported_dtypes=list(response.capabilities.backends[0].supported_dtypes) if response.capabilities.backends else [],
                            preferred_dtype=response.capabilities.backends[0].preferred_dtype if response.capabilities.backends else 'f32',
                            memory_available=response.capabilities.total_memory,
                        )
                        peer.last_seen = time.time()

                except Exception as e:
                    logger.warning(f"Failed to refresh capabilities for {peer_id}: {e}")

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the entire cluster"""
        status = {
            'node_id': self.node_id,
            'local_backend': self.local_backend.name if self.local_backend else None,
            'peers': {},
        }

        async with self._lock:
            for peer_id, peer in self.peers.items():
                status['peers'][peer_id] = {
                    'address': peer.address,
                    'connected': peer.is_connected,
                    'backend': peer.capabilities.name,
                    'device_type': peer.capabilities.device_type,
                    'latency_ms': peer.latency_ms,
                    'last_seen': time.time() - peer.last_seen,
                }

        return status

    async def cleanup(self) -> None:
        """Cleanup all resources"""
        async with self._lock:
            for peer in self.peers.values():
                await peer.disconnect()
            self.peers.clear()

        if self.local_backend:
            await self.local_backend.cleanup()


class InterweaveServiceImpl:
    """
    gRPC service implementation for InterweaveService.

    This class handles incoming forward requests from peers.
    """

    def __init__(self, router: InterweaveRouter):
        self.router = router
        self._start_time = time.time()
        self._active_requests = 0

    async def Forward(
        self,
        request: ForwardRequest,
        context: grpc.aio.ServicerContext,
    ) -> ForwardResponse:
        """Handle incoming forward request"""
        self._active_requests += 1
        start_time = time.time()

        try:
            # Convert proto to internal format
            shard = InterweaveShard(
                model_id=request.shard.model_id,
                start_layer=request.shard.start_layer,
                end_layer=request.shard.end_layer,
                n_layers=request.shard.n_layers,
                preferred_backends=tuple(request.shard.preferred_backends),
                required_dtype=request.shard.required_dtype or None,
                memory_estimate=request.shard.memory_estimate,
                is_embedding=request.shard.is_embedding,
                is_output=request.shard.is_output,
                requires_kv_cache=request.shard.requires_kv_cache,
            )

            input_tensor = self.router._tensor_from_proto(request.input)

            state = None
            if request.HasField('state'):
                state = self.router._state_from_proto(request.state)

            # Execute locally (don't route to peers to avoid loops)
            if self.router.local_backend is None:
                raise RuntimeError("No local backend available")

            output, new_state = await self.router.local_backend.forward(
                input_tensor, shard, state
            )

            # Build response
            response = ForwardResponse(
                output=self.router._tensor_to_proto(output),
                backend_used=self.router.local_backend.name,
                inference_time_ms=(time.time() - start_time) * 1000,
            )

            if new_state:
                response.updated_state.CopyFrom(
                    self.router._state_to_proto(new_state)
                )

            return response

        except Exception as e:
            logger.error(f"Forward failed: {e}")
            return ForwardResponse(
                error_message=str(e),
                inference_time_ms=(time.time() - start_time) * 1000,
            )

        finally:
            self._active_requests -= 1

    async def HealthCheck(
        self,
        request: 'InterweaveHealthRequest',
        context: grpc.aio.ServicerContext,
    ) -> 'InterweaveHealthResponse':
        """Handle health check request"""
        from .proto import InterweaveHealthResponse, NodeCapabilities, BackendCapabilities

        response = InterweaveHealthResponse(
            is_healthy=True,
            uptime_seconds=int(time.time() - self._start_time),
            active_requests=self._active_requests,
        )

        if request.include_capabilities and self.router.local_backend:
            caps = await self.router.local_backend.get_capabilities()
            mem = await self.router.local_backend.get_memory_available()

            response.capabilities.CopyFrom(NodeCapabilities(
                node_id=self.router.node_id,
                total_memory=mem,
                backends=[BackendCapabilities(
                    name=caps.name,
                    device_type=caps.device_type,
                    supported_dtypes=caps.supported_dtypes,
                    preferred_dtype=caps.preferred_dtype,
                    memory_available=mem,
                )],
            ))

        return response
