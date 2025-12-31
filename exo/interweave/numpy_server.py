#!/usr/bin/env python3
"""
Numpy-Only Distributed Server for Power8/CPU nodes

This server runs on systems where tinygrad is not available (e.g., ppc64le Power8).
It uses pure numpy for tensor operations and HTTP for communication.

Key Features:
- No tinygrad dependency (pure numpy + aiohttp)
- Participates in Interweave distributed inference
- 576GB RAM on Power8 can cache large tensors/KV-cache
- Forwards tensors between GPU nodes
- Optional numpy-based layer computation

Usage:
    python3 -m exo.interweave.numpy_server --port 8090 --mode relay

Modes:
- relay: Forward tensors between nodes (memory server role)
- compute: Apply simple numpy operations (slow but works)
- cache: Store KV-cache for GPU nodes to retrieve
"""

import asyncio
import json
import time
import socket
import base64
import argparse
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

# Only require aiohttp - no tinygrad!
try:
    from aiohttp import web, ClientSession
except ImportError:
    print("ERROR: aiohttp required. Install with: pip3 install aiohttp")
    raise

# Import tensor format (numpy-only parts)
import struct
from enum import Enum


class DType(str, Enum):
    """Canonical dtype identifiers"""
    F32 = 'f32'
    F16 = 'f16'
    BF16 = 'bf16'
    I32 = 'i32'
    I16 = 'i16'
    I8 = 'i8'
    I4 = 'i4'
    U8 = 'u8'

    def to_numpy(self):
        mapping = {
            DType.F32: np.float32,
            DType.F16: np.float16,
            DType.BF16: np.float32,  # BF16 stored as F32
            DType.I32: np.int32,
            DType.I16: np.int16,
            DType.I8: np.int8,
            DType.I4: np.int8,
            DType.U8: np.uint8,
        }
        return np.dtype(mapping[self])


@dataclass
class UniversalTensorLite:
    """Lightweight tensor class using only numpy (no tinygrad dependency)"""
    data: bytes
    shape: tuple
    dtype: DType
    scale: Optional[float] = None
    zero_point: Optional[int] = None

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        np_dtype = self.dtype.to_numpy()
        arr = np.frombuffer(self.data, dtype=np_dtype)
        return arr.reshape(self.shape)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'UniversalTensorLite':
        """Create from numpy array"""
        arr = np.ascontiguousarray(arr)

        # Map numpy dtype to DType
        dtype_map = {
            np.float32: DType.F32,
            np.float16: DType.F16,
            np.int32: DType.I32,
            np.int8: DType.I8,
            np.uint8: DType.U8,
        }
        dtype = dtype_map.get(arr.dtype.type, DType.F32)

        return cls(
            data=arr.tobytes(),
            shape=tuple(arr.shape),
            dtype=dtype
        )

    def serialize(self) -> bytes:
        """Serialize for wire transfer (compatible with UniversalTensor)"""
        parts = []

        # Header (same format as UniversalTensor)
        parts.append(struct.pack('<I', 0x494E5457))  # Magic "INTW"
        parts.append(struct.pack('<B', 1))  # Version
        parts.append(struct.pack('<B', list(DType).index(self.dtype)))
        parts.append(struct.pack('<B', 0))  # Layout: row_major

        # Flags
        flags = 0
        if self.scale is not None:
            flags |= 1
        if self.zero_point is not None:
            flags |= 2
        parts.append(struct.pack('<B', flags))

        # Shape
        parts.append(struct.pack('<I', len(self.shape)))
        for dim in self.shape:
            parts.append(struct.pack('<q', dim))

        # Optional metadata
        if self.scale is not None:
            parts.append(struct.pack('<d', self.scale))
        if self.zero_point is not None:
            parts.append(struct.pack('<i', self.zero_point))

        # Data
        parts.append(self.data)

        return b''.join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> 'UniversalTensorLite':
        """Deserialize from bytes"""
        offset = 0

        # Magic
        magic, = struct.unpack_from('<I', data, offset)
        offset += 4
        if magic != 0x494E5457:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        # Version
        version, = struct.unpack_from('<B', data, offset)
        offset += 1

        # Dtype
        dtype_idx, = struct.unpack_from('<B', data, offset)
        offset += 1
        dtype = list(DType)[dtype_idx]

        # Layout (skip)
        offset += 1

        # Flags
        flags, = struct.unpack_from('<B', data, offset)
        offset += 1
        has_scale = bool(flags & 1)
        has_zero_point = bool(flags & 2)

        # Shape
        ndim, = struct.unpack_from('<I', data, offset)
        offset += 4
        shape = []
        for _ in range(ndim):
            dim, = struct.unpack_from('<q', data, offset)
            offset += 8
            shape.append(dim)
        shape = tuple(shape)

        # Optional metadata
        scale = None
        zero_point = None
        if has_scale:
            scale, = struct.unpack_from('<d', data, offset)
            offset += 8
        if has_zero_point:
            zero_point, = struct.unpack_from('<i', data, offset)
            offset += 4

        # Data
        tensor_data = data[offset:]

        return cls(
            data=tensor_data,
            shape=shape,
            dtype=dtype,
            scale=scale,
            zero_point=zero_point
        )


class NumpyNode:
    """A CPU-only node using numpy for tensor operations"""

    def __init__(self, port: int = 8090, mode: str = 'relay'):
        self.node_id = socket.gethostname()
        self.port = port
        self.mode = mode  # 'relay', 'compute', or 'cache'
        self.peers: Dict[str, str] = {}

        # Tensor cache (for cache mode - leverage 576GB RAM)
        self.tensor_cache: Dict[str, np.ndarray] = {}
        self.kv_cache: Dict[str, np.ndarray] = {}

        # Stats
        self.requests_processed = 0
        self.bytes_transferred = 0

        # Setup web app
        self.app = web.Application(client_max_size=2 * 1024 * 1024 * 1024)
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/info', self.info_handler)
        self.app.router.add_post('/forward', self.forward_handler)
        self.app.router.add_post('/register_peer', self.register_peer_handler)
        self.app.router.add_post('/store', self.store_handler)
        self.app.router.add_get('/retrieve/{key}', self.retrieve_handler)
        self.app.router.add_get('/memory', self.memory_handler)

    async def health_handler(self, request):
        return web.json_response({
            'status': 'ok',
            'node_id': self.node_id,
            'backend': 'numpy_cpu'
        })

    async def info_handler(self, request):
        # Get memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024
        except:
            meminfo = {}

        return web.json_response({
            'node_id': self.node_id,
            'backend': 'numpy_cpu',
            'mode': self.mode,
            'peers': self.peers,
            'requests_processed': self.requests_processed,
            'bytes_transferred': self.bytes_transferred,
            'cache_entries': len(self.tensor_cache),
            'kv_cache_entries': len(self.kv_cache),
            'memory_total': meminfo.get('MemTotal', 0),
            'memory_available': meminfo.get('MemAvailable', 0),
        })

    async def register_peer_handler(self, request):
        data = await request.json()
        peer_id = data['node_id']
        peer_addr = data['address']
        self.peers[peer_id] = peer_addr
        print(f"[{self.node_id}] Registered peer: {peer_id} @ {peer_addr}")
        return web.json_response({'status': 'registered'})

    async def forward_handler(self, request):
        """Handle tensor forward pass"""
        start = time.perf_counter()
        data = await request.json()

        request_id = data.get('request_id', 'unknown')
        next_peer = data.get('next_peer')

        self.requests_processed += 1

        # Get input tensor
        if 'tensor' in data:
            tensor_bytes = base64.b64decode(data['tensor'])
            input_tensor = UniversalTensorLite.deserialize(tensor_bytes)
            input_np = input_tensor.to_numpy()
            self.bytes_transferred += len(tensor_bytes)
            print(f"[{self.node_id}] Forward: tensor shape={input_np.shape}, dtype={input_np.dtype}")
        elif 'tokens' in data:
            # Token input - convert to numpy
            tokens = np.array(data['tokens'], dtype=np.int64)
            input_np = tokens
            print(f"[{self.node_id}] Forward: tokens shape={tokens.shape}")
        else:
            return web.json_response({'error': 'No input provided'}, status=400)

        # Process based on mode
        process_start = time.perf_counter()

        if self.mode == 'relay':
            # Just pass through (memory server role)
            output_np = input_np.astype(np.float32)
        elif self.mode == 'compute':
            # Apply simple numpy operations (slow but works)
            output_np = self._numpy_layer_forward(input_np)
        else:
            # Cache mode - store and forward
            cache_key = f"{request_id}_{self.requests_processed}"
            self.tensor_cache[cache_key] = input_np
            output_np = input_np.astype(np.float32)

        process_time = (time.perf_counter() - process_start) * 1000
        print(f"[{self.node_id}] Processed in {process_time:.1f}ms (mode={self.mode})")

        # Forward to next peer if specified
        if next_peer and next_peer in self.peers:
            peer_addr = self.peers[next_peer]
            output_tensor = UniversalTensorLite.from_numpy(output_np.astype(np.float32))
            tensor_bytes = base64.b64encode(output_tensor.serialize()).decode()

            async with ClientSession() as session:
                next_data = {
                    'tensor': tensor_bytes,
                    'request_id': request_id,
                    'next_peer': data.get('final_peer'),
                }
                async with session.post(f"http://{peer_addr}/forward", json=next_data) as resp:
                    result = await resp.json()
                    result['chain'] = result.get('chain', []) + [{
                        'node': self.node_id,
                        'backend': 'numpy_cpu',
                        'mode': self.mode,
                        'time_ms': process_time
                    }]
                    return web.json_response(result)

        # Return result
        total_time = (time.perf_counter() - start) * 1000
        output_tensor = UniversalTensorLite.from_numpy(output_np.astype(np.float32))
        tensor_bytes = base64.b64encode(output_tensor.serialize()).decode()

        return web.json_response({
            'status': 'success',
            'tensor': tensor_bytes,
            'output_shape': list(output_np.shape),
            'process_time_ms': process_time,
            'total_time_ms': total_time,
            'chain': [{
                'node': self.node_id,
                'backend': 'numpy_cpu',
                'mode': self.mode,
                'time_ms': process_time
            }]
        })

    def _numpy_layer_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Simple numpy-based layer computation.

        This is SLOW compared to GPU but works on any architecture.
        Useful for:
        - Testing distributed protocol
        - Processing overflow from GPU VRAM
        - Simple transformations
        """
        # Ensure f32 for computation
        x = x.astype(np.float32)

        # Simple layer: LayerNorm + Linear projection
        # (This is a placeholder - real layers would use model weights)

        # Layer normalization
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)

        # Identity projection (no weights loaded)
        output = x_norm

        return output

    async def store_handler(self, request):
        """Store tensor in cache (leverage 576GB RAM)"""
        data = await request.json()
        key = data['key']
        tensor_bytes = base64.b64decode(data['tensor'])
        tensor = UniversalTensorLite.deserialize(tensor_bytes)

        self.tensor_cache[key] = tensor.to_numpy()
        self.bytes_transferred += len(tensor_bytes)

        print(f"[{self.node_id}] Stored tensor: {key}, shape={tensor.shape}")

        return web.json_response({
            'status': 'stored',
            'key': key,
            'shape': list(tensor.shape),
            'cache_size': len(self.tensor_cache)
        })

    async def retrieve_handler(self, request):
        """Retrieve tensor from cache"""
        key = request.match_info['key']

        if key not in self.tensor_cache:
            return web.json_response({'error': f'Key not found: {key}'}, status=404)

        arr = self.tensor_cache[key]
        tensor = UniversalTensorLite.from_numpy(arr)
        tensor_bytes = base64.b64encode(tensor.serialize()).decode()

        return web.json_response({
            'status': 'retrieved',
            'key': key,
            'tensor': tensor_bytes,
            'shape': list(arr.shape)
        })

    async def memory_handler(self, request):
        """Get detailed memory information"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1]) * 1024
                        meminfo[key] = value
        except:
            meminfo = {}

        # Calculate cache memory usage
        cache_bytes = sum(arr.nbytes for arr in self.tensor_cache.values())
        kv_cache_bytes = sum(arr.nbytes for arr in self.kv_cache.values())

        return web.json_response({
            'node_id': self.node_id,
            'total_gb': meminfo.get('MemTotal', 0) / (1024**3),
            'available_gb': meminfo.get('MemAvailable', 0) / (1024**3),
            'cache_gb': cache_bytes / (1024**3),
            'kv_cache_gb': kv_cache_bytes / (1024**3),
            'cache_entries': len(self.tensor_cache),
            'kv_cache_entries': len(self.kv_cache),
        })

    async def start(self):
        """Start the numpy server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        # Get memory info for display
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_gb = int(line.split()[1]) / (1024**2)
                        break
                else:
                    mem_gb = 0
        except:
            mem_gb = 0

        print(f"\n{'='*60}")
        print(f"NUMPY CPU NODE STARTED")
        print(f"{'='*60}")
        print(f"Node ID: {self.node_id}")
        print(f"Address: 0.0.0.0:{self.port}")
        print(f"Backend: numpy_cpu (pure numpy, no tinygrad)")
        print(f"Mode: {self.mode}")
        print(f"Memory: {mem_gb:.1f} GB")
        print(f"{'='*60}")
        print(f"\nThis node can:")
        if self.mode == 'relay':
            print(f"  - Relay tensors between GPU nodes")
            print(f"  - Cache tensors in {mem_gb:.0f}GB RAM")
        elif self.mode == 'compute':
            print(f"  - Run numpy-based layer computations (slow)")
            print(f"  - Process overflow from GPU VRAM")
        else:
            print(f"  - Store tensors for GPU nodes to retrieve")
            print(f"  - Act as distributed KV-cache server")
        print(f"{'='*60}\n")

        return runner


def main():
    parser = argparse.ArgumentParser(description='Numpy CPU Distributed Server')
    parser.add_argument('--port', type=int, default=8090, help='Port')
    parser.add_argument('--mode', choices=['relay', 'compute', 'cache'], default='relay',
                        help='Operation mode: relay (forward only), compute (numpy ops), cache (store tensors)')
    args = parser.parse_args()

    node = NumpyNode(port=args.port, mode=args.mode)

    async def run():
        await node.start()
        while True:
            await asyncio.sleep(3600)

    asyncio.run(run())


if __name__ == '__main__':
    main()
