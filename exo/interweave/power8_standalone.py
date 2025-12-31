#!/usr/bin/env python3
"""
Standalone Power8 Distributed Server

This is a ZERO-DEPENDENCY server for Power8/ppc64le that only requires:
- Python 3.8+
- numpy (pip3 install numpy)
- aiohttp (pip3 install aiohttp)

NO tinygrad required! This can run on any architecture.

Usage:
    pip3 install numpy aiohttp
    python3 power8_standalone.py --port 8090 --mode relay

Copy this single file to Power8 and run it directly.
"""

import asyncio
import json
import time
import socket
import base64
import argparse
import struct
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Check dependencies
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Run: pip3 install numpy")
    exit(1)

try:
    from aiohttp import web, ClientSession
except ImportError:
    print("ERROR: aiohttp required. Run: pip3 install aiohttp")
    exit(1)


# ============================================================================
# Universal Tensor Format (compatible with Interweave protocol)
# ============================================================================

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
            DType.BF16: np.float32,  # BF16 stored as F32 in numpy
            DType.I32: np.int32,
            DType.I16: np.int16,
            DType.I8: np.int8,
            DType.I4: np.int8,
            DType.U8: np.uint8,
        }
        return np.dtype(mapping[self])


@dataclass
class UniversalTensor:
    """Wire-compatible tensor format (same as Interweave UniversalTensor)"""
    data: bytes
    shape: tuple
    dtype: DType
    scale: Optional[float] = None
    zero_point: Optional[int] = None

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        np_dtype = self.dtype.to_numpy()
        arr = np.frombuffer(self.data, dtype=np_dtype)
        return arr.reshape(self.shape).copy()  # Copy to make writable

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'UniversalTensor':
        """Create from numpy array"""
        arr = np.ascontiguousarray(arr)
        dtype_map = {
            np.float32: DType.F32,
            np.float16: DType.F16,
            np.int32: DType.I32,
            np.int64: DType.I32,  # Map int64 to int32 for compatibility
            np.int8: DType.I8,
            np.uint8: DType.U8,
        }
        dtype = dtype_map.get(arr.dtype.type, DType.F32)

        # Convert int64 to int32 if needed
        if arr.dtype == np.int64:
            arr = arr.astype(np.int32)

        return cls(
            data=arr.tobytes(),
            shape=tuple(arr.shape),
            dtype=dtype
        )

    def serialize(self) -> bytes:
        """Serialize for wire transfer"""
        parts = []

        # Header (compatible with Interweave protocol)
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
    def deserialize(cls, data: bytes) -> 'UniversalTensor':
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


# ============================================================================
# Power8 Node Server
# ============================================================================

class Power8Node:
    """
    CPU-only node using numpy for tensor operations.

    Designed for Power8 with 576GB RAM as a memory server that:
    - Relays tensors between GPU nodes
    - Caches large tensors in RAM
    - Can perform simple numpy operations if needed
    """

    def __init__(self, port: int = 8090, mode: str = 'relay'):
        self.node_id = socket.gethostname()
        self.port = port
        self.mode = mode
        self.peers: Dict[str, str] = {}

        # Tensor cache (leverage 576GB RAM!)
        self.tensor_cache: Dict[str, np.ndarray] = {}
        self.kv_cache: Dict[str, np.ndarray] = {}

        # Stats
        self.requests_processed = 0
        self.bytes_transferred = 0

        # Setup web app (2GB max request for large tensors)
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

    def _get_memory_info(self) -> dict:
        """Get system memory info"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1]) * 1024  # KB to bytes
                        meminfo[key] = value
            return meminfo
        except:
            return {}

    async def health_handler(self, request):
        return web.json_response({
            'status': 'ok',
            'node_id': self.node_id,
            'backend': 'numpy_cpu',
            'model_loaded': True  # Always ready
        })

    async def info_handler(self, request):
        meminfo = self._get_memory_info()
        return web.json_response({
            'node_id': self.node_id,
            'backend': 'numpy_cpu',
            'device': 'CPU',
            'mode': self.mode,
            'peers': self.peers,
            'requests_processed': self.requests_processed,
            'bytes_transferred': self.bytes_transferred,
            'cache_entries': len(self.tensor_cache),
            'memory_total_gb': meminfo.get('MemTotal', 0) / (1024**3),
            'memory_available_gb': meminfo.get('MemAvailable', 0) / (1024**3),
            'model_loaded': True,
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
            input_tensor = UniversalTensor.deserialize(tensor_bytes)
            input_np = input_tensor.to_numpy()
            self.bytes_transferred += len(tensor_bytes)
            print(f"[{self.node_id}] Forward: tensor shape={input_np.shape}, dtype={input_np.dtype}")
        elif 'tokens' in data:
            tokens = np.array(data['tokens'], dtype=np.int64)
            input_np = tokens
            print(f"[{self.node_id}] Forward: tokens shape={tokens.shape}")
        else:
            return web.json_response({'error': 'No input provided'}, status=400)

        # Process based on mode
        process_start = time.perf_counter()

        if self.mode == 'relay':
            # Pass through unchanged (memory server role)
            output_np = input_np.astype(np.float32) if input_np.dtype != np.float32 else input_np
        elif self.mode == 'compute':
            # Apply numpy operations (slow but works)
            output_np = self._numpy_forward(input_np)
        else:  # cache
            # Store in cache and forward
            cache_key = f"{request_id}_{self.requests_processed}"
            self.tensor_cache[cache_key] = input_np.copy()
            output_np = input_np.astype(np.float32) if input_np.dtype != np.float32 else input_np

        process_time = (time.perf_counter() - process_start) * 1000
        print(f"[{self.node_id}] Processed in {process_time:.1f}ms (mode={self.mode})")

        # Forward to next peer if specified
        if next_peer and next_peer in self.peers:
            peer_addr = self.peers[next_peer]
            output_tensor = UniversalTensor.from_numpy(output_np.astype(np.float32))
            tensor_bytes_out = base64.b64encode(output_tensor.serialize()).decode()

            async with ClientSession() as session:
                next_data = {
                    'tensor': tensor_bytes_out,
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

        # Return result (final node)
        total_time = (time.perf_counter() - start) * 1000
        output_tensor = UniversalTensor.from_numpy(output_np.astype(np.float32))
        tensor_bytes_out = base64.b64encode(output_tensor.serialize()).decode()

        return web.json_response({
            'status': 'success',
            'tensor': tensor_bytes_out,
            'output_shape': list(output_np.shape),
            'inference_time_ms': process_time,
            'total_time_ms': total_time,
            'chain': [{
                'node': self.node_id,
                'backend': 'numpy_cpu',
                'mode': self.mode,
                'time_ms': process_time
            }]
        })

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Simple numpy layer forward pass.

        This is SLOW compared to GPU but works on any CPU including Power8.
        """
        x = x.astype(np.float32)

        # Simple operations (placeholder for real layer computation)
        # Layer norm approximation
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)

        return x_norm

    async def store_handler(self, request):
        """Store tensor in cache (use that 576GB RAM!)"""
        data = await request.json()
        key = data['key']
        tensor_bytes = base64.b64decode(data['tensor'])
        tensor = UniversalTensor.deserialize(tensor_bytes)

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
        tensor = UniversalTensor.from_numpy(arr)
        tensor_bytes = base64.b64encode(tensor.serialize()).decode()

        return web.json_response({
            'status': 'retrieved',
            'key': key,
            'tensor': tensor_bytes,
            'shape': list(arr.shape)
        })

    async def memory_handler(self, request):
        """Get detailed memory info"""
        meminfo = self._get_memory_info()
        cache_bytes = sum(arr.nbytes for arr in self.tensor_cache.values())

        return web.json_response({
            'node_id': self.node_id,
            'total_gb': meminfo.get('MemTotal', 0) / (1024**3),
            'available_gb': meminfo.get('MemAvailable', 0) / (1024**3),
            'cache_gb': cache_bytes / (1024**3),
            'cache_entries': len(self.tensor_cache),
        })

    async def start(self):
        """Start the server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        meminfo = self._get_memory_info()
        mem_gb = meminfo.get('MemTotal', 0) / (1024**3)

        print(f"\n{'='*60}")
        print(f"POWER8 NUMPY SERVER STARTED")
        print(f"{'='*60}")
        print(f"Node ID: {self.node_id}")
        print(f"Address: 0.0.0.0:{self.port}")
        print(f"Backend: numpy_cpu (pure Python + numpy)")
        print(f"Mode: {self.mode}")
        print(f"Memory: {mem_gb:.1f} GB total")
        print(f"{'='*60}")
        print(f"\nEndpoints:")
        print(f"  GET  /health       - Health check")
        print(f"  GET  /info         - Node info")
        print(f"  POST /forward      - Forward tensor")
        print(f"  POST /register_peer - Register peer node")
        print(f"  POST /store        - Store tensor in cache")
        print(f"  GET  /retrieve/key - Retrieve tensor from cache")
        print(f"  GET  /memory       - Memory stats")
        print(f"{'='*60}\n")

        return runner


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Power8 Standalone Numpy Server for Interweave Protocol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  relay   - Forward tensors between GPU nodes (default)
  compute - Apply numpy operations (slow but works)
  cache   - Store tensors in RAM for GPU nodes to retrieve

Examples:
  # Run as relay (memory server for GPU nodes)
  python3 power8_standalone.py --port 8090 --mode relay

  # Run as compute node (slow numpy inference)
  python3 power8_standalone.py --port 8090 --mode compute

  # Run as cache server (store/retrieve tensors)
  python3 power8_standalone.py --port 8090 --mode cache
        """
    )
    parser.add_argument('--port', type=int, default=8090, help='Port (default: 8090)')
    parser.add_argument('--mode', choices=['relay', 'compute', 'cache'], default='relay',
                        help='Operation mode (default: relay)')
    args = parser.parse_args()

    node = Power8Node(port=args.port, mode=args.mode)

    async def run():
        await node.start()
        print(f"Server running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(3600)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
