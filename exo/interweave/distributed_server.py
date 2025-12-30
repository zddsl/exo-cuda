#!/usr/bin/env python3
"""
Interweave Distributed Inference Server

HTTP-based server for cross-node tensor transfer and distributed inference.
Works on all platforms (no gRPC required).
"""

import asyncio
import json
import time
import socket
import base64
import numpy as np
from typing import Dict, Optional, List
from aiohttp import web, ClientSession

from .tensor_format import UniversalTensor, DType
from .backend import BackendRegistry


class InterweaveNode:
    """A node in the distributed inference network"""

    def __init__(self, node_id: str = None, port: int = 8089):
        self.node_id = node_id or socket.gethostname()
        self.port = port
        self.peers: Dict[str, str] = {}  # node_id -> "host:port"
        self.backends = BackendRegistry.detect_available()

        # Determine primary backend
        if 'tinygrad_cuda' in self.backends:
            self.primary_backend = 'tinygrad_cuda'
            self.device = 'CUDA'
        elif 'tinygrad_opencl' in self.backends:
            self.primary_backend = 'tinygrad_opencl'
            self.device = 'GPU'
        else:
            self.primary_backend = 'cpu'
            self.device = 'CPU'

        # Support large tensors up to 2GB (for 70B model activations)
        self.app = web.Application(client_max_size=2 * 1024 * 1024 * 1024)
        self._setup_routes()

        # Stats
        self.requests_processed = 0
        self.tensors_received = 0
        self.tensors_sent = 0

    def _setup_routes(self):
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/info', self.info_handler)
        self.app.router.add_post('/forward', self.forward_handler)
        self.app.router.add_post('/register_peer', self.register_peer_handler)

    async def health_handler(self, request):
        return web.json_response({'status': 'ok', 'node_id': self.node_id})

    async def info_handler(self, request):
        return web.json_response({
            'node_id': self.node_id,
            'backends': self.backends,
            'primary_backend': self.primary_backend,
            'device': self.device,
            'peers': self.peers,
            'stats': {
                'requests_processed': self.requests_processed,
                'tensors_received': self.tensors_received,
                'tensors_sent': self.tensors_sent,
            }
        })

    async def register_peer_handler(self, request):
        data = await request.json()
        peer_id = data['node_id']
        peer_addr = data['address']
        self.peers[peer_id] = peer_addr
        print(f"[{self.node_id}] Registered peer: {peer_id} @ {peer_addr}")
        return web.json_response({'status': 'registered', 'peer_id': peer_id})

    async def forward_handler(self, request):
        """Handle forward pass request with tensor data"""
        start = time.perf_counter()
        self.requests_processed += 1
        self.tensors_received += 1

        data = await request.json()

        # Deserialize input tensor
        tensor_bytes = base64.b64decode(data['tensor'])
        input_tensor = UniversalTensor.deserialize(tensor_bytes)

        layer_start = data.get('layer_start', 0)
        layer_end = data.get('layer_end', 0)
        next_node = data.get('next_node')

        print(f"[{self.node_id}] Forward: layers {layer_start}-{layer_end}, shape={input_tensor.shape}")

        # Run computation
        output_tensor = await self._compute_forward(input_tensor, layer_start, layer_end)

        compute_time = (time.perf_counter() - start) * 1000

        # If there's a next node, forward to it
        if next_node and next_node in self.peers:
            self.tensors_sent += 1
            result = await self._forward_to_peer(
                next_node,
                output_tensor,
                data.get('next_layer_start', layer_end + 1),
                data.get('next_layer_end', layer_end + 1),
                data.get('final_node')
            )
            return web.json_response(result)

        # Otherwise return the result
        output_bytes = base64.b64encode(output_tensor.serialize()).decode()
        total_time = (time.perf_counter() - start) * 1000

        return web.json_response({
            'status': 'success',
            'node_id': self.node_id,
            'backend': self.primary_backend,
            'output_tensor': output_bytes,
            'output_shape': list(output_tensor.shape),
            'compute_time_ms': compute_time,
            'total_time_ms': total_time,
        })

    async def _compute_forward(self, input_tensor: UniversalTensor, layer_start: int, layer_end: int) -> UniversalTensor:
        """Run transformer layer computation"""

        if self.primary_backend in ('tinygrad_cuda', 'tinygrad_opencl'):
            return await self._compute_tinygrad(input_tensor)
        else:
            return await self._compute_numpy(input_tensor)

    async def _compute_tinygrad(self, input_tensor: UniversalTensor) -> UniversalTensor:
        """Compute with tinygrad backend"""
        from tinygrad import Tensor

        device = self.device
        np_input = input_tensor.to_numpy().astype(np.float32).copy()
        batch, seq, hidden = np_input.shape

        x = Tensor(np_input, device=device)

        # Transformer layer computation
        n_heads = 32
        head_dim = hidden // n_heads

        # Attention weights
        wq = Tensor.randn(hidden, hidden, device=device) * 0.02
        wk = Tensor.randn(hidden, hidden, device=device) * 0.02
        wv = Tensor.randn(hidden, hidden, device=device) * 0.02
        wo = Tensor.randn(hidden, hidden, device=device) * 0.02

        # Attention forward
        q = (x @ wq).reshape(batch, seq, n_heads, head_dim).transpose(1, 2)
        k = (x @ wk).reshape(batch, seq, n_heads, head_dim).transpose(1, 2)
        v = (x @ wv).reshape(batch, seq, n_heads, head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
        attn = scores.softmax(axis=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(batch, seq, hidden) @ wo

        # FFN
        w1 = Tensor.randn(hidden, hidden * 4, device=device) * 0.02
        w2 = Tensor.randn(hidden * 4, hidden, device=device) * 0.02
        ffn_out = (attn_out @ w1).relu() @ w2

        # Residual
        output = (x + attn_out + ffn_out).numpy()

        return UniversalTensor.from_numpy(output.astype(np.float16))

    async def _compute_numpy(self, input_tensor: UniversalTensor) -> UniversalTensor:
        """Compute with numpy (CPU fallback)"""
        np_input = input_tensor.to_numpy().astype(np.float32).copy()
        batch, seq, hidden = np_input.shape

        # Simple transformer simulation
        wq = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        q = np_input @ wq
        k = np_input @ wq
        v = np_input @ wq

        attn = q @ k.transpose(0, 2, 1) * (hidden ** -0.5)
        attn_out = attn @ v

        # FFN
        w1 = np.random.randn(hidden, hidden * 4).astype(np.float32) * 0.02
        w2 = np.random.randn(hidden * 4, hidden).astype(np.float32) * 0.02
        ffn_out = np.maximum(0, attn_out @ w1) @ w2

        output = np_input + attn_out + ffn_out

        return UniversalTensor.from_numpy(output.astype(np.float16))

    async def _forward_to_peer(self, peer_id: str, tensor: UniversalTensor,
                               layer_start: int, layer_end: int, final_node: str = None) -> dict:
        """Forward tensor to a peer node"""
        peer_addr = self.peers[peer_id]
        url = f"http://{peer_addr}/forward"

        tensor_bytes = base64.b64encode(tensor.serialize()).decode()

        async with ClientSession() as session:
            async with session.post(url, json={
                'tensor': tensor_bytes,
                'layer_start': layer_start,
                'layer_end': layer_end,
                'next_node': final_node,
            }) as resp:
                return await resp.json()

    async def start(self):
        """Start the server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        print(f"\n{'='*60}")
        print(f"INTERWEAVE NODE STARTED")
        print(f"{'='*60}")
        print(f"Node ID: {self.node_id}")
        print(f"Address: 0.0.0.0:{self.port}")
        print(f"Backend: {self.primary_backend} ({self.device})")
        print(f"{'='*60}\n")
        return runner


async def run_distributed_test(nodes: List[str]):
    """
    Run a distributed inference test across multiple nodes.

    Args:
        nodes: List of node addresses ["host1:port1", "host2:port2", ...]
    """
    print("\n" + "="*60)
    print("DISTRIBUTED INFERENCE TEST")
    print("="*60)

    if len(nodes) < 2:
        print("Need at least 2 nodes for distributed test")
        return

    async with ClientSession() as session:
        # Check all nodes are up
        print("\n[1/4] Checking node health...")
        node_info = {}
        for addr in nodes:
            try:
                async with session.get(f"http://{addr}/info") as resp:
                    info = await resp.json()
                    node_info[addr] = info
                    print(f"   {addr}: {info['primary_backend']} ({info['device']})")
            except Exception as e:
                print(f"   {addr}: FAILED - {e}")
                return

        # Register peers
        print("\n[2/4] Registering peers...")
        for i, addr in enumerate(nodes):
            for j, peer_addr in enumerate(nodes):
                if i != j:
                    await session.post(f"http://{addr}/register_peer", json={
                        'node_id': node_info[peer_addr]['node_id'],
                        'address': peer_addr,
                    })

        # Create input tensor
        print("\n[3/4] Creating input tensor...")
        batch, seq, hidden = 1, 32, 2048
        input_data = np.random.randn(batch, seq, hidden).astype(np.float16)
        input_tensor = UniversalTensor.from_numpy(input_data)
        tensor_bytes = base64.b64encode(input_tensor.serialize()).decode()
        print(f"   Shape: {input_tensor.shape}")
        print(f"   Size: {len(tensor_bytes) / 1024:.1f} KB")

        # Run distributed forward
        print("\n[4/4] Running distributed forward pass...")
        print(f"   Path: {' -> '.join([node_info[a]['node_id'] for a in nodes])}")

        start = time.perf_counter()

        # Start at first node, chain through all
        current_tensor = tensor_bytes
        results = []

        for i, addr in enumerate(nodes):
            next_node = node_info[nodes[i+1]]['node_id'] if i < len(nodes) - 1 else None

            async with session.post(f"http://{addr}/forward", json={
                'tensor': current_tensor,
                'layer_start': i * 10,
                'layer_end': (i + 1) * 10 - 1,
            }) as resp:
                result = await resp.json()
                results.append(result)
                print(f"   {node_info[addr]['node_id']}: {result.get('compute_time_ms', 0):.1f}ms")

                if 'output_tensor' in result:
                    current_tensor = result['output_tensor']

        total_time = (time.perf_counter() - start) * 1000

        # Decode final output
        final_tensor = UniversalTensor.deserialize(base64.b64decode(current_tensor))

        print(f"\n{'='*60}")
        print("DISTRIBUTED INFERENCE COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}ms")
        print(f"Output shape: {final_tensor.shape}")
        print(f"Nodes used: {len(nodes)}")
        print(f"{'='*60}\n")

        return {
            'total_time_ms': total_time,
            'output_shape': list(final_tensor.shape),
            'node_results': results,
        }


def main():
    """Start the server"""
    import argparse

    parser = argparse.ArgumentParser(description='Interweave Distributed Inference Server')
    parser.add_argument('--port', type=int, default=8089, help='Port to listen on')
    parser.add_argument('--node-id', type=str, default=None, help='Node ID')
    args = parser.parse_args()

    node = InterweaveNode(node_id=args.node_id, port=args.port)

    async def run():
        await node.start()
        # Keep running
        while True:
            await asyncio.sleep(3600)

    asyncio.run(run())


if __name__ == '__main__':
    main()
