#!/usr/bin/env python3
"""
Real Model Distributed Inference Server

Runs actual Llama model inference with real weights across heterogeneous nodes.
"""

import asyncio
import json
import time
import socket
import base64
import argparse
import numpy as np
from typing import Dict, Optional, List
from aiohttp import web, ClientSession

from .tensor_format import UniversalTensor, DType


class RealModelNode:
    """A node that runs real Llama inference with actual weights"""

    def __init__(self, model_id: str, start_layer: int, end_layer: int, n_layers: int, port: int = 8090):
        self.node_id = socket.gethostname()
        self.port = port
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.n_layers = n_layers
        self.peers: Dict[str, str] = {}

        self.engine = None
        self.shard = None
        self.model_loaded = False

        # Detect backend
        self.backend = self._detect_backend()
        self.device = self._get_device()

        # Large request support (2GB for tensors)
        self.app = web.Application(client_max_size=2 * 1024 * 1024 * 1024)
        self._setup_routes()

    def _detect_backend(self) -> str:
        try:
            from tinygrad import Device
            if 'CUDA' in Device.DEFAULT:
                return 'tinygrad_cuda'
            elif 'GPU' in Device.DEFAULT or 'METAL' in Device.DEFAULT:
                return 'tinygrad_gpu'
        except:
            pass
        return 'tinygrad_cpu'

    def _get_device(self) -> str:
        if 'cuda' in self.backend:
            return 'CUDA'
        elif 'gpu' in self.backend or 'metal' in self.backend:
            return 'GPU'
        return 'CPU'

    def _setup_routes(self):
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/info', self.info_handler)
        self.app.router.add_post('/forward', self.forward_handler)
        self.app.router.add_post('/register_peer', self.register_peer_handler)

    async def load_model(self):
        """Load the real model weights"""
        print(f"[{self.node_id}] Loading {self.model_id} layers {self.start_layer}-{self.end_layer}...")

        from exo.download.new_shard_download import NewShardDownloader
        from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
        from exo.inference.shard import Shard

        self.shard = Shard(
            model_id=self.model_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            n_layers=self.n_layers
        )

        downloader = NewShardDownloader()
        self.engine = TinygradDynamicShardInferenceEngine(downloader)

        start = time.perf_counter()
        await self.engine.ensure_shard(self.shard)
        load_time = time.perf_counter() - start

        self.model_loaded = True
        print(f"[{self.node_id}] Model loaded in {load_time:.2f}s")

    async def health_handler(self, request):
        return web.json_response({
            'status': 'ok',
            'node_id': self.node_id,
            'model_loaded': self.model_loaded
        })

    async def info_handler(self, request):
        return web.json_response({
            'node_id': self.node_id,
            'model_id': self.model_id,
            'layers': f"{self.start_layer}-{self.end_layer}",
            'n_layers': self.n_layers,
            'backend': self.backend,
            'device': self.device,
            'model_loaded': self.model_loaded,
            'peers': self.peers,
        })

    async def register_peer_handler(self, request):
        data = await request.json()
        peer_id = data['node_id']
        peer_addr = data['address']
        self.peers[peer_id] = peer_addr
        print(f"[{self.node_id}] Registered peer: {peer_id} @ {peer_addr}")
        return web.json_response({'status': 'registered'})

    async def forward_handler(self, request):
        """Handle forward pass with real model inference"""
        if not self.model_loaded:
            return web.json_response({'error': 'Model not loaded'}, status=503)

        start = time.perf_counter()
        data = await request.json()

        request_id = data.get('request_id', 'unknown')
        next_peer = data.get('next_peer')

        # Get input - either tokens or tensor
        if 'tokens' in data:
            # First layer - receives tokens
            tokens = np.array(data['tokens'], dtype=np.int64)
            input_data = tokens
            print(f"[{self.node_id}] Forward: tokens shape={tokens.shape}")
        else:
            # Middle/last layer - receives tensor
            tensor_bytes = base64.b64decode(data['tensor'])
            input_tensor = UniversalTensor.deserialize(tensor_bytes)
            input_data = input_tensor.to_numpy()
            print(f"[{self.node_id}] Forward: tensor shape={input_data.shape}")

        # Run real inference
        inference_start = time.perf_counter()
        result = await self.engine.infer_tensor(request_id, self.shard, input_data)

        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result

        inference_time = (time.perf_counter() - inference_start) * 1000
        print(f"[{self.node_id}] Inference: {inference_time:.1f}ms, output shape={output.shape}")

        # Forward to next peer if specified
        if next_peer and next_peer in self.peers:
            peer_addr = self.peers[next_peer]
            output_tensor = UniversalTensor.from_numpy(output.astype(np.float16))
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
                        'layers': f"{self.start_layer}-{self.end_layer}",
                        'time_ms': inference_time
                    }]
                    return web.json_response(result)

        # Final node - return result
        total_time = (time.perf_counter() - start) * 1000

        # Check if this is the last layer - if so, sample a token
        if self.end_layer == self.n_layers - 1:
            # Sample from logits
            logits = output[0, -1, :] if len(output.shape) == 3 else output[-1, :]
            next_token = int(np.argmax(logits))

            return web.json_response({
                'status': 'success',
                'next_token': next_token,
                'output_shape': list(output.shape),
                'inference_time_ms': inference_time,
                'total_time_ms': total_time,
                'chain': [{
                    'node': self.node_id,
                    'layers': f"{self.start_layer}-{self.end_layer}",
                    'time_ms': inference_time
                }]
            })
        else:
            # Return tensor for next node
            output_tensor = UniversalTensor.from_numpy(output.astype(np.float16))
            tensor_bytes = base64.b64encode(output_tensor.serialize()).decode()

            return web.json_response({
                'status': 'success',
                'tensor': tensor_bytes,
                'output_shape': list(output.shape),
                'inference_time_ms': inference_time,
                'total_time_ms': total_time,
            })

    async def start(self):
        """Start the server"""
        # Load model first
        await self.load_model()

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        print(f"\n{'='*60}")
        print(f"REAL MODEL NODE STARTED")
        print(f"{'='*60}")
        print(f"Node ID: {self.node_id}")
        print(f"Address: 0.0.0.0:{self.port}")
        print(f"Model: {self.model_id}")
        print(f"Layers: {self.start_layer}-{self.end_layer}/{self.n_layers}")
        print(f"Backend: {self.backend} ({self.device})")
        print(f"{'='*60}\n")

        return runner


async def run_real_distributed_inference(nodes: List[dict], prompt_tokens: List[int]):
    """
    Run distributed inference across nodes with real model.

    nodes: [{'addr': 'host:port', 'next': 'peer_id'}, ...]
    """
    print("\n" + "="*60)
    print("REAL DISTRIBUTED LLAMA INFERENCE")
    print("="*60)

    async with ClientSession() as session:
        # Check all nodes
        print("\n[1/3] Checking nodes...")
        node_info = {}
        for node in nodes:
            addr = node['addr']
            async with session.get(f"http://{addr}/info") as resp:
                info = await resp.json()
                node_info[addr] = info
                status = "READY" if info['model_loaded'] else "LOADING"
                print(f"   {info['node_id']}: layers {info['layers']} ({info['backend']}) - {status}")

        # Register peers
        print("\n[2/3] Registering peers...")
        for i, node in enumerate(nodes):
            if i < len(nodes) - 1:
                next_node = nodes[i + 1]
                next_info = node_info[next_node['addr']]
                await session.post(f"http://{node['addr']}/register_peer", json={
                    'node_id': next_info['node_id'],
                    'address': next_node['addr'],
                })

        # Run inference
        print("\n[3/3] Running inference...")
        print(f"   Input tokens: {prompt_tokens}")

        start = time.perf_counter()

        # Send to first node
        first_node = nodes[0]
        next_peer = node_info[nodes[1]['addr']]['node_id'] if len(nodes) > 1 else None

        async with session.post(f"http://{first_node['addr']}/forward", json={
            'tokens': [prompt_tokens],
            'request_id': 'real-test-001',
            'next_peer': next_peer,
            'final_peer': node_info[nodes[-1]['addr']]['node_id'] if len(nodes) > 2 else None,
        }) as resp:
            result = await resp.json()

        total_time = (time.perf_counter() - start) * 1000

        print(f"\n{'='*60}")
        print("REAL DISTRIBUTED INFERENCE COMPLETE")
        print(f"{'='*60}")

        if 'chain' in result:
            print("\nInference chain:")
            for step in result['chain']:
                print(f"   {step['node']}: layers {step['layers']} - {step['time_ms']:.1f}ms")

        if 'next_token' in result:
            print(f"\nGenerated token ID: {result['next_token']}")

        print(f"Total time: {total_time:.1f}ms")
        print(f"{'='*60}\n")

        return result


def main():
    parser = argparse.ArgumentParser(description='Real Model Distributed Server')
    parser.add_argument('--model', type=str, default='llama-3.2-1b', help='Model ID')
    parser.add_argument('--start-layer', type=int, required=True, help='Start layer')
    parser.add_argument('--end-layer', type=int, required=True, help='End layer')
    parser.add_argument('--n-layers', type=int, default=16, help='Total layers in model')
    parser.add_argument('--port', type=int, default=8090, help='Port')
    args = parser.parse_args()

    node = RealModelNode(
        model_id=args.model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        n_layers=args.n_layers,
        port=args.port
    )

    async def run():
        await node.start()
        while True:
            await asyncio.sleep(3600)

    asyncio.run(run())


if __name__ == '__main__':
    main()
