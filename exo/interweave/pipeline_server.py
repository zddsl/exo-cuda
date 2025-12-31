#!/usr/bin/env python3
"""
Pipelined Distributed Inference Server

Solves the 75% overhead problem by:
1. Request queuing - always have work ready for GPU
2. Async transfers - send results while computing next batch
3. Keep-alive tasks - generate warmup work during idle

The GPU should NEVER be idle waiting for network.
"""

import asyncio
import json
import time
import socket
import base64
import argparse
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from aiohttp import web, ClientSession

from .tensor_format import UniversalTensor, DType


@dataclass
class InferenceRequest:
    """A queued inference request"""
    request_id: str
    input_data: np.ndarray
    next_peer: Optional[str] = None
    final_peer: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent


@dataclass
class PipelineStats:
    """Track pipeline efficiency"""
    requests_processed: int = 0
    total_compute_ms: float = 0
    total_transfer_ms: float = 0
    total_idle_ms: float = 0
    queue_depth_sum: int = 0

    @property
    def efficiency(self) -> float:
        total = self.total_compute_ms + self.total_transfer_ms + self.total_idle_ms
        return (self.total_compute_ms / total * 100) if total > 0 else 0

    @property
    def avg_queue_depth(self) -> float:
        return self.queue_depth_sum / self.requests_processed if self.requests_processed > 0 else 0


class PipelinedNode:
    """
    Pipelined inference node that keeps GPU busy.

    Key innovation: Separate queues for input, compute, and output.
    GPU processes from compute queue while network handles I/O queues.
    """

    def __init__(self, model_id: str, start_layer: int, end_layer: int,
                 n_layers: int, port: int = 8090, queue_size: int = 8):
        self.node_id = socket.gethostname()
        self.port = port
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.n_layers = n_layers

        # Queues for pipelining
        self.input_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=queue_size)
        self.output_queue: asyncio.Queue[Tuple[str, np.ndarray, dict]] = asyncio.Queue(maxsize=queue_size)

        # Peer connections (keep-alive)
        self.peers: Dict[str, str] = {}
        self.peer_sessions: Dict[str, ClientSession] = {}

        # Stats
        self.stats = PipelineStats()

        # Engine
        self.engine = None
        self.shard = None
        self.model_loaded = False

        # Pipeline tasks
        self._compute_task: Optional[asyncio.Task] = None
        self._transfer_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        # Setup
        self.backend = self._detect_backend()
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

    def _setup_routes(self):
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/info', self.info_handler)
        self.app.router.add_get('/stats', self.stats_handler)
        self.app.router.add_post('/forward', self.forward_handler)
        self.app.router.add_post('/register_peer', self.register_peer_handler)

    async def load_model(self):
        """Load model weights"""
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

    async def _compute_loop(self):
        """
        Continuous compute loop - processes requests from queue.
        NEVER idles if there's work in the queue.
        """
        print(f"[{self.node_id}] Compute loop started")

        while True:
            # Get next request (blocks if empty)
            idle_start = time.perf_counter()
            request = await self.input_queue.get()
            idle_time = (time.perf_counter() - idle_start) * 1000

            if idle_time > 10:  # Only count significant idle time
                self.stats.total_idle_ms += idle_time

            # Track queue depth
            self.stats.queue_depth_sum += self.input_queue.qsize()

            # Run inference
            compute_start = time.perf_counter()
            try:
                result = await self.engine.infer_tensor(
                    request.request_id,
                    self.shard,
                    request.input_data
                )

                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result

                compute_time = (time.perf_counter() - compute_start) * 1000
                self.stats.total_compute_ms += compute_time
                self.stats.requests_processed += 1

                # Queue output for async transfer
                await self.output_queue.put((
                    request.request_id,
                    output,
                    {
                        'next_peer': request.next_peer,
                        'final_peer': request.final_peer,
                        'compute_time_ms': compute_time,
                    }
                ))

                print(f"[{self.node_id}] Computed {request.request_id}: {compute_time:.1f}ms, queue={self.input_queue.qsize()}")

            except Exception as e:
                print(f"[{self.node_id}] Compute error: {e}")

            self.input_queue.task_done()

    async def _transfer_loop(self):
        """
        Async transfer loop - sends results to next peer.
        Runs in parallel with compute loop.
        """
        print(f"[{self.node_id}] Transfer loop started")

        while True:
            request_id, output, metadata = await self.output_queue.get()

            next_peer = metadata.get('next_peer')

            if next_peer and next_peer in self.peers:
                transfer_start = time.perf_counter()

                try:
                    peer_addr = self.peers[next_peer]
                    session = await self._get_peer_session(next_peer)

                    # Serialize and send
                    output_tensor = UniversalTensor.from_numpy(output.astype(np.float32))
                    tensor_bytes = base64.b64encode(output_tensor.serialize()).decode()

                    async with session.post(f"http://{peer_addr}/forward", json={
                        'tensor': tensor_bytes,
                        'request_id': request_id,
                        'next_peer': metadata.get('final_peer'),
                    }) as resp:
                        result = await resp.json()

                    transfer_time = (time.perf_counter() - transfer_start) * 1000
                    self.stats.total_transfer_ms += transfer_time

                    print(f"[{self.node_id}] Transferred {request_id}: {transfer_time:.1f}ms")

                except Exception as e:
                    print(f"[{self.node_id}] Transfer error: {e}")

            self.output_queue.task_done()

    async def _keepalive_loop(self):
        """
        Keep-alive loop - generates warmup work during idle.
        Prevents GPU from going cold.
        """
        print(f"[{self.node_id}] Keep-alive loop started")

        warmup_interval = 30  # seconds

        while True:
            await asyncio.sleep(warmup_interval)

            # Only warmup if truly idle
            if self.input_queue.empty() and self.output_queue.empty():
                print(f"[{self.node_id}] Running keep-alive warmup...")

                # Small warmup inference to keep GPU hot
                warmup_input = np.zeros((1, 1, 2048), dtype=np.float32)
                try:
                    await self.engine.infer_tensor("warmup", self.shard, warmup_input)
                except:
                    pass

    async def _get_peer_session(self, peer_id: str) -> ClientSession:
        """Get or create keep-alive session for peer"""
        if peer_id not in self.peer_sessions or self.peer_sessions[peer_id].closed:
            self.peer_sessions[peer_id] = ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self.peer_sessions[peer_id]

    async def health_handler(self, request):
        return web.json_response({
            'status': 'ok',
            'node_id': self.node_id,
            'model_loaded': self.model_loaded,
            'queue_depth': self.input_queue.qsize(),
            'efficiency': f"{self.stats.efficiency:.1f}%"
        })

    async def info_handler(self, request):
        return web.json_response({
            'node_id': self.node_id,
            'model_id': self.model_id,
            'layers': f"{self.start_layer}-{self.end_layer}",
            'backend': self.backend,
            'pipeline_mode': True,
            'queue_depth': self.input_queue.qsize(),
            'peers': self.peers,
        })

    async def stats_handler(self, request):
        return web.json_response({
            'node_id': self.node_id,
            'requests_processed': self.stats.requests_processed,
            'total_compute_ms': self.stats.total_compute_ms,
            'total_transfer_ms': self.stats.total_transfer_ms,
            'total_idle_ms': self.stats.total_idle_ms,
            'efficiency': f"{self.stats.efficiency:.1f}%",
            'avg_queue_depth': self.stats.avg_queue_depth,
            'current_queue_depth': self.input_queue.qsize(),
        })

    async def register_peer_handler(self, request):
        data = await request.json()
        peer_id = data['node_id']
        peer_addr = data['address']
        self.peers[peer_id] = peer_addr
        print(f"[{self.node_id}] Registered peer: {peer_id} @ {peer_addr}")
        return web.json_response({'status': 'registered'})

    async def forward_handler(self, request):
        """
        Queue request for pipelined processing.
        Returns immediately - result sent async to next peer.
        """
        if not self.model_loaded:
            return web.json_response({'error': 'Model not loaded'}, status=503)

        data = await request.json()
        request_id = data.get('request_id', f'req-{time.time()}')

        # Parse input
        if 'tokens' in data:
            input_data = np.array(data['tokens'], dtype=np.int64)
        else:
            tensor_bytes = base64.b64decode(data['tensor'])
            input_tensor = UniversalTensor.deserialize(tensor_bytes)
            input_data = input_tensor.to_numpy()

            import os
            if os.environ.get('USE_FP32', '0') == '1':
                input_data = input_data.astype(np.float32)

        # Create request
        req = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            next_peer=data.get('next_peer'),
            final_peer=data.get('final_peer'),
        )

        # Queue for processing
        queue_start = time.perf_counter()
        await self.input_queue.put(req)
        queue_time = (time.perf_counter() - queue_start) * 1000

        # Return immediately with queue status
        return web.json_response({
            'status': 'queued',
            'request_id': request_id,
            'queue_depth': self.input_queue.qsize(),
            'queue_time_ms': queue_time,
        })

    async def start(self):
        """Start the pipelined server"""
        # Load model
        await self.load_model()

        # Start pipeline loops
        self._compute_task = asyncio.create_task(self._compute_loop())
        self._transfer_task = asyncio.create_task(self._transfer_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        # Start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

        print(f"\n{'='*60}")
        print(f"PIPELINED INFERENCE NODE STARTED")
        print(f"{'='*60}")
        print(f"Node ID: {self.node_id}")
        print(f"Address: 0.0.0.0:{self.port}")
        print(f"Model: {self.model_id}")
        print(f"Layers: {self.start_layer}-{self.end_layer}/{self.n_layers}")
        print(f"Backend: {self.backend}")
        print(f"Pipeline: ENABLED (queue size: {self.input_queue.maxsize})")
        print(f"{'='*60}")
        print(f"\nPipeline threads:")
        print(f"  - Compute loop: Running")
        print(f"  - Transfer loop: Running")
        print(f"  - Keep-alive loop: Running (30s interval)")
        print(f"{'='*60}\n")

        return runner


# Need to import aiohttp for ClientTimeout
import aiohttp


def main():
    parser = argparse.ArgumentParser(description='Pipelined Distributed Server')
    parser.add_argument('--model', type=str, default='llama-3.2-1b', help='Model ID')
    parser.add_argument('--start-layer', type=int, required=True, help='Start layer')
    parser.add_argument('--end-layer', type=int, required=True, help='End layer')
    parser.add_argument('--n-layers', type=int, default=16, help='Total layers')
    parser.add_argument('--port', type=int, default=8090, help='Port')
    parser.add_argument('--queue-size', type=int, default=8, help='Pipeline queue size')
    args = parser.parse_args()

    node = PipelinedNode(
        model_id=args.model,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        n_layers=args.n_layers,
        port=args.port,
        queue_size=args.queue_size,
    )

    async def run():
        await node.start()
        while True:
            await asyncio.sleep(3600)

    asyncio.run(run())


if __name__ == '__main__':
    main()
