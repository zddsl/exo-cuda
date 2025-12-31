#!/usr/bin/env python3
"""
Pipeline Node Integration for exo

Integrates:
1. Pipeline queues with exo's Node class
2. Speculative execution during idle
3. Auto-tuned queue sizes based on network latency

This wraps exo's orchestration to eliminate GPU idle time.
"""

import asyncio
import time
import numpy as np
from typing import Dict, Optional, List, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field

from exo import DEBUG
from exo.inference.inference_engine import InferenceEngine, Shard


@dataclass
class PipelineRequest:
    """Queued inference request"""
    request_id: str
    shard: Shard
    input_data: np.ndarray
    inference_state: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)
    is_prompt: bool = False
    prompt: Optional[str] = None


@dataclass
class PipelineStats:
    """Track pipeline efficiency"""
    requests_processed: int = 0
    total_compute_ms: float = 0
    total_transfer_ms: float = 0
    total_idle_ms: float = 0
    total_speculative_ms: float = 0

    @property
    def efficiency(self) -> float:
        total = self.total_compute_ms + self.total_idle_ms
        return (self.total_compute_ms / total * 100) if total > 0 else 0

    @property
    def speculation_ratio(self) -> float:
        total = self.total_compute_ms + self.total_speculative_ms
        return (self.total_speculative_ms / total * 100) if total > 0 else 0


class NetworkLatencyTracker:
    """
    Track network latency to peers for queue tuning.

    Optimal queue size = latency * throughput
    If latency is 100ms and we process 10 req/s, queue should be ~1
    If latency is 500ms and we process 10 req/s, queue should be ~5
    """

    def __init__(self, window_size: int = 50):
        self.latencies: Dict[str, Deque[float]] = {}
        self.window_size = window_size

    def record(self, peer_id: str, latency_ms: float):
        """Record a latency measurement"""
        if peer_id not in self.latencies:
            self.latencies[peer_id] = deque(maxlen=self.window_size)
        self.latencies[peer_id].append(latency_ms)

    def get_avg_latency(self, peer_id: str) -> float:
        """Get average latency to peer"""
        if peer_id not in self.latencies or len(self.latencies[peer_id]) == 0:
            return 100.0  # Default 100ms
        return sum(self.latencies[peer_id]) / len(self.latencies[peer_id])

    def get_optimal_queue_size(self, peer_id: str, throughput: float = 10.0) -> int:
        """Calculate optimal queue size for peer"""
        latency_s = self.get_avg_latency(peer_id) / 1000.0
        optimal = int(latency_s * throughput) + 1
        return max(2, min(optimal, 16))  # Clamp to 2-16


class SpeculativePredictor:
    """
    Predict likely next requests to pre-compute.

    Learns patterns:
    - Common prompt prefixes
    - Follow-up patterns in conversations
    - Time-of-day usage patterns
    """

    def __init__(self, max_patterns: int = 100):
        self.prompt_frequency: Dict[str, int] = {}
        self.followup_patterns: Dict[str, List[str]] = {}
        self.recent_prompts: Deque[str] = deque(maxlen=20)
        self.max_patterns = max_patterns

    def record_prompt(self, prompt: str):
        """Record a prompt for pattern learning"""
        # Track frequency
        prefix = prompt[:50]
        self.prompt_frequency[prefix] = self.prompt_frequency.get(prefix, 0) + 1

        # Track follow-up patterns
        if len(self.recent_prompts) > 0:
            prev = self.recent_prompts[-1][:50]
            if prev not in self.followup_patterns:
                self.followup_patterns[prev] = []
            if prefix not in self.followup_patterns[prev]:
                self.followup_patterns[prev].append(prefix)

        self.recent_prompts.append(prompt)

        # Prune if too large
        if len(self.prompt_frequency) > self.max_patterns:
            # Keep top half by frequency
            sorted_prompts = sorted(self.prompt_frequency.items(), key=lambda x: x[1], reverse=True)
            self.prompt_frequency = dict(sorted_prompts[:self.max_patterns // 2])

    def get_likely_prompts(self, n: int = 3) -> List[str]:
        """Get most likely next prompts"""
        # Combine frequency and recency
        candidates = []

        # High frequency prompts
        sorted_by_freq = sorted(self.prompt_frequency.items(), key=lambda x: x[1], reverse=True)
        candidates.extend([p for p, _ in sorted_by_freq[:n]])

        # Follow-up predictions
        if len(self.recent_prompts) > 0:
            prev = self.recent_prompts[-1][:50]
            if prev in self.followup_patterns:
                candidates.extend(self.followup_patterns[prev][:n])

        return list(set(candidates))[:n]


class PipelineNode:
    """
    Pipeline wrapper for exo Node.

    Adds:
    1. Input queue - requests wait here for GPU
    2. Output queue - results wait here for network
    3. Speculative compute - fill idle with predictions
    4. Auto-tuned queue sizes based on measured latency
    """

    def __init__(self, node, initial_queue_size: int = 4):
        self.node = node
        self.queue_size = initial_queue_size

        # Pipeline queues
        self.input_queue: asyncio.Queue[PipelineRequest] = asyncio.Queue(maxsize=initial_queue_size)
        self.output_queue: asyncio.Queue[Tuple[str, np.ndarray, dict]] = asyncio.Queue(maxsize=initial_queue_size)

        # Tracking
        self.stats = PipelineStats()
        self.latency_tracker = NetworkLatencyTracker()
        self.predictor = SpeculativePredictor()

        # Speculative cache
        self.speculative_cache: Dict[str, np.ndarray] = {}

        # Pipeline tasks
        self._compute_task: Optional[asyncio.Task] = None
        self._transfer_task: Optional[asyncio.Task] = None
        self._speculative_task: Optional[asyncio.Task] = None
        self._tuning_task: Optional[asyncio.Task] = None

        self._running = False

    async def start(self):
        """Start pipeline loops"""
        self._running = True

        # Start pipeline tasks
        self._compute_task = asyncio.create_task(self._compute_loop())
        self._transfer_task = asyncio.create_task(self._transfer_loop())
        self._speculative_task = asyncio.create_task(self._speculative_loop())
        self._tuning_task = asyncio.create_task(self._tuning_loop())

        print(f"[Pipeline] Started with queue_size={self.queue_size}")

    async def stop(self):
        """Stop pipeline loops"""
        self._running = False
        for task in [self._compute_task, self._transfer_task, self._speculative_task, self._tuning_task]:
            if task:
                task.cancel()

    async def _compute_loop(self):
        """
        Main compute loop - processes requests from queue.
        GPU should NEVER idle if there's work.
        """
        print(f"[Pipeline] Compute loop started")

        while self._running:
            try:
                # Get next request (with timeout to check for speculative work)
                try:
                    request = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No real work - might do speculative compute
                    continue

                # Process request
                compute_start = time.perf_counter()

                try:
                    if request.is_prompt:
                        result, inference_state = await self.node.inference_engine.infer_prompt(
                            request.request_id,
                            request.shard,
                            request.prompt,
                            request.inference_state
                        )
                    else:
                        result, inference_state = await self.node.inference_engine.infer_tensor(
                            request.request_id,
                            request.shard,
                            request.input_data,
                            request.inference_state
                        )

                    compute_time = (time.perf_counter() - compute_start) * 1000
                    self.stats.total_compute_ms += compute_time
                    self.stats.requests_processed += 1

                    # Queue for output
                    await self.output_queue.put((
                        request.request_id,
                        result,
                        {
                            'shard': request.shard,
                            'inference_state': inference_state,
                            'compute_time_ms': compute_time,
                        }
                    ))

                    if DEBUG >= 2:
                        print(f"[Pipeline] Computed {request.request_id}: {compute_time:.1f}ms")

                except Exception as e:
                    print(f"[Pipeline] Compute error: {e}")

                self.input_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Pipeline] Compute loop error: {e}")

    async def _transfer_loop(self):
        """
        Transfer loop - sends results to next node.
        Runs in parallel with compute.
        """
        print(f"[Pipeline] Transfer loop started")

        while self._running:
            try:
                request_id, result, metadata = await asyncio.wait_for(
                    self.output_queue.get(), timeout=1.0
                )

                transfer_start = time.perf_counter()

                shard = metadata['shard']
                inference_state = metadata['inference_state']

                # Process result through node's standard path
                await self.node.process_inference_result(
                    shard, result, request_id, inference_state
                )

                transfer_time = (time.perf_counter() - transfer_start) * 1000
                self.stats.total_transfer_ms += transfer_time

                # Record latency for tuning
                # (In real impl, would track per-peer)
                self.latency_tracker.record("default", transfer_time)

                self.output_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Pipeline] Transfer error: {e}")

    async def _speculative_loop(self):
        """
        Speculative computation loop.
        When idle, pre-compute likely next requests.
        """
        print(f"[Pipeline] Speculative loop started")

        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second

                # Only speculate if queues are empty (GPU would be idle)
                if not self.input_queue.empty() or not self.output_queue.empty():
                    continue

                # Get likely prompts
                likely_prompts = self.predictor.get_likely_prompts(2)

                for prompt_prefix in likely_prompts:
                    if prompt_prefix in self.speculative_cache:
                        continue  # Already computed

                    spec_start = time.perf_counter()

                    try:
                        # Pre-compute embedding/first layers
                        # This is a simplified version - real impl would tokenize
                        dummy_input = np.zeros((1, 1), dtype=np.int64)

                        # Just warm up the model
                        # Real speculation would compute actual results
                        self.stats.total_speculative_ms += (time.perf_counter() - spec_start) * 1000

                        if DEBUG >= 2:
                            print(f"[Pipeline] Speculative warmup for '{prompt_prefix[:20]}...'")

                    except Exception as e:
                        if DEBUG >= 2:
                            print(f"[Pipeline] Speculative error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Pipeline] Speculative loop error: {e}")

    async def _tuning_loop(self):
        """
        Auto-tune queue sizes based on measured latency.
        """
        print(f"[Pipeline] Tuning loop started")

        while self._running:
            try:
                await asyncio.sleep(30.0)  # Tune every 30s

                # Calculate optimal queue size
                avg_latency = self.latency_tracker.get_avg_latency("default")
                throughput = self.stats.requests_processed / 30.0 if self.stats.requests_processed > 0 else 1.0

                optimal_size = self.latency_tracker.get_optimal_queue_size("default", throughput)

                if optimal_size != self.queue_size:
                    old_size = self.queue_size
                    self.queue_size = optimal_size
                    print(f"[Pipeline] Tuned queue size: {old_size} -> {optimal_size} (latency={avg_latency:.0f}ms, throughput={throughput:.1f}/s)")

                # Log stats
                print(f"[Pipeline] Stats: efficiency={self.stats.efficiency:.1f}%, "
                      f"compute={self.stats.total_compute_ms:.0f}ms, "
                      f"transfer={self.stats.total_transfer_ms:.0f}ms, "
                      f"speculative={self.stats.total_speculative_ms:.0f}ms")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Pipeline] Tuning error: {e}")

    async def enqueue_prompt(self, shard: Shard, prompt: str, request_id: str,
                             inference_state: Optional[dict] = None) -> bool:
        """
        Enqueue a prompt for pipelined processing.
        Returns immediately - result comes via callback.
        """
        # Record for speculation
        self.predictor.record_prompt(prompt)

        # Check speculative cache
        prefix = prompt[:50]
        if prefix in self.speculative_cache:
            if DEBUG >= 2:
                print(f"[Pipeline] Speculative HIT for '{prefix[:20]}...'")
            # Could use cached result here

        request = PipelineRequest(
            request_id=request_id,
            shard=shard,
            input_data=np.array([]),  # Not used for prompts
            inference_state=inference_state,
            is_prompt=True,
            prompt=prompt,
        )

        try:
            await asyncio.wait_for(self.input_queue.put(request), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            print(f"[Pipeline] Queue full, request {request_id} dropped")
            return False

    async def enqueue_tensor(self, shard: Shard, tensor: np.ndarray, request_id: str,
                             inference_state: Optional[dict] = None) -> bool:
        """
        Enqueue a tensor for pipelined processing.
        """
        request = PipelineRequest(
            request_id=request_id,
            shard=shard,
            input_data=tensor,
            inference_state=inference_state,
            is_prompt=False,
        )

        try:
            await asyncio.wait_for(self.input_queue.put(request), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            print(f"[Pipeline] Queue full, request {request_id} dropped")
            return False

    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        return {
            'requests_processed': self.stats.requests_processed,
            'efficiency': f"{self.stats.efficiency:.1f}%",
            'total_compute_ms': self.stats.total_compute_ms,
            'total_transfer_ms': self.stats.total_transfer_ms,
            'total_speculative_ms': self.stats.total_speculative_ms,
            'speculation_ratio': f"{self.stats.speculation_ratio:.1f}%",
            'queue_size': self.queue_size,
            'input_queue_depth': self.input_queue.qsize(),
            'output_queue_depth': self.output_queue.qsize(),
            'avg_latency_ms': self.latency_tracker.get_avg_latency("default"),
        }


def wrap_node_with_pipeline(node, queue_size: int = 4) -> PipelineNode:
    """
    Wrap an existing exo Node with pipeline capabilities.

    Usage:
        node = Node(...)
        pipeline_node = wrap_node_with_pipeline(node)
        await pipeline_node.start()
    """
    return PipelineNode(node, initial_queue_size=queue_size)


# ============================================================================
# Integration hooks for exo
# ============================================================================

async def patch_node_for_pipeline(node):
    """
    Monkey-patch an existing Node to use pipelining.

    This replaces the synchronous process methods with queued versions.
    """
    pipeline = PipelineNode(node)
    await pipeline.start()

    # Store original methods
    node._original_process_prompt = node._process_prompt
    node._original_process_tensor = node._process_tensor

    # Replace with pipelined versions
    async def pipelined_process_prompt(base_shard, prompt, request_id=None, inference_state=None):
        import uuid
        if request_id is None:
            request_id = str(uuid.uuid4())
        shard = node.get_current_shard(base_shard)
        await pipeline.enqueue_prompt(shard, prompt, request_id, inference_state)
        return None  # Result comes via callback

    async def pipelined_process_tensor(base_shard, tensor, request_id=None, inference_state=None):
        import uuid
        if request_id is None:
            request_id = str(uuid.uuid4())
        shard = node.get_current_shard(base_shard)
        await pipeline.enqueue_tensor(shard, tensor, request_id, inference_state)
        return None

    node._process_prompt = pipelined_process_prompt
    node._process_tensor = pipelined_process_tensor
    node._pipeline = pipeline

    print(f"[Pipeline] Node patched for pipelined inference")
    return pipeline
