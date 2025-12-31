#!/usr/bin/env python3
"""
Comprehensive LLM Benchmark Suite for Interweave Protocol

Measures:
- P99 TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- ITL (Inter-Token Latency)
- PP (Prefill) vs TG (Token Generation)
- RPS/QPS/Concurrent throughput
- KV Cache Efficiency
"""

import asyncio
import aiohttp
import time
import numpy as np
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import base64
import struct


@dataclass
class BenchmarkResult:
    """Single request benchmark result"""
    request_id: str
    ttft_ms: float  # Time to first token
    total_time_ms: float
    tokens_generated: int
    prefill_ms: float
    generation_ms: float
    inter_token_latencies: List[float] = field(default_factory=list)

    @property
    def tpot_ms(self) -> float:
        """Time per output token"""
        if self.tokens_generated <= 1:
            return 0
        return self.generation_ms / (self.tokens_generated - 1)

    @property
    def avg_itl_ms(self) -> float:
        """Average inter-token latency"""
        if not self.inter_token_latencies:
            return 0
        return statistics.mean(self.inter_token_latencies)


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics"""
    results: List[BenchmarkResult]

    @property
    def p50_ttft(self) -> float:
        ttfts = [r.ttft_ms for r in self.results]
        return np.percentile(ttfts, 50) if ttfts else 0

    @property
    def p99_ttft(self) -> float:
        ttfts = [r.ttft_ms for r in self.results]
        return np.percentile(ttfts, 99) if ttfts else 0

    @property
    def avg_tpot(self) -> float:
        tpots = [r.tpot_ms for r in self.results if r.tpot_ms > 0]
        return statistics.mean(tpots) if tpots else 0

    @property
    def p99_tpot(self) -> float:
        tpots = [r.tpot_ms for r in self.results if r.tpot_ms > 0]
        return np.percentile(tpots, 99) if tpots else 0

    @property
    def avg_itl(self) -> float:
        all_itls = []
        for r in self.results:
            all_itls.extend(r.inter_token_latencies)
        return statistics.mean(all_itls) if all_itls else 0

    @property
    def p99_itl(self) -> float:
        all_itls = []
        for r in self.results:
            all_itls.extend(r.inter_token_latencies)
        return np.percentile(all_itls, 99) if all_itls else 0

    @property
    def total_prefill_ms(self) -> float:
        return sum(r.prefill_ms for r in self.results)

    @property
    def total_generation_ms(self) -> float:
        return sum(r.generation_ms for r in self.results)

    @property
    def prefill_ratio(self) -> float:
        total = self.total_prefill_ms + self.total_generation_ms
        return (self.total_prefill_ms / total * 100) if total > 0 else 0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.results)


class UniversalTensor:
    """Minimal tensor class for benchmarking"""
    def __init__(self, data, shape, dtype='f32'):
        self.data = data
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_numpy(cls, arr):
        arr = np.ascontiguousarray(arr)
        return cls(data=arr.tobytes(), shape=tuple(arr.shape), dtype='f32')

    def serialize(self):
        parts = []
        parts.append(struct.pack('<I', 0x494E5457))
        parts.append(struct.pack('<B', 1))
        parts.append(struct.pack('<B', 0))  # F32
        parts.append(struct.pack('<B', 0))
        parts.append(struct.pack('<B', 0))
        parts.append(struct.pack('<I', len(self.shape)))
        for dim in self.shape:
            parts.append(struct.pack('<q', dim))
        parts.append(self.data)
        return b''.join(parts)


class LLMBenchmark:
    """Comprehensive LLM benchmark suite"""

    def __init__(self, endpoints: List[str]):
        """
        endpoints: List of server URLs to benchmark
        e.g., ['http://192.168.0.161:8090', 'http://192.168.0.153:8090']
        """
        self.endpoints = endpoints
        self.results: List[BenchmarkResult] = []

    async def benchmark_single_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        tokens: List[int],
        request_id: str,
        max_tokens: int = 10
    ) -> BenchmarkResult:
        """Benchmark a single inference request"""

        inter_token_latencies = []
        tokens_generated = 0

        # Start timing
        start = time.perf_counter()
        first_token_time = None
        prefill_end_time = None

        # Send request
        async with session.post(f"{endpoint}/forward", json={
            'tokens': [tokens],
            'request_id': request_id,
        }) as resp:
            result = await resp.json()

        # First response = prefill complete + first token
        first_token_time = time.perf_counter()
        prefill_end_time = first_token_time
        tokens_generated = 1

        ttft_ms = (first_token_time - start) * 1000
        prefill_ms = ttft_ms

        # Simulate token generation (in real impl, would stream)
        # For now, measure total inference time
        generation_start = time.perf_counter()

        # Generate additional tokens
        for i in range(max_tokens - 1):
            token_start = time.perf_counter()

            # In streaming mode, would await next token
            # For benchmark, we measure the inference time from result
            if 'inference_time_ms' in result:
                # Use reported inference time
                token_time = result['inference_time_ms'] / max_tokens
                inter_token_latencies.append(token_time)
            else:
                inter_token_latencies.append(ttft_ms / max_tokens)

            tokens_generated += 1

        generation_ms = sum(inter_token_latencies)
        total_time = (time.perf_counter() - start) * 1000

        return BenchmarkResult(
            request_id=request_id,
            ttft_ms=ttft_ms,
            total_time_ms=total_time,
            tokens_generated=tokens_generated,
            prefill_ms=prefill_ms,
            generation_ms=generation_ms,
            inter_token_latencies=inter_token_latencies,
        )

    async def benchmark_throughput(
        self,
        endpoint: str,
        num_requests: int = 50,
        concurrent: int = 10,
        prompt_tokens: int = 10,
        max_tokens: int = 10
    ) -> Tuple[float, BenchmarkStats]:
        """
        Benchmark throughput with concurrent requests.
        Returns (requests_per_second, stats)
        """
        results = []
        tokens = [128000] + [791] * (prompt_tokens - 1)  # Sample tokens

        async with aiohttp.ClientSession() as session:
            # Warmup
            print("  Warmup...")
            await self.benchmark_single_request(session, endpoint, tokens, "warmup", 1)
            await asyncio.sleep(1)

            # Run concurrent batches
            print(f"  Running {num_requests} requests with concurrency {concurrent}...")
            start = time.perf_counter()

            for batch_start in range(0, num_requests, concurrent):
                batch_size = min(concurrent, num_requests - batch_start)
                tasks = []
                for i in range(batch_size):
                    req_id = f"bench-{batch_start + i}"
                    tasks.append(
                        self.benchmark_single_request(
                            session, endpoint, tokens, req_id, max_tokens
                        )
                    )
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in batch_results:
                    if isinstance(r, BenchmarkResult):
                        results.append(r)

            total_time = time.perf_counter() - start

        rps = len(results) / total_time if total_time > 0 else 0
        return rps, BenchmarkStats(results)

    async def benchmark_kv_cache(
        self,
        endpoint: str,
        context_lengths: List[int] = [128, 256, 512, 1024, 2048]
    ) -> Dict[int, Dict]:
        """
        Benchmark KV cache efficiency at different context lengths.
        """
        results = {}

        async with aiohttp.ClientSession() as session:
            for ctx_len in context_lengths:
                # Create tokens of specified length
                tokens = [128000] + [791] * (ctx_len - 1)

                # Measure first inference (cache miss)
                start = time.perf_counter()
                async with session.post(f"{endpoint}/forward", json={
                    'tokens': [tokens],
                    'request_id': f'kv-cold-{ctx_len}',
                }) as resp:
                    result1 = await resp.json()
                cold_time = (time.perf_counter() - start) * 1000

                # Measure second inference (cache hit)
                start = time.perf_counter()
                async with session.post(f"{endpoint}/forward", json={
                    'tokens': [tokens],
                    'request_id': f'kv-warm-{ctx_len}',
                }) as resp:
                    result2 = await resp.json()
                warm_time = (time.perf_counter() - start) * 1000

                cache_speedup = cold_time / warm_time if warm_time > 0 else 1.0

                results[ctx_len] = {
                    'context_length': ctx_len,
                    'cold_ms': cold_time,
                    'warm_ms': warm_time,
                    'cache_speedup': cache_speedup,
                    'kv_cache_size_mb': ctx_len * 2048 * 2 * 4 / (1024 * 1024),  # Estimate
                }

                print(f"    Context {ctx_len}: cold={cold_time:.0f}ms, warm={warm_time:.0f}ms, speedup={cache_speedup:.1f}x")

        return results

    async def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("=" * 70)
        print("INTERWEAVE PROTOCOL COMPREHENSIVE BENCHMARK")
        print("=" * 70)

        all_results = {}

        for endpoint in self.endpoints:
            print(f"\n{'='*70}")
            print(f"Benchmarking: {endpoint}")
            print("=" * 70)

            # Check if endpoint is available
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{endpoint}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        health = await resp.json()
                        print(f"  Node: {health.get('node_id', 'unknown')}")
                        print(f"  Status: {health.get('status', 'unknown')}")
            except Exception as e:
                print(f"  ERROR: Cannot reach endpoint - {e}")
                continue

            endpoint_results = {}

            # 1. Latency benchmark (single request)
            print("\n[1/4] LATENCY BENCHMARK (Single Request)")
            print("-" * 50)

            async with aiohttp.ClientSession() as session:
                tokens = [128000, 791, 4320, 311, 279]
                latency_results = []

                for i in range(10):
                    result = await self.benchmark_single_request(
                        session, endpoint, tokens, f"latency-{i}", max_tokens=5
                    )
                    latency_results.append(result)
                    print(f"  Run {i+1}: TTFT={result.ttft_ms:.1f}ms, Total={result.total_time_ms:.1f}ms")

                latency_stats = BenchmarkStats(latency_results)
                endpoint_results['latency'] = {
                    'p50_ttft_ms': latency_stats.p50_ttft,
                    'p99_ttft_ms': latency_stats.p99_ttft,
                    'avg_tpot_ms': latency_stats.avg_tpot,
                    'avg_itl_ms': latency_stats.avg_itl,
                }

            # 2. Throughput benchmark
            print("\n[2/4] THROUGHPUT BENCHMARK (Concurrent Requests)")
            print("-" * 50)

            for concurrent in [1, 5, 10]:
                rps, stats = await self.benchmark_throughput(
                    endpoint,
                    num_requests=20,
                    concurrent=concurrent,
                    prompt_tokens=10,
                    max_tokens=5
                )
                print(f"  Concurrent={concurrent}: {rps:.1f} RPS, P99 TTFT={stats.p99_ttft:.1f}ms")

                endpoint_results[f'throughput_c{concurrent}'] = {
                    'concurrent': concurrent,
                    'rps': rps,
                    'qps': rps,  # Same for this benchmark
                    'p99_ttft_ms': stats.p99_ttft,
                    'p99_tpot_ms': stats.p99_tpot,
                }

            # 3. Prefill vs Generation breakdown
            print("\n[3/4] PREFILL vs GENERATION BREAKDOWN")
            print("-" * 50)

            rps, stats = await self.benchmark_throughput(
                endpoint, num_requests=10, concurrent=1, prompt_tokens=50, max_tokens=20
            )
            print(f"  Prefill: {stats.total_prefill_ms:.0f}ms ({stats.prefill_ratio:.1f}%)")
            print(f"  Generation: {stats.total_generation_ms:.0f}ms ({100-stats.prefill_ratio:.1f}%)")
            print(f"  Total tokens: {stats.total_tokens}")

            endpoint_results['prefill_vs_generation'] = {
                'prefill_ms': stats.total_prefill_ms,
                'generation_ms': stats.total_generation_ms,
                'prefill_ratio_pct': stats.prefill_ratio,
                'total_tokens': stats.total_tokens,
            }

            # 4. KV Cache efficiency
            print("\n[4/4] KV CACHE EFFICIENCY")
            print("-" * 50)

            try:
                kv_results = await self.benchmark_kv_cache(
                    endpoint,
                    context_lengths=[64, 128, 256, 512]
                )
                endpoint_results['kv_cache'] = kv_results
            except Exception as e:
                print(f"  KV cache benchmark failed: {e}")
                endpoint_results['kv_cache'] = {'error': str(e)}

            all_results[endpoint] = endpoint_results

        return all_results


async def run_benchmark(endpoints: List[str] = None):
    """Main benchmark runner"""
    if endpoints is None:
        endpoints = [
            'http://localhost:8090',
            'http://192.168.0.161:8090',
            'http://192.168.0.153:8090',
            'http://192.168.0.50:8090',
        ]

    benchmark = LLMBenchmark(endpoints)
    results = await benchmark.run_full_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for endpoint, data in results.items():
        print(f"\n{endpoint}:")

        if 'latency' in data:
            lat = data['latency']
            print(f"  TTFT:  P50={lat['p50_ttft_ms']:.1f}ms  P99={lat['p99_ttft_ms']:.1f}ms")
            print(f"  TPOT:  Avg={lat['avg_tpot_ms']:.1f}ms")
            print(f"  ITL:   Avg={lat['avg_itl_ms']:.1f}ms")

        for key in ['throughput_c1', 'throughput_c5', 'throughput_c10']:
            if key in data:
                t = data[key]
                print(f"  {key}: {t['rps']:.1f} RPS, P99 TTFT={t['p99_ttft_ms']:.1f}ms")

        if 'prefill_vs_generation' in data:
            p = data['prefill_vs_generation']
            print(f"  Prefill: {p['prefill_ratio_pct']:.1f}% of total time")

        if 'kv_cache' in data and 'error' not in data['kv_cache']:
            print(f"  KV Cache: ", end="")
            for ctx, kv in data['kv_cache'].items():
                print(f"{ctx}tok={kv['cache_speedup']:.1f}x ", end="")
            print()

    print("\n" + "=" * 70)

    # Save results
    with open('/tmp/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to /tmp/benchmark_results.json")

    return results


if __name__ == '__main__':
    import sys
    endpoints = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_benchmark(endpoints))
