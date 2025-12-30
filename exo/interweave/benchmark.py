#!/usr/bin/env python3
"""
Interweave Protocol Benchmark

Tests tensor operations, serialization, and compute performance
across heterogeneous backends (CUDA, OpenCL, CPU).
"""

import json
import time
import numpy as np
import platform
import socket
import sys
from typing import Dict, List, Tuple

from .tensor_format import UniversalTensor, DType
from .shard import InterweaveShard
from .backend import BackendRegistry


def get_system_info() -> Dict[str, str]:
    """Gather system information"""
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.system(),
        'arch': platform.machine(),
        'python': platform.python_version(),
    }

    # Try to get GPU info
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            info['gpu'] = result.stdout.decode().strip().split('\n')[0]
    except:
        pass

    return info


def benchmark_tensor_creation(sizes: List[Tuple[int, ...]], iterations: int = 100) -> Dict:
    """Benchmark UniversalTensor creation from numpy"""
    results = {}

    for shape in sizes:
        times = []
        for _ in range(iterations):
            arr = np.random.randn(*shape).astype(np.float32)
            start = time.perf_counter()
            t = UniversalTensor.from_numpy(arr)
            times.append(time.perf_counter() - start)

        results[str(shape)] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'elements': np.prod(shape),
        }

    return results


def benchmark_serialization(sizes: List[Tuple[int, ...]], iterations: int = 50) -> Dict:
    """Benchmark tensor serialization/deserialization"""
    results = {}

    for shape in sizes:
        arr = np.random.randn(*shape).astype(np.float32)
        tensor = UniversalTensor.from_numpy(arr)

        # Serialize
        ser_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            data = tensor.serialize()
            ser_times.append(time.perf_counter() - start)

        # Deserialize
        deser_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            t2 = UniversalTensor.deserialize(data)
            deser_times.append(time.perf_counter() - start)

        size_mb = len(data) / (1024 * 1024)
        results[str(shape)] = {
            'serialize_ms': np.mean(ser_times) * 1000,
            'deserialize_ms': np.mean(deser_times) * 1000,
            'size_mb': size_mb,
            'throughput_mbps': size_mb / np.mean(ser_times) if np.mean(ser_times) > 0 else 0,
        }

    return results


def benchmark_dtype_conversion(size: Tuple[int, ...] = (1024, 1024), iterations: int = 20) -> Dict:
    """Benchmark dtype conversions"""
    results = {}

    arr = np.random.randn(*size).astype(np.float32)
    tensor = UniversalTensor.from_numpy(arr)

    conversions = [
        (DType.F32, DType.F16),
        (DType.F16, DType.F32),
        (DType.F32, DType.I8),
        (DType.I8, DType.F32),
    ]

    for src, dst in conversions:
        # Prepare source tensor
        if src == DType.F32:
            src_tensor = tensor
        else:
            src_tensor = tensor.convert_dtype(src)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = src_tensor.convert_dtype(dst)
            times.append(time.perf_counter() - start)

        results[f"{src.value}->{dst.value}"] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
        }

    return results


def benchmark_backend_matmul(size: int = 1024, iterations: int = 10) -> Dict:
    """Benchmark matrix multiplication on available backends"""
    results = {}
    available = BackendRegistry.detect_available()

    for backend_name in available:
        if 'cpu' in backend_name and size > 512:
            # Skip large matmuls on CPU
            test_size = 512
        else:
            test_size = size

        try:
            # Create test matrices
            a = np.random.randn(test_size, test_size).astype(np.float32)
            b = np.random.randn(test_size, test_size).astype(np.float32)

            if 'tinygrad' in backend_name:
                from tinygrad import Tensor, Device

                # Set device
                if 'cuda' in backend_name:
                    device = 'CUDA'
                elif 'opencl' in backend_name:
                    device = 'GPU'
                else:
                    device = 'CPU'

                ta = Tensor(a, device=device)
                tb = Tensor(b, device=device)

                # Warmup
                _ = (ta @ tb).numpy()

                times = []
                for _ in range(iterations):
                    ta = Tensor(a, device=device)
                    tb = Tensor(b, device=device)
                    start = time.perf_counter()
                    result = (ta @ tb).numpy()
                    times.append(time.perf_counter() - start)

                # Calculate TFLOPS
                flops = 2 * test_size ** 3  # matmul FLOPs
                tflops = flops / (np.mean(times) * 1e12)

                results[backend_name] = {
                    'size': test_size,
                    'mean_ms': np.mean(times) * 1000,
                    'std_ms': np.std(times) * 1000,
                    'tflops': tflops,
                }
            else:
                results[backend_name] = {'status': 'skipped', 'reason': 'not tinygrad'}

        except Exception as e:
            results[backend_name] = {'status': 'error', 'error': str(e)}

    return results


def benchmark_simulated_transfer(sizes: List[int], iterations: int = 20) -> Dict:
    """Simulate cross-node tensor transfer (serialize -> transfer -> deserialize)"""
    results = {}

    for size in sizes:
        shape = (size, size)
        arr = np.random.randn(*shape).astype(np.float16)  # f16 for efficiency
        tensor = UniversalTensor.from_numpy(arr)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            # Simulate: serialize -> (network would go here) -> deserialize
            data = tensor.serialize()
            t2 = UniversalTensor.deserialize(data)
            _ = t2.to_numpy()  # Ensure data is accessible
            times.append(time.perf_counter() - start)

        size_mb = len(data) / (1024 * 1024)
        results[f"{size}x{size}"] = {
            'total_ms': np.mean(times) * 1000,
            'size_mb': size_mb,
            'effective_bandwidth_mbps': size_mb / np.mean(times) if np.mean(times) > 0 else 0,
        }

    return results


def run_full_benchmark() -> Dict:
    """Run complete benchmark suite"""
    print("=" * 60)
    print("INTERWEAVE PROTOCOL BENCHMARK")
    print("=" * 60)

    results = {
        'system': get_system_info(),
        'backends': BackendRegistry.detect_available(),
        'benchmarks': {},
    }

    print(f"\nSystem: {results['system']}")
    print(f"Available backends: {results['backends']}")

    # Tensor creation
    print("\n[1/5] Benchmarking tensor creation...")
    sizes = [(64, 64), (256, 256), (1024, 1024), (2048, 2048)]
    results['benchmarks']['tensor_creation'] = benchmark_tensor_creation(sizes)

    # Serialization
    print("[2/5] Benchmarking serialization...")
    results['benchmarks']['serialization'] = benchmark_serialization(sizes)

    # Dtype conversion
    print("[3/5] Benchmarking dtype conversion...")
    results['benchmarks']['dtype_conversion'] = benchmark_dtype_conversion()

    # Backend matmul
    print("[4/5] Benchmarking backend compute...")
    results['benchmarks']['backend_matmul'] = benchmark_backend_matmul(size=1024)

    # Simulated transfer
    print("[5/5] Benchmarking simulated transfer...")
    results['benchmarks']['simulated_transfer'] = benchmark_simulated_transfer([256, 512, 1024])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Print key metrics
    print(f"\nHostname: {results['system']['hostname']}")
    print(f"Backends: {', '.join(results['backends'])}")

    if 'backend_matmul' in results['benchmarks']:
        print("\nCompute Performance:")
        for backend, data in results['benchmarks']['backend_matmul'].items():
            if 'tflops' in data:
                print(f"  {backend}: {data['tflops']:.3f} TFLOPS ({data['size']}x{data['size']} matmul)")

    if 'simulated_transfer' in results['benchmarks']:
        print("\nTransfer Overhead:")
        for size, data in results['benchmarks']['simulated_transfer'].items():
            print(f"  {size}: {data['total_ms']:.2f}ms ({data['effective_bandwidth_mbps']:.1f} MB/s)")

    return results


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    """CLI entry point"""
    results = run_full_benchmark()

    # Save results
    output_file = f"benchmark_{results['system']['hostname']}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == '__main__':
    main()
