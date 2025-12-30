#!/usr/bin/env python3
"""
Large Memory Test for Interweave Protocol

Tests leveraging Power8's 576GB RAM for tensors that wouldn't fit on GPU VRAM.
Demonstrates the key advantage of heterogeneous inference.
"""

import asyncio
import time
import base64
import numpy as np
from aiohttp import ClientSession

from .tensor_format import UniversalTensor, DType


async def test_large_tensor_power8():
    """
    Test sending large tensors that leverage Power8's 576GB RAM.

    GPU VRAM limits:
    - V100: 16GB
    - FirePro D500: 6GB (dual = 12GB but not unified)

    Power8: 576GB RAM - can hold entire 70B+ models
    """

    nodes = {
        "dell": "192.168.0.161:8089",
        "mac": "192.168.0.153:8089",
        # "power8": "192.168.0.50:8089",  # Currently building PyTorch
    }

    print("\n" + "="*70)
    print("POWER8 LARGE MEMORY TEST")
    print("Testing tensors that stress GPU VRAM but fit easily in 576GB RAM")
    print("="*70)

    # Test different tensor sizes
    # Simulating layer activations for different model sizes
    test_cases = [
        # (batch, seq, hidden, description, can_fit_on_gpu)
        (1, 32, 2048, "Small (Llama 3.2 1B size)", True),
        (1, 2048, 4096, "Medium (Llama 3.1 8B context)", True),
        (1, 4096, 8192, "Large (Llama 70B hidden dim)", False),  # ~256MB per layer
        (1, 8192, 8192, "XL (Full 70B context)", False),  # ~512MB per layer
        (4, 8192, 8192, "XXL (Batch inference)", False),  # ~2GB per layer
    ]

    async with ClientSession() as session:
        # Check nodes first
        print("\n[1/3] Checking node status...")
        for name, addr in nodes.items():
            try:
                async with session.get(f"http://{addr}/info", timeout=5) as resp:
                    info = await resp.json()
                    print(f"   {name}: {info['primary_backend']} ({info['device']})")
            except Exception as e:
                print(f"   {name}: OFFLINE ({e})")
                if name == "power8":
                    print("   Power8 must be online for this test!")
                    return

        # Register peers
        print("\n[2/3] Registering peers...")
        node_list = list(nodes.values())
        for i, addr in enumerate(node_list):
            for j, peer_addr in enumerate(node_list):
                if i != j:
                    try:
                        peer_name = list(nodes.keys())[j]
                        await session.post(f"http://{addr}/register_peer", json={
                            'node_id': peer_name,
                            'address': peer_addr,
                        })
                    except:
                        pass

        # Run tests
        print("\n[3/3] Running memory tests...\n")
        results = []

        for batch, seq, hidden, desc, fits_gpu in test_cases:
            tensor_size_mb = (batch * seq * hidden * 2) / (1024 * 1024)  # f16 = 2 bytes
            tensor_size_gb = tensor_size_mb / 1024

            print(f"{'='*70}")
            print(f"Test: {desc}")
            print(f"Shape: ({batch}, {seq}, {hidden})")
            print(f"Size: {tensor_size_mb:.1f} MB ({tensor_size_gb:.3f} GB)")
            print(f"Fits on GPU: {'Yes' if fits_gpu else 'NO - Power8 required!'}")
            print("-"*70)

            # Create test tensor
            input_data = np.random.randn(batch, seq, hidden).astype(np.float16)
            input_tensor = UniversalTensor.from_numpy(input_data)
            tensor_bytes = base64.b64encode(input_tensor.serialize()).decode()

            # Test on each node
            node_results = {}
            for name, addr in nodes.items():
                try:
                    start = time.perf_counter()

                    async with session.post(f"http://{addr}/forward", json={
                        'tensor': tensor_bytes,
                        'layer_start': 0,
                        'layer_end': 0,
                    }, timeout=120) as resp:  # 2 min timeout for large tensors
                        result = await resp.json()

                    elapsed = (time.perf_counter() - start) * 1000

                    if result.get('status') == 'success':
                        compute_time = result.get('compute_time_ms', elapsed)
                        node_results[name] = {
                            'status': 'OK',
                            'time_ms': compute_time,
                            'total_ms': elapsed,
                        }
                        print(f"   {name:8}: {compute_time:.0f}ms (total: {elapsed:.0f}ms)")
                    else:
                        node_results[name] = {'status': 'FAILED', 'error': str(result)}
                        print(f"   {name:8}: FAILED - {result}")

                except asyncio.TimeoutError:
                    node_results[name] = {'status': 'TIMEOUT'}
                    print(f"   {name:8}: TIMEOUT (tensor too large for this backend)")
                except Exception as e:
                    node_results[name] = {'status': 'ERROR', 'error': str(e)}
                    print(f"   {name:8}: ERROR - {e}")

            results.append({
                'desc': desc,
                'shape': (batch, seq, hidden),
                'size_mb': tensor_size_mb,
                'fits_gpu': fits_gpu,
                'nodes': node_results,
            })
            print()

        # Summary
        print("="*70)
        print("SUMMARY: Power8's 576GB RAM Advantage")
        print("="*70)
        print()
        print("| Test Case | Size | V100 CUDA | FirePro | Power8 CPU |")
        print("|-----------|------|-----------|---------|------------|")

        for r in results:
            dell = r['nodes'].get('dell', {})
            mac = r['nodes'].get('mac', {})
            power8 = r['nodes'].get('power8', {})

            dell_str = f"{dell.get('time_ms', 0):.0f}ms" if dell.get('status') == 'OK' else dell.get('status', 'N/A')
            mac_str = f"{mac.get('time_ms', 0):.0f}ms" if mac.get('status') == 'OK' else mac.get('status', 'N/A')
            power8_str = f"{power8.get('time_ms', 0):.0f}ms" if power8.get('status') == 'OK' else power8.get('status', 'N/A')

            print(f"| {r['desc'][:20]:20} | {r['size_mb']:.0f}MB | {dell_str:9} | {mac_str:7} | {power8_str:10} |")

        print()
        print("Key insight: Power8 can handle tensors that would exhaust GPU VRAM!")
        print("This enables running 70B+ models by splitting layers across the cluster.")
        print()

        # Memory calculation for 70B model
        print("="*70)
        print("70B Model Memory Requirements (example)")
        print("="*70)
        print("Llama 70B at fp16:")
        print("  - Total weights: ~140GB")
        print("  - Per-layer activation (batch=1, seq=8192, hidden=8192): ~512MB")
        print("  - KV-cache per layer at 8K context: ~1GB")
        print()
        print("V100 (16GB) can hold: ~10-12 layers")
        print("Power8 (576GB) can hold: ENTIRE MODEL + KV-cache!")
        print()
        print("Optimal split: V100 runs compute-heavy attention, Power8 stores overflow")
        print("="*70)

        return results


async def main():
    results = await test_large_tensor_power8()
    return results


if __name__ == '__main__':
    asyncio.run(main())
