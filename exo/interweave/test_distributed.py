#!/usr/bin/env python3
"""
Distributed Inference Test for Interweave Protocol

Tests real model inference across heterogeneous nodes:
- Dell C4130 (V100 CUDA) - tinygrad
- Mac Pro (FirePro OpenCL) - tinygrad
- Power8 (CPU) - numpy/llama.cpp

This test validates tensor transfer and computation across different backends.
"""

import asyncio
import time
import socket
import numpy as np
from typing import Dict, Optional, Tuple

from .tensor_format import UniversalTensor, DType
from .shard import InterweaveShard
from .backend import BackendRegistry


def get_node_info() -> Dict:
    """Get current node information"""
    hostname = socket.gethostname()
    backends = BackendRegistry.detect_available()

    # Determine primary backend
    if 'tinygrad_cuda' in backends:
        primary = 'tinygrad_cuda'
        device = 'CUDA'
    elif 'tinygrad_opencl' in backends:
        primary = 'tinygrad_opencl'
        device = 'OpenCL'
    else:
        primary = 'cpu'
        device = 'CPU'

    return {
        'hostname': hostname,
        'backends': backends,
        'primary': primary,
        'device': device,
    }


def test_local_forward_pass(hidden_size: int = 2048, seq_len: int = 32) -> Dict:
    """
    Test a simulated transformer layer forward pass on local backend.

    This mimics what happens during distributed inference:
    1. Receive input tensor (from previous node)
    2. Run computation (attention + FFN simulation)
    3. Prepare output tensor (for next node)
    """
    info = get_node_info()
    results = {'node': info, 'tests': {}}

    print(f"\n{'='*60}")
    print(f"DISTRIBUTED INFERENCE TEST - {info['hostname']}")
    print(f"Backend: {info['primary']} ({info['device']})")
    print(f"{'='*60}")

    # Test 1: Simulate receiving tensor from another node
    print("\n[1/4] Simulating tensor receive from remote node...")

    # Create input as if received over network
    input_data = np.random.randn(1, seq_len, hidden_size).astype(np.float16)
    input_tensor = UniversalTensor.from_numpy(input_data)

    # Serialize/deserialize to simulate network transfer
    start = time.perf_counter()
    wire_data = input_tensor.serialize()
    received_tensor = UniversalTensor.deserialize(wire_data)
    transfer_time = (time.perf_counter() - start) * 1000

    results['tests']['tensor_receive'] = {
        'shape': received_tensor.shape,
        'size_kb': len(wire_data) / 1024,
        'transfer_ms': transfer_time,
    }
    print(f"   Received tensor: {received_tensor.shape}, {len(wire_data)/1024:.1f}KB in {transfer_time:.2f}ms")

    # Test 2: Run computation on local backend
    print(f"\n[2/4] Running transformer layer simulation on {info['primary']}...")

    if info['primary'] in ('tinygrad_cuda', 'tinygrad_opencl'):
        try:
            from tinygrad import Tensor, Device

            # Set device
            if info['primary'] == 'tinygrad_cuda':
                device = 'CUDA'
            else:
                device = 'GPU'

            # Convert to tinygrad tensor (copy to ensure writable, use float32 for stability)
            np_input = received_tensor.to_numpy().astype(np.float32).copy()
            x = Tensor(np_input, device=device)

            # Simulate attention: Q, K, V projections + output
            # This is a simplified version - real attention is more complex
            wq = Tensor.randn(hidden_size, hidden_size, device=device)
            wk = Tensor.randn(hidden_size, hidden_size, device=device)
            wv = Tensor.randn(hidden_size, hidden_size, device=device)
            wo = Tensor.randn(hidden_size, hidden_size, device=device)

            # Warmup
            _ = (x @ wq).numpy()

            # Timed run
            start = time.perf_counter()
            q = x @ wq
            k = x @ wk
            v = x @ wv
            # Simplified attention (no softmax for speed)
            attn = q @ k.transpose(-1, -2) * (hidden_size ** -0.5)
            attn_out = attn @ v
            output = attn_out @ wo
            result = output.numpy()
            compute_time = (time.perf_counter() - start) * 1000

            results['tests']['attention_layer'] = {
                'backend': info['primary'],
                'compute_ms': compute_time,
                'output_shape': result.shape,
            }
            print(f"   Attention layer: {compute_time:.2f}ms")

            # Simulate FFN
            w1 = Tensor.randn(hidden_size, hidden_size * 4, device=device)
            w2 = Tensor.randn(hidden_size * 4, hidden_size, device=device)

            start = time.perf_counter()
            x_tensor = Tensor(result, device=device)
            ffn_hidden = x_tensor @ w1
            ffn_out = ffn_hidden.relu() @ w2
            ffn_result = ffn_out.numpy()
            ffn_time = (time.perf_counter() - start) * 1000

            results['tests']['ffn_layer'] = {
                'backend': info['primary'],
                'compute_ms': ffn_time,
                'output_shape': ffn_result.shape,
            }
            print(f"   FFN layer: {ffn_time:.2f}ms")

            final_output = ffn_result

        except Exception as e:
            results['tests']['compute_error'] = str(e)
            print(f"   Error: {e}")
            final_output = received_tensor.to_numpy()
    else:
        # CPU fallback - numpy only
        print("   Running on CPU (numpy)...")
        x = received_tensor.to_numpy().astype(np.float32).copy()

        # Simplified computation with float32 for stability
        start = time.perf_counter()
        wq = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02  # Scale down
        q = x @ wq
        k = x @ wq  # Reuse for simplicity
        v = x @ wq
        attn = q @ k.transpose(0, 2, 1) * (hidden_size ** -0.5)
        attn_out = attn @ v
        compute_time = (time.perf_counter() - start) * 1000

        results['tests']['cpu_compute'] = {
            'compute_ms': compute_time,
            'output_shape': attn_out.shape,
        }
        print(f"   CPU compute: {compute_time:.2f}ms")
        final_output = attn_out

    # Test 3: Prepare output for next node
    print("\n[3/4] Preparing output tensor for next node...")

    output_tensor = UniversalTensor.from_numpy(final_output.astype(np.float16))

    start = time.perf_counter()
    output_wire = output_tensor.serialize()
    send_time = (time.perf_counter() - start) * 1000

    results['tests']['tensor_send'] = {
        'shape': output_tensor.shape,
        'size_kb': len(output_wire) / 1024,
        'serialize_ms': send_time,
    }
    print(f"   Output tensor: {output_tensor.shape}, {len(output_wire)/1024:.1f}KB in {send_time:.2f}ms")

    # Test 4: Validate roundtrip
    print("\n[4/4] Validating tensor integrity...")

    roundtrip_tensor = UniversalTensor.deserialize(output_wire)
    original = output_tensor.to_numpy()
    recovered = roundtrip_tensor.to_numpy()

    if np.allclose(original, recovered, rtol=1e-3):
        results['tests']['integrity'] = 'PASS'
        print("   Tensor integrity: PASS")
    else:
        results['tests']['integrity'] = 'FAIL'
        print("   Tensor integrity: FAIL")

    # Summary
    total_time = (
        results['tests'].get('tensor_receive', {}).get('transfer_ms', 0) +
        results['tests'].get('attention_layer', {}).get('compute_ms', 0) +
        results['tests'].get('ffn_layer', {}).get('compute_ms', 0) +
        results['tests'].get('tensor_send', {}).get('serialize_ms', 0)
    )

    print(f"\n{'='*60}")
    print(f"TOTAL LAYER TIME: {total_time:.2f}ms")
    print(f"{'='*60}")

    results['total_ms'] = total_time
    return results


def run_distributed_test():
    """Main entry point for distributed test"""
    import json

    print("\n" + "="*60)
    print("INTERWEAVE DISTRIBUTED INFERENCE TEST")
    print("="*60)
    print("\nThis test simulates a transformer layer running on this node")
    print("as part of a distributed inference pipeline.\n")

    results = test_local_forward_pass(hidden_size=2048, seq_len=32)

    # Save results
    hostname = results['node']['hostname']
    output_file = f"distributed_test_{hostname}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    run_distributed_test()
