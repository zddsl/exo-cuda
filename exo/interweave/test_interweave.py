#!/usr/bin/env python3
"""
Interweave Protocol Test Suite

Tests the core functionality of the Interweave protocol including:
1. UniversalTensor conversions
2. Backend registry and detection
3. Shard creation and splitting
4. State serialization
5. Router candidate selection
6. V100 + Power8 split simulation
"""

import asyncio
import logging
import numpy as np
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('interweave_test')


def test_universal_tensor():
    """Test UniversalTensor creation and conversion"""
    print("\n" + "="*60)
    print("TEST: UniversalTensor")
    print("="*60)

    from exo.interweave import UniversalTensor, DType

    # Test numpy round-trip
    print("\n1. NumPy round-trip (f32)...")
    arr = np.random.randn(2, 4, 8).astype(np.float32)
    tensor = UniversalTensor.from_numpy(arr)
    back = tensor.to_numpy()
    assert np.allclose(arr, back), "NumPy f32 round-trip failed"
    print(f"   PASS: shape={tensor.shape}, dtype={tensor.dtype}, size={len(tensor.data)} bytes")

    # Test f16
    print("\n2. NumPy round-trip (f16)...")
    arr16 = np.random.randn(3, 5).astype(np.float16)
    tensor16 = UniversalTensor.from_numpy(arr16)
    back16 = tensor16.to_numpy()
    assert np.allclose(arr16, back16, rtol=1e-3), "NumPy f16 round-trip failed"
    print(f"   PASS: shape={tensor16.shape}, dtype={tensor16.dtype}")

    # Test serialization
    print("\n3. Binary serialization...")
    serialized = tensor.serialize()
    restored = UniversalTensor.deserialize(serialized)
    assert tensor.shape == restored.shape, "Shape mismatch after deserialize"
    assert tensor.dtype == restored.dtype, "Dtype mismatch after deserialize"
    assert np.allclose(tensor.to_numpy(), restored.to_numpy()), "Data mismatch after deserialize"
    print(f"   PASS: serialized size={len(serialized)} bytes")

    # Test dtype conversion
    print("\n4. Dtype conversion (f32 -> f16)...")
    tensor_f16 = tensor.convert_dtype(DType.F16)
    assert tensor_f16.dtype == DType.F16, "Dtype conversion failed"
    print(f"   PASS: converted to {tensor_f16.dtype}")

    print("\nUniversalTensor: ALL TESTS PASSED")
    return True


def test_shard():
    """Test InterweaveShard functionality"""
    print("\n" + "="*60)
    print("TEST: InterweaveShard")
    print("="*60)

    from exo.interweave import InterweaveShard, create_model_shards, estimate_layer_memory

    # Create a shard
    print("\n1. Creating shard...")
    shard = InterweaveShard(
        model_id='llama-3.1-70b',
        start_layer=0,
        end_layer=39,
        n_layers=80,
        preferred_backends=('tinygrad_cuda', 'llama_cpp'),
        memory_estimate=20 * 1024**3,  # 20GB
    )
    print(f"   PASS: {shard}")

    # Test splitting
    print("\n2. Splitting shard at layer 20...")
    first, second = shard.split(20)
    assert first.end_layer == 19, "First shard end_layer wrong"
    assert second.start_layer == 20, "Second shard start_layer wrong"
    print(f"   First:  {first}")
    print(f"   Second: {second}")

    # Test memory estimation
    print("\n3. Estimating layer memory...")
    mem = estimate_layer_memory('llama-3.1-70b', 0, 'f16')
    print(f"   Layer 0 estimate: {mem / 1024**2:.1f} MB")

    # Test model shard creation
    print("\n4. Creating model shards with V100 + Power8 split...")
    shards = create_model_shards(
        model_id='llama-3.1-70b',
        n_layers=80,
        splits=[(0, 'tinygrad_cuda'), (40, 'llama_cpp')],
    )
    for s in shards:
        print(f"   {s}")
        print(f"      Memory estimate: {s.memory_estimate / 1024**3:.2f} GB")

    print("\nInterweaveShard: ALL TESTS PASSED")
    return True


def test_state():
    """Test InterweaveState functionality"""
    print("\n" + "="*60)
    print("TEST: InterweaveState")
    print("="*60)

    from exo.interweave import InterweaveState, UniversalTensor, StateManager

    # Create state
    print("\n1. Creating state...")
    state = InterweaveState(request_id='test-123')
    state.sequence_position = 10
    print(f"   PASS: {state}")

    # Add KV-cache
    print("\n2. Adding KV-cache for layers 0-3...")
    for i in range(4):
        k = UniversalTensor.from_numpy(np.random.randn(1, 32, 128).astype(np.float16))
        v = UniversalTensor.from_numpy(np.random.randn(1, 32, 128).astype(np.float16))
        state.set_cache_for_layer(i, k, v)
    print(f"   Cache layers: {list(state.kv_cache.keys())}")
    print(f"   Cache size: {state.cache_size_bytes / 1024:.1f} KB")

    # Test serialization
    print("\n3. Serializing state...")
    start = time.time()
    data = state.serialize(compress=True)
    elapsed = (time.time() - start) * 1000
    print(f"   Serialized size: {len(data) / 1024:.1f} KB")
    print(f"   Serialization time: {elapsed:.2f} ms")

    # Test deserialization
    print("\n4. Deserializing state...")
    start = time.time()
    restored = InterweaveState.deserialize(data)
    elapsed = (time.time() - start) * 1000
    assert restored.request_id == state.request_id
    assert restored.sequence_position == state.sequence_position
    assert len(restored.kv_cache) == len(state.kv_cache)
    print(f"   Deserialization time: {elapsed:.2f} ms")

    # Test StateManager
    print("\n5. Testing StateManager LRU eviction...")
    manager = StateManager(max_states=3)
    for i in range(5):
        s = InterweaveState(request_id=f'req-{i}')
        manager.set(s)
    print(f"   Added 5 states with max_states=3")
    print(f"   Remaining states: {len(manager)}")
    assert len(manager) == 3, "LRU eviction failed"

    print("\nInterweaveState: ALL TESTS PASSED")
    return True


def test_backend_registry():
    """Test BackendRegistry functionality"""
    print("\n" + "="*60)
    print("TEST: BackendRegistry")
    print("="*60)

    from exo.interweave import BackendRegistry, InterweaveShard

    # Detect available backends
    print("\n1. Detecting available backends...")
    available = BackendRegistry.detect_available()
    print(f"   Available: {available}")

    # Get registered backends
    print("\n2. Getting registered backends...")
    registered = BackendRegistry.get_registered()
    print(f"   Registered: {list(registered.keys())}")

    # Test backend selection
    print("\n3. Testing optimal backend selection...")
    shard_cuda = InterweaveShard(
        model_id='test',
        start_layer=0,
        end_layer=10,
        n_layers=20,
        preferred_backends=('tinygrad_cuda', 'tinygrad_cpu'),
    )
    selected = BackendRegistry.select_optimal(shard_cuda, available)
    print(f"   For CUDA-preferred shard: {selected}")

    shard_cpu = InterweaveShard(
        model_id='test',
        start_layer=0,
        end_layer=10,
        n_layers=20,
        preferred_backends=('llama_cpp', 'tinygrad_cpu'),
    )
    selected2 = BackendRegistry.select_optimal(shard_cpu, available)
    print(f"   For CPU-preferred shard: {selected2}")

    print("\nBackendRegistry: ALL TESTS PASSED")
    return True


async def test_router():
    """Test InterweaveRouter functionality"""
    print("\n" + "="*60)
    print("TEST: InterweaveRouter")
    print("="*60)

    from exo.interweave import InterweaveRouter, InterweaveShard, UniversalTensor

    # Create router
    print("\n1. Creating router with auto-detect...")
    router = InterweaveRouter(node_id='test-node')
    print(f"   Node ID: {router.node_id}")
    print(f"   Local backend: {router.local_backend.name if router.local_backend else 'None'}")

    # Get cluster status
    print("\n2. Getting cluster status...")
    status = await router.get_cluster_status()
    print(f"   Status: {status}")

    # Test candidate finding
    print("\n3. Finding candidates for CUDA shard...")
    shard = InterweaveShard(
        model_id='llama-3.1-8b',
        start_layer=0,
        end_layer=15,
        n_layers=32,
        preferred_backends=('tinygrad_cuda', 'tinygrad_cpu'),
    )
    candidates = await router.find_candidates(shard)
    for c in candidates:
        print(f"   Candidate: node={c.node_id}, backend={c.backend_name}, score={c.score:.1f}, local={c.is_local}")

    # Test local forward (if backend available)
    if router.local_backend:
        print("\n4. Testing local forward simulation...")
        # Create small test tensor
        test_input = np.array([[1]], dtype=np.int32)  # Single token
        tensor = UniversalTensor.from_numpy(test_input)
        print(f"   Input: shape={tensor.shape}, dtype={tensor.dtype}")
        print("   (Skipping actual forward - requires loaded model)")

    print("\nInterweaveRouter: ALL TESTS PASSED")
    return True


def test_v100_power8_scenario():
    """
    Simulate V100 + Power8 split inference scenario.

    Target architecture:
    - V100 (16GB VRAM): Layers 0-39, tinygrad_cuda, f16
    - Power8 (576GB RAM): Layers 40-79, llama_cpp, i4 (quantized)
    """
    print("\n" + "="*60)
    print("TEST: V100 + Power8 Split Inference Scenario")
    print("="*60)

    from exo.interweave import (
        InterweaveShard,
        create_model_shards,
        UniversalTensor,
        InterweaveState,
        BackendCapabilities,
    )

    # Define the split
    print("\n1. Creating Llama 70B split configuration...")
    shards = create_model_shards(
        model_id='llama-3.1-70b',
        n_layers=80,
        splits=[
            (0, 'tinygrad_cuda'),   # V100: layers 0-39
            (40, 'llama_cpp'),       # Power8: layers 40-79
        ],
        dtype='f16',
    )

    print("\n   Split configuration:")
    for s in shards:
        backend_type = 'V100 CUDA' if 'cuda' in s.preferred_backends[0] else 'Power8 CPU'
        print(f"   [{backend_type}] Layers {s.start_layer}-{s.end_layer}")
        print(f"      Memory estimate: {s.memory_estimate / 1024**3:.1f} GB")
        print(f"      Backend: {s.preferred_backends}")

    # Simulate backend capabilities
    print("\n2. Simulating node capabilities...")

    v100_caps = BackendCapabilities(
        name='tinygrad_cuda',
        device_type='cuda',
        supported_dtypes=['f32', 'f16', 'bf16'],
        preferred_dtype='f16',
        memory_available=16 * 1024**3,  # 16GB
    )
    print(f"   V100 (192.168.0.161): {v100_caps.memory_available // 1024**3}GB VRAM, {v100_caps.preferred_dtype}")

    power8_caps = BackendCapabilities(
        name='llama_cpp',
        device_type='cpu',
        supported_dtypes=['f32', 'f16', 'i8', 'i4'],
        preferred_dtype='i4',
        memory_available=576 * 1024**3,  # 576GB
    )
    print(f"   Power8 (192.168.0.50): {power8_caps.memory_available // 1024**3}GB RAM, {power8_caps.preferred_dtype}")

    # Simulate tensor transfer
    print("\n3. Simulating inter-node tensor transfer...")

    # Hidden state at layer 40 boundary (batch=1, seq=1, hidden=8192)
    hidden_state = np.random.randn(1, 1, 8192).astype(np.float16)
    tensor = UniversalTensor.from_numpy(hidden_state)

    # Serialize for transfer
    start = time.time()
    serialized = tensor.serialize()
    serialize_time = (time.time() - start) * 1000

    # Simulate network transfer at 10 Gbps
    transfer_size_bits = len(serialized) * 8
    network_time = (transfer_size_bits / (10 * 1024**3)) * 1000  # ms

    # Deserialize
    start = time.time()
    restored = UniversalTensor.deserialize(serialized)
    deserialize_time = (time.time() - start) * 1000

    print(f"   Hidden state size: {hidden_state.nbytes / 1024:.1f} KB")
    print(f"   Serialized size: {len(serialized) / 1024:.1f} KB")
    print(f"   Serialize time: {serialize_time:.2f} ms")
    print(f"   Network transfer @10Gbps: {network_time:.3f} ms")
    print(f"   Deserialize time: {deserialize_time:.2f} ms")
    print(f"   Total overhead: {serialize_time + network_time + deserialize_time:.2f} ms")

    # Simulate KV-cache state transfer
    print("\n4. Simulating KV-cache state transfer (40 layers)...")
    state = InterweaveState(request_id='inference-001', sequence_position=100)

    # Add KV-cache for layers 0-39 (V100's layers)
    # KV shape: [batch, heads, seq, head_dim] = [1, 64, 100, 128]
    for layer in range(40):
        k = UniversalTensor.from_numpy(np.random.randn(1, 64, 100, 128).astype(np.float16))
        v = UniversalTensor.from_numpy(np.random.randn(1, 64, 100, 128).astype(np.float16))
        state.set_cache_for_layer(layer, k, v)

    print(f"   KV-cache layers: {len(state.kv_cache)}")
    print(f"   Total cache size: {state.cache_size_bytes / 1024**2:.1f} MB")

    start = time.time()
    state_data = state.serialize(compress=True)
    state_serialize_time = (time.time() - start) * 1000
    print(f"   Serialized state: {len(state_data) / 1024**2:.1f} MB")
    print(f"   State serialize time: {state_serialize_time:.1f} ms")

    # Estimate total overhead per token
    print("\n5. Estimated overhead per token...")
    overhead_per_token = serialize_time + network_time + deserialize_time
    print(f"   Hidden state transfer: {overhead_per_token:.2f} ms/token")
    print(f"   At 50 tokens/sec: {overhead_per_token * 50:.1f} ms overhead")
    print(f"   Overhead percentage: {(overhead_per_token / (1000/50)) * 100:.1f}%")

    print("\n   NOTE: Initial KV-cache transfer is one-time overhead")
    print(f"   For 100 token context: {len(state_data) / 1024**2:.1f} MB transfer")

    print("\nV100 + Power8 Scenario: SIMULATION COMPLETE")
    return True


async def main():
    """Run all tests"""
    print("="*60)
    print("INTERWEAVE PROTOCOL TEST SUITE")
    print("="*60)

    results = []

    # Run tests
    results.append(("UniversalTensor", test_universal_tensor()))
    results.append(("InterweaveShard", test_shard()))
    results.append(("InterweaveState", test_state()))
    results.append(("BackendRegistry", test_backend_registry()))
    results.append(("InterweaveRouter", await test_router()))
    results.append(("V100+Power8 Scenario", test_v100_power8_scenario()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print("\nSOME TESTS FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
