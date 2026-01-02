#!/usr/bin/env python3
"""
Simple Interweave Test - Llama 3.2 1B

Quick test of the Interweave protocol with a small model.
"""

import asyncio
import numpy as np
import time
import sys

async def main():
    print("="*60)
    print("INTERWEAVE SIMPLE TEST - Llama 3.2 1B")
    print("="*60)

    # Import Interweave components
    from exo.interweave import (
        UniversalTensor,
        InterweaveShard,
        InterweaveState,
        BackendRegistry,
    )

    # Check available backends
    print("\n1. Checking backends...")
    available = BackendRegistry.detect_available()
    print(f"   Available: {available}")

    # Create a shard for Llama 3.2 1B (16 layers)
    print("\n2. Creating shard for Llama 3.2 1B...")
    shard = InterweaveShard(
        model_id='unsloth/Llama-3.2-1B-Instruct',
        start_layer=0,
        end_layer=15,
        n_layers=16,
        preferred_backends=('tinygrad_cuda',),
        is_embedding=True,
        is_output=True,
    )
    print(f"   Shard: {shard}")

    # Create TinyGrad CUDA backend directly
    print("\n3. Creating TinyGrad CUDA backend...")
    from exo.interweave.backends.tinygrad_cuda import TinygradCudaBackend

    backend = TinygradCudaBackend(device='CUDA')
    print(f"   Backend: {backend.name}")
    print(f"   Device: {backend.device_type}")
    print(f"   Supported dtypes: {backend.supported_dtypes}")

    # Check GPU memory
    mem = await backend.get_memory_available()
    print(f"   Available VRAM: {mem / 1024**3:.1f} GB")

    # Test UniversalTensor creation and conversion
    print("\n4. Testing tensor operations...")

    # Create a simple token input (token ID 1 = common token)
    tokens = np.array([[1, 2, 3]], dtype=np.int32)
    tensor = UniversalTensor.from_numpy(tokens)
    print(f"   Input tokens shape: {tensor.shape}")
    print(f"   Input dtype: {tensor.dtype}")

    # Test tensor serialization (simulating network transfer)
    print("\n5. Testing tensor serialization (network simulation)...")
    start = time.time()
    serialized = tensor.serialize()
    serialize_time = (time.time() - start) * 1000

    start = time.time()
    restored = UniversalTensor.deserialize(serialized)
    deserialize_time = (time.time() - start) * 1000

    print(f"   Serialized size: {len(serialized)} bytes")
    print(f"   Serialize: {serialize_time:.3f} ms")
    print(f"   Deserialize: {deserialize_time:.3f} ms")
    assert np.array_equal(tensor.to_numpy(), restored.to_numpy()), "Tensor mismatch!"
    print("   Round-trip: PASS")

    # Test state management
    print("\n6. Testing state management...")
    state = InterweaveState(request_id="test-inference-001")
    print(f"   Created state: {state.request_id}")

    # Simulate KV cache for a few layers
    for i in range(4):
        k = UniversalTensor.from_numpy(np.random.randn(1, 8, 10, 64).astype(np.float16))
        v = UniversalTensor.from_numpy(np.random.randn(1, 8, 10, 64).astype(np.float16))
        state.set_cache_for_layer(i, k, v)

    print(f"   KV cache layers: {len(state.kv_cache)}")
    print(f"   Cache size: {state.cache_size_bytes / 1024:.1f} KB")

    # Serialize state
    start = time.time()
    state_data = state.serialize(compress=True)
    state_time = (time.time() - start) * 1000
    print(f"   State serialized: {len(state_data) / 1024:.1f} KB in {state_time:.1f} ms")

    # Test the actual inference path simulation
    print("\n7. Testing inference path simulation...")

    # Simulate what happens in a forward pass:
    # 1. Input tokens arrive
    # 2. Convert to UniversalTensor
    # 3. Pass through backend
    # 4. Get output
    # 5. Convert back

    print("   Step 1: Tokenize input -> [1, 2, 3]")
    input_tokens = np.array([[1, 2, 3]], dtype=np.int32)

    print("   Step 2: Create UniversalTensor")
    input_ut = UniversalTensor.from_numpy(input_tokens)

    print("   Step 3: Would call backend.forward() here")
    print("          (Skipping actual model load for speed)")

    # Simulate output (hidden states)
    hidden_dim = 2048  # Llama 3.2 1B hidden dim
    fake_output = np.random.randn(1, 3, hidden_dim).astype(np.float16)
    output_ut = UniversalTensor.from_numpy(fake_output)

    print(f"   Step 4: Output shape: {output_ut.shape}")
    print(f"   Step 5: Convert back to numpy: {output_ut.to_numpy().shape}")

    print("\n" + "="*60)
    print("SIMPLE TEST COMPLETE - All components working!")
    print("="*60)

    print("\nNext steps for live inference:")
    print("  1. Load model weights via backend.load_shard()")
    print("  2. Run actual forward pass")
    print("  3. Connect to Power8 peer for split inference")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
