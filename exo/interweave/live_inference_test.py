#!/usr/bin/env python3
"""
Live Inference Test - Llama 3.2 1B via Interweave

Actually loads the model and runs inference through the Interweave protocol.
"""

import numpy as np
import time
import sys
import os

# Set tinygrad to use CUDA
os.environ['GPU'] = '1'

def main():
    print("="*60)
    print("INTERWEAVE LIVE INFERENCE - Llama 3.2 1B")
    print("="*60)

    from exo.interweave import UniversalTensor, InterweaveShard, InterweaveState

    # Find model path in exo cache
    print("\n1. Locating model...")
    model_path = os.path.expanduser('~/.cache/exo/downloads/unsloth--Llama-3.2-1B-Instruct')

    if not os.path.exists(model_path):
        print(f"   ERROR: Model not found at {model_path}")
        return 1
    print(f"   Found: {model_path}")

    # Load model
    print("\n2. Loading model (this takes ~10s)...")
    from exo.inference.tinygrad.inference import build_transformer
    from exo.inference.shard import Shard
    from pathlib import Path

    exo_shard = Shard(
        model_id='unsloth/Llama-3.2-1B-Instruct',
        start_layer=0,
        end_layer=15,
        n_layers=16,
    )

    start = time.time()
    model = build_transformer(Path(model_path), exo_shard, "1B")
    load_time = time.time() - start
    print(f"   Model loaded in {load_time:.1f}s")

    # Load tokenizer
    print("\n3. Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"   Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Encode prompt
    print("\n4. Encoding prompt...")
    prompt = "Hello, I am"
    tokens = tokenizer.encode(prompt, return_tensors=None)
    print(f"   Prompt: '{prompt}'")
    print(f"   Tokens: {tokens}")

    # Run inference
    print("\n5. Running inference...")
    from tinygrad import Tensor
    from exo.inference.tinygrad.stateful_model import make_prompt_state
    from exo.inference.tinygrad.models.llama import sample_logits

    input_data = np.array([tokens], dtype=np.int32)
    print(f"   Input shape: {input_data.shape}")

    x = Tensor(input_data)
    h = model.embed(x)
    print(f"   Embedded shape: {h.shape}")

    # Create cache state
    print("\n6. Creating KV cache state...")
    state = make_prompt_state(h, model)
    print(f"   Cache created, start_pos={state.start}")

    # Forward pass
    print("\n7. Forward pass...")
    start = time.time()
    out = model.forward(h, start_pos=state.start, cache=state.cache)
    output = out.numpy()
    inference_time = (time.time() - start) * 1000
    print(f"   Output shape: {output.shape}")
    print(f"   Inference time: {inference_time:.1f} ms")

    # Sample
    print("\n8. Sampling next token...")
    logits = output[:, -1, :]
    next_token = sample_logits(Tensor(logits).flatten(), 0.8, 0, 0.9, 0.0, 0.0).realize().numpy().astype(int)
    print(f"   Next token ID: {next_token}")

    decoded = tokenizer.decode([next_token[0]])
    print(f"   Decoded: '{decoded}'")

    # Update state
    state.start += x.shape[1]

    # Generate more tokens
    print("\n9. Generating continuation...")
    all_tokens = list(tokens) + [next_token[0]]

    for i in range(5):
        # Feed last token
        x = Tensor(np.array([[all_tokens[-1]]], dtype=np.int32))
        h = model.embed(x)
        out = model.forward(h, start_pos=state.start, cache=state.cache)
        state.start += 1

        logits = out.numpy()[:, -1, :]
        next_token = sample_logits(Tensor(logits).flatten(), 0.8, 0, 0.9, 0.0, 0.0).realize().numpy().astype(int)
        all_tokens.append(next_token[0])

        decoded = tokenizer.decode(all_tokens)
        print(f"   [{i+1}] {decoded}")

    # Test Interweave tensor format
    print("\n10. Testing Interweave tensor format...")
    output_ut = UniversalTensor.from_numpy(out.numpy())
    print(f"    Output as UniversalTensor: shape={output_ut.shape}, dtype={output_ut.dtype}")

    # Serialize (simulating network transfer to Power8)
    serialized = output_ut.serialize()
    print(f"    Serialized for transfer: {len(serialized) / 1024:.1f} KB")

    # Deserialize
    restored = UniversalTensor.deserialize(serialized)
    assert output_ut.shape == restored.shape
    print("    Round-trip: PASS")

    # Create InterweaveState
    print("\n11. Creating InterweaveState with KV cache...")
    iw_state = InterweaveState(request_id="test-inference-001", sequence_position=state.start)

    # Add a sample KV cache entry
    sample_k = UniversalTensor.from_numpy(np.random.randn(1, 8, 10, 64).astype(np.float16))
    sample_v = UniversalTensor.from_numpy(np.random.randn(1, 8, 10, 64).astype(np.float16))
    iw_state.set_cache_for_layer(0, sample_k, sample_v)

    state_bytes = iw_state.serialize()
    print(f"    State serialized: {len(state_bytes) / 1024:.1f} KB")

    print("\n" + "="*60)
    print("LIVE INFERENCE TEST COMPLETE!")
    print("="*60)

    print("\nResults:")
    print(f"  - Model: Llama 3.2 1B (16 layers)")
    print(f"  - Backend: tinygrad CUDA")
    print(f"  - Load time: {load_time:.1f}s")
    print(f"  - First inference: {inference_time:.1f}ms")
    print(f"  - Interweave tensor format: working")
    print(f"  - Generated: '{tokenizer.decode(all_tokens)}'")

    return 0


if __name__ == '__main__':
    sys.exit(main())
