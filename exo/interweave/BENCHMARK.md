# Interweave Protocol Benchmark Results

Benchmarks run on 2024-12-30 across a heterogeneous 3-node cluster.

## Cluster Configuration

| Node | Hostname | Hardware | Backend | Memory |
|------|----------|----------|---------|--------|
| Dell C4130 | sophiacord2-PowerEdge-C4130 | Tesla V100 16GB + M40 12GB | `tinygrad_cuda` | 128GB RAM |
| Mac Pro | Sophias-Mac-Trashcan.local | Dual AMD FirePro D500 6GB | `tinygrad_opencl` | 64GB RAM |
| IBM Power8 | ubuntu | POWER8 20-core | `llama_cpp`, `tinygrad_cpu` | 576GB RAM |

## Compute Performance (1024x1024 MatMul)

| Node | Backend | TFLOPS | Notes |
|------|---------|--------|-------|
| **Dell C4130** | tinygrad_cuda | **0.284** | V100 CUDA |
| **Mac Pro** | tinygrad_opencl | **0.141** | FirePro D500 OpenCL |
| **Power8** | llama_cpp | N/A | CPU-only, optimized for quantized inference |

## Tensor Transfer Overhead

Simulated cross-node transfer (serialize → deserialize cycle):

| Tensor Size | Dell C4130 | Mac Pro | Power8 |
|-------------|------------|---------|--------|
| 256x256 (128KB) | 0.04ms (3.2 GB/s) | 0.08ms (1.6 GB/s) | 0.04ms (2.8 GB/s) |
| 512x512 (512KB) | 0.10ms (5.1 GB/s) | 0.26ms (1.9 GB/s) | 0.07ms (7.3 GB/s) |
| 1024x1024 (2MB) | 0.42ms (4.8 GB/s) | 0.55ms (3.6 GB/s) | 0.47ms (4.3 GB/s) |

## Serialization Performance

UniversalTensor serialize/deserialize throughput:

| Node | Serialize | Deserialize |
|------|-----------|-------------|
| Dell C4130 | 7.6 GB/s | 6.7 GB/s |
| Mac Pro | 4.0 GB/s | 4.1 GB/s |
| Power8 | 8.1 GB/s | 5.4 GB/s |

## Dtype Conversion (1024x1024 tensor)

| Conversion | Dell C4130 | Mac Pro | Power8 |
|------------|------------|---------|--------|
| f32 → f16 | 6.1ms | 5.7ms | 6.6ms |
| f16 → f32 | 3.9ms | 3.2ms | 3.7ms |
| f32 → i8 (quantize) | 2.2ms | 3.0ms | 27.5ms |
| i8 → f32 (dequantize) | 2.1ms | 1.4ms | 2.6ms |

## Key Findings

1. **CUDA is fastest** for compute-intensive operations (V100 = 2x FirePro D500)
2. **Power8 excels at memory bandwidth** - 576GB RAM ideal for large model sharding
3. **Serialization overhead is minimal** - <0.5ms for 2MB tensors
4. **OpenCL works on AMD GPUs** via tinygrad's GPU backend
5. **gRPC is optional** - Power8 runs without gRPC (ppc64le compatibility)

## Distributed Inference Strategy

Based on benchmarks, optimal shard distribution for 70B model:

```
┌─────────────────────────┐     ┌─────────────────────────┐
│    Dell C4130 (V100)    │     │    Power8 (576GB RAM)   │
│    Layers 0-40          │◄───►│    Layers 41-80         │
│    Compute-heavy        │     │    Memory overflow      │
│    ~0.28 TFLOPS         │     │    llama.cpp Q4_K_M     │
└─────────────────────────┘     └─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│   Mac Pro (FirePro)     │
│   Auxiliary compute     │
│   ~0.14 TFLOPS          │
└─────────────────────────┘
```

## Running Benchmarks

```bash
# On any node with Interweave installed:
python3 -m exo.interweave.benchmark

# Results saved to: benchmark_<hostname>.json
```
