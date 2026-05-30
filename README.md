# Grand Pattern — CUDA Implementation

**CUDA C++ 12+ kernels** for the Fibonacci Dual-Direction Architecture.

High-performance GPU kernels using shared memory tiling, warp-level primitives, and thrust for reductions.

## Architecture

The GPU kernels handle the parallel compute-heavy operations:
- **Embedding operations**: cosine similarity, centroid, distance — massively parallel
- **JEPA prediction**: batch prediction across all rooms simultaneously
- **Double-entry balance**: parallel reduction to check all rooms balance
- **GC merge**: parallel similarity computation → merge pairs
- **Vibe computation**: parallel reduction across perception/prediction DBs
- **Cross-room correlation**: matrix of all-pairs cosine similarity

## Kernels

| Kernel | Description |
|--------|-------------|
| `cosine_similarity_kernel` | Parallel cosine similarity using shared memory tiling |
| `batch_predict_kernel` | Per-room JEPA prediction |
| `balance_check_kernel` | Parallel double-entry balance verification |
| `decay_kernel` | Element-wise exponential decay on strengths |
| `vibe_compute_kernel` | Parallel vibe computation + normalization |
| `correlation_matrix_kernel` | All-pairs cosine similarity matrix |
| `merge_candidates_kernel` | Identify merge candidates above threshold |

## Building

```bash
mkdir build && cd build
cmake .. -DCUDA_ARCH=sm_70
make -j$(nproc)
```

Requires:
- CUDA Toolkit 12+
- CMake 3.20+
- C++17 compatible compiler

## Running Tests

```bash
./build/test_cuda_kernels
```

## Implementation Details

- **Shared memory tiling** for embedding dot products (reduces global memory bandwidth)
- **Warp-level primitives** (`__shfl_down_sync`) for efficient reductions
- **Thrust** for host-side data management and complex reductions
- **Coalesced memory access** patterns for all global loads/stores
- **Atomic operations** for balance checking and merge candidate counting

## License

MIT
