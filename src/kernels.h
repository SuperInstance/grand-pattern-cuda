//
// kernels.h — CUDA kernel declarations
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#ifndef GRAND_PATTERN_CUDA_KERNELS_H
#define GRAND_PATTERN_CUDA_KERNELS_H

#include <cuda_runtime.h>

// Cosine similarity: compute cosine similarity between pairs of embeddings
// grid: (n, 1, 1), block: (block_size, 1, 1)
__global__ void cosine_similarity_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n, int dim);

// Batch predict: each block handles one room
// prediction = perception + delta * (target - perception)
__global__ void batch_predict_kernel(
    const float* __restrict__ perceptions,
    const float* __restrict__ targets,
    float* __restrict__ result,
    int n, int dim, float delta);

// Balance check: each thread checks one room
// result[i] = (z_in[i] == z_out[i]) ? 0 : 1
__global__ void balance_check_kernel(
    const unsigned int* __restrict__ z_in,
    const unsigned int* __restrict__ z_out,
    unsigned int* __restrict__ result,
    unsigned int* __restrict__ imbalance_count,
    int n);

// Decay: strengths[i] *= exp(-rate * ages[i])
__global__ void decay_kernel(
    float* __restrict__ strengths,
    const float* __restrict__ ages,
    int n, float rate);

// Vibe compute: vibe = normalize(embedding + velocity * dt)
__global__ void vibe_compute_kernel(
    const float* __restrict__ embeddings,
    const float* __restrict__ velocities,
    float* __restrict__ vibes,
    int n, int dim, float dt);

// Correlation matrix: all-pairs cosine similarity
// grid: (n_rooms, n_rooms, 1)
__global__ void correlation_matrix_kernel(
    const float* __restrict__ vibes,
    float* __restrict__ matrix,
    int n_rooms, int dim);

// Merge candidates: identify consecutive pairs above threshold
__global__ void merge_candidates_kernel(
    const float* __restrict__ embeddings,
    const float* __restrict__ strengths,
    unsigned int* __restrict__ candidates,
    unsigned int* __restrict__ candidate_count,
    int n, int dim, float threshold);

#endif
