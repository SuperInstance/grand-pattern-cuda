//
// kernels.cu — CUDA kernel implementations
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#include "kernels.h"
#include <cmath>

// Helper: warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ===========================================================================
// Cosine Similarity Kernel
// ===========================================================================
__global__ void cosine_similarity_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int n, int dim)
{
    extern __shared__ float smem[];  // 3 * blockDim.x

    int pair_id = blockIdx.x;
    if (pair_id >= n) return;

    const float* a_ptr = a + pair_id * dim;
    const float* b_ptr = b + pair_id * dim;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float va = a_ptr[i];
        float vb = b_ptr[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    // Warp reduce
    dot = warp_reduce_sum(dot);
    norm_a = warp_reduce_sum(norm_a);
    norm_b = warp_reduce_sum(norm_b);

    // Use shared memory for cross-warp reduction
    float* s_dot = smem;
    float* s_na = smem + blockDim.x;
    float* s_nb = smem + 2 * blockDim.x;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_dot[warp_id] = dot;
        s_na[warp_id] = norm_a;
        s_nb[warp_id] = norm_b;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        dot = (threadIdx.x < (blockDim.x >> 5)) ? s_dot[lane] : 0.0f;
        norm_a = (threadIdx.x < (blockDim.x >> 5)) ? s_na[lane] : 0.0f;
        norm_b = (threadIdx.x < (blockDim.x >> 5)) ? s_nb[lane] : 0.0f;

        dot = warp_reduce_sum(dot);
        norm_a = warp_reduce_sum(norm_a);
        norm_b = warp_reduce_sum(norm_b);

        if (threadIdx.x == 0) {
            float denom = sqrtf(norm_a) * sqrtf(norm_b);
            result[pair_id] = (denom > 1e-8f) ? dot / denom : 0.0f;
        }
    }
}

// ===========================================================================
// Batch Predict Kernel
// ===========================================================================
__global__ void batch_predict_kernel(
    const float* __restrict__ perceptions,
    const float* __restrict__ targets,
    float* __restrict__ result,
    int n, int dim, float delta)
{
    int room = blockIdx.x;
    if (room >= n) return;

    const float* p = perceptions + room * dim;
    const float* t = targets + room * dim;
    float* r = result + room * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        r[i] = p[i] + delta * (t[i] - p[i]);
    }
}

// ===========================================================================
// Balance Check Kernel
// ===========================================================================
__global__ void balance_check_kernel(
    const unsigned int* __restrict__ z_in,
    const unsigned int* __restrict__ z_out,
    unsigned int* __restrict__ result,
    unsigned int* __restrict__ imbalance_count,
    int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    if (z_in[i] != z_out[i]) {
        result[i] = 1;
        atomicAdd(imbalance_count, 1u);
    } else {
        result[i] = 0;
    }
}

// ===========================================================================
// Decay Kernel
// ===========================================================================
__global__ void decay_kernel(
    float* __restrict__ strengths,
    const float* __restrict__ ages,
    int n, float rate)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    strengths[i] *= expf(-rate * ages[i]);
}

// ===========================================================================
// Vibe Compute Kernel
// ===========================================================================
__global__ void vibe_compute_kernel(
    const float* __restrict__ embeddings,
    const float* __restrict__ velocities,
    float* __restrict__ vibes,
    int n, int dim, float dt)
{
    extern __shared__ float smem[];

    int room = blockIdx.x;
    if (room >= n) return;

    const float* emb = embeddings + room * dim;
    const float* vel = velocities + room * dim;
    float* vib = vibes + room * dim;

    float norm_sq = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = emb[i] + vel[i] * dt;
        vib[i] = v;
        norm_sq += v * v;
    }

    norm_sq = warp_reduce_sum(norm_sq);

    float* s_norm = smem;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) s_norm[warp_id] = norm_sq;
    __syncthreads();

    float norm = 0.0f;
    if (warp_id == 0) {
        float val = (threadIdx.x < (blockDim.x >> 5)) ? s_norm[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0) {
            s_norm[0] = sqrtf(val);
        }
    }
    __syncthreads();

    norm = s_norm[0];
    if (norm > 1e-8f) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            vib[i] /= norm;
        }
    }
}

// ===========================================================================
// Correlation Matrix Kernel
// ===========================================================================
__global__ void correlation_matrix_kernel(
    const float* __restrict__ vibes,
    float* __restrict__ matrix,
    int n_rooms, int dim)
{
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= n_rooms || col >= n_rooms) return;
    if (row > col) return;  // only upper triangle

    const float* a = vibes + row * dim;
    const float* b = vibes + col * dim;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float va = a[i], vb = b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }

    dot = warp_reduce_sum(dot);
    na = warp_reduce_sum(na);
    nb = warp_reduce_sum(nb);

    __shared__ float s_dot, s_na, s_nb;
    if (threadIdx.x == 0) { s_dot = 0; s_na = 0; s_nb = 0; }
    __syncthreads();

    atomicAdd(&s_dot, dot);
    atomicAdd(&s_na, na);
    atomicAdd(&s_nb, nb);
    __syncthreads();

    if (threadIdx.x == 0) {
        float denom = sqrtf(s_na) * sqrtf(s_nb);
        float sim = (denom > 1e-8f) ? s_dot / denom : 0.0f;
        matrix[row * n_rooms + col] = sim;
        matrix[col * n_rooms + row] = sim;
    }
}

// ===========================================================================
// Merge Candidates Kernel
// ===========================================================================
__global__ void merge_candidates_kernel(
    const float* __restrict__ embeddings,
    const float* __restrict__ strengths,
    unsigned int* __restrict__ candidates,
    unsigned int* __restrict__ candidate_count,
    int n, int dim, float threshold)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n - 1) return;

    const float* a = embeddings + i * dim;
    const float* b = embeddings + (i + 1) * dim;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;

    for (int j = 0; j < dim; j++) {
        float va = a[j], vb = b[j];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }

    float denom = sqrtf(na) * sqrtf(nb);
    float sim = (denom > 1e-8f) ? dot / denom : 0.0f;

    if (sim >= threshold) {
        candidates[i] = 1;
        atomicAdd(candidate_count, 1u);
    } else {
        candidates[i] = 0;
    }
}
