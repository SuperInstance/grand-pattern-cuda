#include "grand_pattern_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VIBE_DIM GPC_VIBE_DIM
#define MAX_HISTORY GPC_MAX_HISTORY
#define MAX_MURMURS GPC_MAX_MURMURS
#define WARP_SIZE 32

/* ------------------------------------------------------------------ */
/*  Device-side structs                                                */
/* ------------------------------------------------------------------ */
struct DeviceState {
    int room_count;
    int vibe_dim;
    int edge_count;
    int max_murmurs;

    /* Graph CSR */
    int*   d_row_ptr;
    int*   d_col_idx;
    float* d_edge_weights;
    int*   d_edge_src;

    /* Room state */
    float* d_vibes;          /* room_count * vibe_dim */
    float* d_vibes_new;      /* room_count * vibe_dim */
    float* d_surprise;       /* room_count */
    uint8_t* d_anomaly;      /* room_count */

    /* JEPA history: room_count * MAX_HISTORY * vibe_dim */
    float* d_history;
    int*   d_history_len;    /* room_count */

    /* Murmurs */
    GpcMurmur* d_murmurs;
    int*       d_murmur_count;
    GpcMurmur* d_murmurs_new;
    int*       d_murmur_count_new;

    /* Signals */
    float* d_signals_in;
    float* d_signals_out;

    /* Conservation scratch */
    float* d_conservation_old;
    float* d_conservation_new;
    int*   d_violation_count;

    /* Fleet scratch */
    float* d_fleet_vibe;     /* vibe_dim */
    float* d_fleet_surprise; /* 1 */

    /* Parameters */
    float diffusion_rate;
    float surprise_threshold;
    float conservation_tolerance;
};

struct GpcContext {
    DeviceState dev;
    float* h_vibes;          /* host pinned or pageable buffers for queries */
    float* h_surprise;
    uint8_t* h_anomaly;
    GpcMurmur* h_murmurs;
    float* h_signals;
};

/* ------------------------------------------------------------------ */
/*  Helper: CUDA error checking macro                                  */
/* ------------------------------------------------------------------ */
#define CHECK_CUDA(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                              \
        return -1;                                                     \
    }                                                                  \
} while(0)

#define CHECK_CUDA_VOID(call) do {                                     \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                              \
        return;                                                        \
    }                                                                  \
} while(0)

/* ------------------------------------------------------------------ */
/*  Kernels                                                            */
/* ------------------------------------------------------------------ */

/* 1. Vibe diffusion across rooms using CSR neighbors. */
__global__ void vibe_diffusion_kernel(const float* vibes, float* vibes_new,
                                      const int* row_ptr, const int* col_idx,
                                      const float* edge_weights,
                                      int room_count, int vibe_dim,
                                      float diffusion_rate)
{
    int room = blockIdx.x * blockDim.x + threadIdx.x;
    if (room >= room_count) return;

    int base = room * vibe_dim;

    /* Start with own vibe weighted by (1 - diffusion_rate). */
    float own_weight = 1.0f - diffusion_rate;
    float neighbor_sum[16]; /* VIBE_DIM <= 16 */
    for (int d = 0; d < vibe_dim; ++d) neighbor_sum[d] = 0.0f;

    int start = row_ptr[room];
    int end   = row_ptr[room + 1];
    float total_nw = 0.0f;

    for (int e = start; e < end; ++e) {
        int neighbor = col_idx[e];
        float w = (edge_weights != nullptr) ? edge_weights[e] : 1.0f;
        total_nw += w;
        int nb = neighbor * vibe_dim;
        for (int d = 0; d < vibe_dim; ++d) {
            neighbor_sum[d] += w * vibes[nb + d];
        }
    }

    for (int d = 0; d < vibe_dim; ++d) {
        float val = own_weight * vibes[base + d];
        if (total_nw > 0.0f) {
            val += diffusion_rate * (neighbor_sum[d] / total_nw);
        }
        vibes_new[base + d] = val;
    }
}

/* 2. JEPA prediction with sliding window + surprise. */
__global__ void jepa_predict_kernel(const float* vibes, const float* history,
                                    const int* history_len,
                                    float* surprise,
                                    int room_count, int vibe_dim)
{
    int room = blockIdx.x * blockDim.x + threadIdx.x;
    if (room >= room_count) return;

    int hlen = history_len[room];
    int base = room * vibe_dim;
    float pred[16];

    if (hlen >= 2) {
        int last_off  = ((hlen - 1) % MAX_HISTORY) * vibe_dim;
        int prev_off  = ((hlen - 2) % MAX_HISTORY) * vibe_dim;
        int hbase = room * MAX_HISTORY * vibe_dim;
        for (int d = 0; d < vibe_dim; ++d) {
            float last = history[hbase + last_off + d];
            float prev = history[hbase + prev_off + d];
            pred[d] = last + (last - prev);
        }
    } else if (hlen == 1) {
        int last_off = 0;
        int hbase = room * MAX_HISTORY * vibe_dim;
        for (int d = 0; d < vibe_dim; ++d) {
            pred[d] = history[hbase + last_off + d];
        }
    } else {
        for (int d = 0; d < vibe_dim; ++d) pred[d] = 0.0f;
    }

    float mse = 0.0f;
    for (int d = 0; d < vibe_dim; ++d) {
        float err = vibes[base + d] - pred[d];
        mse += err * err;
    }
    surprise[room] = mse / vibe_dim;
}

/* Update history after tick (host or device). */
__global__ void history_update_kernel(float* history, int* history_len,
                                      const float* vibes,
                                      int room_count, int vibe_dim)
{
    int room = blockIdx.x * blockDim.x + threadIdx.x;
    if (room >= room_count) return;

    int hlen = history_len[room];
    int slot = (hlen < MAX_HISTORY) ? hlen : (hlen % MAX_HISTORY);
    int hbase = room * MAX_HISTORY * vibe_dim;
    int vbase = room * vibe_dim;
    for (int d = 0; d < vibe_dim; ++d) {
        history[hbase + slot * vibe_dim + d] = vibes[vbase + d];
    }
    if (hlen < MAX_HISTORY) {
        history_len[room] = hlen + 1;
    } else {
        history_len[room] = hlen + 1; /* keep counting, mod for slot */
    }
}

/* 3. Murmur gossip propagation with TTL decay. */
__global__ void murmur_gossip_kernel(const GpcMurmur* murmurs_in, int count_in,
                                     GpcMurmur* murmurs_out, int* count_out,
                                     const int* row_ptr, const int* col_idx,
                                     int room_count, int max_murmurs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count_in) return;

    GpcMurmur m = murmurs_in[idx];
    if (m.ttl == 0) return;

    /* Decrement TTL. */
    m.ttl -= 1;
    if (m.ttl == 0) return;

    /* Forward to all neighbors of current origin_room (used as position). */
    int room = (int)m.origin_room; /* current room index */
    if (room < 0 || room >= room_count) return;

    int start = row_ptr[room];
    int end   = row_ptr[room + 1];

    for (int e = start; e < end; ++e) {
        int neighbor = col_idx[e];
        GpcMurmur nm = m;
        nm.origin_room = (uint64_t)neighbor;
        int out_pos = atomicAdd(count_out, 1);
        if (out_pos < max_murmurs) {
            murmurs_out[out_pos] = nm;
        }
    }
}

/* 4. Conservation kernel: check sum of old vs new vibes. */
__global__ void conservation_kernel(const float* old_vibes, const float* new_vibes,
                                    int room_count, int vibe_dim,
                                    float tolerance, int* violation_count)
{
    extern __shared__ float sdata[];
    float* s_old = sdata;
    float* s_new = sdata + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float local_old = 0.0f;
    float local_new = 0.0f;

    if (gid < room_count) {
        int base = gid * vibe_dim;
        for (int d = 0; d < vibe_dim; ++d) {
            local_old += fabsf(old_vibes[base + d]);
            local_new += fabsf(new_vibes[base + d]);
        }
    }
    s_old[tid] = local_old;
    s_new[tid] = local_new;
    __syncthreads();

    /* Reduction within block. */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_old[tid] += s_old[tid + s];
            s_new[tid] += s_new[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float diff = fabsf(s_old[0] - s_new[0]);
        if (diff > tolerance) {
            atomicAdd(violation_count, 1);
        }
    }
}

/* 5. Fleet reduce: average vibes and surprises. */
__global__ void fleet_reduce_vibe_kernel(const float* vibes, float* fleet_vibe,
                                         int room_count, int vibe_dim)
{
    int d = blockIdx.x; /* one block per dimension */
    if (d >= vibe_dim) return;

    extern __shared__ float sdata_v[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < room_count; i += blockDim.x) {
        sum += vibes[i * vibe_dim + d];
    }
    sdata_v[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_v[tid] += sdata_v[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        fleet_vibe[d] = sdata_v[0] / room_count;
    }
}

__global__ void fleet_reduce_surprise_kernel(const float* surprise, float* fleet_surprise,
                                             int room_count)
{
    extern __shared__ float sdata_s[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < room_count; i += blockDim.x) {
        sum += surprise[i];
    }
    sdata_s[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_s[tid] += sdata_s[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        fleet_surprise[0] = sdata_s[0] / room_count;
    }
}

/* 6. Anomaly detection kernel. */
__global__ void anomaly_detect_kernel(const float* surprise, uint8_t* anomaly,
                                      int room_count, float threshold,
                                      int* anomaly_count)
{
    int room = blockIdx.x * blockDim.x + threadIdx.x;
    if (room >= room_count) return;

    uint8_t flag = (surprise[room] > threshold) ? 1 : 0;
    anomaly[room] = flag;
    if (flag) atomicAdd(anomaly_count, 1);
}

/* 7. Signal route kernel: route signals through graph edges (edge-parallel). */
__global__ void signal_route_kernel(const float* signals_in, float* signals_out,
                                    const int* edge_src, const int* col_idx,
                                    const float* edge_weights,
                                    int edge_count)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edge_count) return;

    int src = edge_src[e];
    int dst = col_idx[e];
    float w = (edge_weights != nullptr) ? edge_weights[e] : 1.0f;
    atomicAdd(&signals_out[dst], w * signals_in[src]);
}

/* ------------------------------------------------------------------ */
/*  Host API implementation                                            */
/* ------------------------------------------------------------------ */

GpcContext* gpc_create(void)
{
    GpcContext* ctx = (GpcContext*)calloc(1, sizeof(GpcContext));
    if (!ctx) return nullptr;

    ctx->dev.diffusion_rate = 0.1f;
    ctx->dev.surprise_threshold = 1.0f;
    ctx->dev.conservation_tolerance = 1e-3f;
    return ctx;
}

void gpc_destroy(GpcContext* ctx)
{
    if (!ctx) return;
    DeviceState* d = &ctx->dev;

    cudaFree(d->d_row_ptr);
    cudaFree(d->d_col_idx);
    cudaFree(d->d_edge_weights);
    cudaFree(d->d_vibes);
    cudaFree(d->d_vibes_new);
    cudaFree(d->d_surprise);
    cudaFree(d->d_anomaly);
    cudaFree(d->d_history);
    cudaFree(d->d_history_len);
    cudaFree(d->d_murmurs);
    cudaFree(d->d_murmur_count);
    cudaFree(d->d_murmurs_new);
    cudaFree(d->d_murmur_count_new);
    cudaFree(d->d_signals_in);
    cudaFree(d->d_signals_out);
    cudaFree(d->d_conservation_old);
    cudaFree(d->d_conservation_new);
    cudaFree(d->d_violation_count);
    cudaFree(d->d_fleet_vibe);
    cudaFree(d->d_fleet_surprise);

    free(ctx->h_vibes);
    free(ctx->h_surprise);
    free(ctx->h_anomaly);
    free(ctx->h_murmurs);
    free(ctx->h_signals);
    free(ctx);
}

int gpc_graph_create(GpcContext* ctx, int room_count, int vibe_dim)
{
    if (!ctx || room_count <= 0 || vibe_dim <= 0 || vibe_dim > VIBE_DIM) return -1;
    DeviceState* d = &ctx->dev;
    d->room_count = room_count;
    d->vibe_dim = vibe_dim;
    d->edge_count = 0;
    d->max_murmurs = MAX_MURMURS;

    size_t vibe_bytes = (size_t)room_count * vibe_dim * sizeof(float);
    size_t hist_bytes = (size_t)room_count * MAX_HISTORY * vibe_dim * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d->d_vibes, vibe_bytes));
    CHECK_CUDA(cudaMalloc(&d->d_vibes_new, vibe_bytes));
    CHECK_CUDA(cudaMalloc(&d->d_surprise, room_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d->d_anomaly, room_count * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d->d_history, hist_bytes));
    CHECK_CUDA(cudaMalloc(&d->d_history_len, room_count * sizeof(int)));
    CHECK_CUDA(cudaMemset(d->d_history_len, 0, room_count * sizeof(int)));

    CHECK_CUDA(cudaMalloc(&d->d_murmurs, MAX_MURMURS * sizeof(GpcMurmur)));
    CHECK_CUDA(cudaMalloc(&d->d_murmur_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d->d_murmur_count, 0, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d->d_murmurs_new, MAX_MURMURS * sizeof(GpcMurmur)));
    CHECK_CUDA(cudaMalloc(&d->d_murmur_count_new, sizeof(int)));
    CHECK_CUDA(cudaMemset(d->d_murmur_count_new, 0, sizeof(int)));

    CHECK_CUDA(cudaMalloc(&d->d_signals_in, room_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d->d_signals_out, room_count * sizeof(float)));
    CHECK_CUDA(cudaMemset(d->d_signals_in, 0, room_count * sizeof(float)));
    CHECK_CUDA(cudaMemset(d->d_signals_out, 0, room_count * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d->d_conservation_old, vibe_bytes));
    CHECK_CUDA(cudaMalloc(&d->d_conservation_new, vibe_bytes));
    CHECK_CUDA(cudaMalloc(&d->d_violation_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d->d_violation_count, 0, sizeof(int)));

    CHECK_CUDA(cudaMalloc(&d->d_fleet_vibe, vibe_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d->d_fleet_surprise, sizeof(float)));

    /* Default empty CSR so kernels don't crash if no edges are set. */
    CHECK_CUDA(cudaMalloc(&d->d_row_ptr, (room_count + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d->d_row_ptr, 0, (room_count + 1) * sizeof(int)));
    d->d_col_idx = nullptr;
    d->d_edge_weights = nullptr;
    d->d_edge_src = nullptr;
    d->edge_count = 0;

    /* Host buffers for queries */
    ctx->h_vibes = (float*)malloc(vibe_bytes);
    ctx->h_surprise = (float*)malloc(room_count * sizeof(float));
    ctx->h_anomaly = (uint8_t*)malloc(room_count * sizeof(uint8_t));
    ctx->h_murmurs = (GpcMurmur*)malloc(MAX_MURMURS * sizeof(GpcMurmur));
    ctx->h_signals = (float*)malloc(room_count * sizeof(float));

    if (!ctx->h_vibes || !ctx->h_surprise || !ctx->h_anomaly || !ctx->h_murmurs || !ctx->h_signals) {
        return -1;
    }
    return 0;
}

int gpc_set_edges_csr(GpcContext* ctx, const GpcCsrEdges* edges, int edge_count)
{
    if (!ctx || !edges || edge_count < 0) return -1;
    DeviceState* d = &ctx->dev;
    int rc = d->room_count;
    if (rc <= 0) return -1;

    cudaFree(d->d_row_ptr);
    cudaFree(d->d_col_idx);
    cudaFree(d->d_edge_weights);
    cudaFree(d->d_edge_src);
    d->d_row_ptr = nullptr;
    d->d_col_idx = nullptr;
    d->d_edge_weights = nullptr;
    d->d_edge_src = nullptr;

    CHECK_CUDA(cudaMalloc(&d->d_row_ptr, (rc + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d->d_col_idx, edge_count * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d->d_edge_src, edge_count * sizeof(int)));
    if (edges->weights) {
        CHECK_CUDA(cudaMalloc(&d->d_edge_weights, edge_count * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d->d_edge_weights, edges->weights, edge_count * sizeof(float), cudaMemcpyHostToDevice));
    }
    /* Build edge_src on host. */
    int* h_edge_src = (int*)malloc(edge_count * sizeof(int));
    for (int r = 0; r < rc; ++r) {
        for (int e = edges->row_ptr[r]; e < edges->row_ptr[r + 1]; ++e) {
            h_edge_src[e] = r;
        }
    }
    CHECK_CUDA(cudaMemcpy(d->d_edge_src, h_edge_src, edge_count * sizeof(int), cudaMemcpyHostToDevice));
    free(h_edge_src);
    CHECK_CUDA(cudaMemcpy(d->d_row_ptr, edges->row_ptr, (rc + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d->d_col_idx, edges->col_idx, edge_count * sizeof(int), cudaMemcpyHostToDevice));
    d->edge_count = edge_count;
    return 0;
}

int gpc_set_vibes(GpcContext* ctx, const float* vibes)
{
    if (!ctx || !vibes) return -1;
    DeviceState* d = &ctx->dev;
    size_t bytes = (size_t)d->room_count * d->vibe_dim * sizeof(float);
    CHECK_CUDA(cudaMemcpy(d->d_vibes, vibes, bytes, cudaMemcpyHostToDevice));
    return 0;
}

int gpc_observe_room(GpcContext* ctx, int room_idx, const float* vibe)
{
    if (!ctx || !vibe || room_idx < 0 || room_idx >= ctx->dev.room_count) return -1;
    DeviceState* d = &ctx->dev;
    int base = room_idx * d->vibe_dim;
    CHECK_CUDA(cudaMemcpy(d->d_vibes + base, vibe, d->vibe_dim * sizeof(float), cudaMemcpyHostToDevice));

    /* Update history on device. */
    int hlen;
    CHECK_CUDA(cudaMemcpy(&hlen, d->d_history_len + room_idx, sizeof(int), cudaMemcpyDeviceToHost));
    int slot = (hlen < MAX_HISTORY) ? hlen : (hlen % MAX_HISTORY);
    int hbase = room_idx * MAX_HISTORY * d->vibe_dim + slot * d->vibe_dim;
    CHECK_CUDA(cudaMemcpy(d->d_history + hbase, vibe, d->vibe_dim * sizeof(float), cudaMemcpyHostToDevice));
    hlen += 1;
    CHECK_CUDA(cudaMemcpy(d->d_history_len + room_idx, &hlen, sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}

int gpc_inject_murmurs(GpcContext* ctx, const GpcMurmur* murmurs, int count)
{
    if (!ctx || !murmurs || count < 0) return -1;
    DeviceState* d = &ctx->dev;
    int existing;
    CHECK_CUDA(cudaMemcpy(&existing, d->d_murmur_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (existing + count > MAX_MURMURS) count = MAX_MURMURS - existing;
    if (count <= 0) return 0;

    GpcMurmur* temp = (GpcMurmur*)malloc(count * sizeof(GpcMurmur));
    if (!temp) return -1;
    memcpy(temp, murmurs, count * sizeof(GpcMurmur));
    /* Set origin_room to the room index for routing purposes if not set. */
    for (int i = 0; i < count; ++i) {
        if (temp[i].origin_room >= (uint64_t)d->room_count) {
            temp[i].origin_room = 0;
        }
    }
    CHECK_CUDA(cudaMemcpy(d->d_murmurs + existing, temp, count * sizeof(GpcMurmur), cudaMemcpyHostToDevice));
    existing += count;
    CHECK_CUDA(cudaMemcpy(d->d_murmur_count, &existing, sizeof(int), cudaMemcpyHostToDevice));
    free(temp);
    return 0;
}

int gpc_set_signals(GpcContext* ctx, const float* signals)
{
    if (!ctx || !signals) return -1;
    DeviceState* d = &ctx->dev;
    CHECK_CUDA(cudaMemcpy(d->d_signals_in, signals, d->room_count * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

int gpc_tick(GpcContext* ctx)
{
    if (!ctx) return -1;
    DeviceState* d = &ctx->dev;
    int rc = d->room_count;
    int vd = d->vibe_dim;
    if (rc <= 0) return -1;

    int threads = 256;
    int blocks = (rc + threads - 1) / threads;

    /* 1. JEPA predict + surprise (based on current vibes). */
    jepa_predict_kernel<<<blocks, threads>>>(
        d->d_vibes, d->d_history, d->d_history_len,
        d->d_surprise, rc, vd);
    CHECK_CUDA(cudaGetLastError());

    /* 2. Vibe diffusion -> d_vibes_new. */
    vibe_diffusion_kernel<<<blocks, threads>>>(
        d->d_vibes, d->d_vibes_new,
        d->d_row_ptr, d->d_col_idx, d->d_edge_weights,
        rc, vd, d->diffusion_rate);
    CHECK_CUDA(cudaGetLastError());

    /* 3. Signal route -> d_signals_out. */
    CHECK_CUDA(cudaMemset(d->d_signals_out, 0, rc * sizeof(float)));
    if (d->edge_count > 0) {
        int eblocks = (d->edge_count + threads - 1) / threads;
        signal_route_kernel<<<eblocks, threads>>>(
            d->d_signals_in, d->d_signals_out,
            d->d_edge_src, d->d_col_idx, d->d_edge_weights,
            d->edge_count);
        CHECK_CUDA(cudaGetLastError());
    }

    /* 4. Murmur gossip. */
    int murmur_count;
    CHECK_CUDA(cudaMemcpy(&murmur_count, d->d_murmur_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (murmur_count > 0) {
        CHECK_CUDA(cudaMemset(d->d_murmur_count_new, 0, sizeof(int)));
        int mblocks = (murmur_count + threads - 1) / threads;
        murmur_gossip_kernel<<<mblocks, threads>>>(
            d->d_murmurs, murmur_count,
            d->d_murmurs_new, d->d_murmur_count_new,
            d->d_row_ptr, d->d_col_idx,
            rc, MAX_MURMURS);
        CHECK_CUDA(cudaGetLastError());

        /* Swap murmur buffers. */
        GpcMurmur* tmp_m = d->d_murmurs;
        d->d_murmurs = d->d_murmurs_new;
        d->d_murmurs_new = tmp_m;
        int* tmp_c = d->d_murmur_count;
        d->d_murmur_count = d->d_murmur_count_new;
        d->d_murmur_count_new = tmp_c;
    }

    /* 5. Conservation check (old vs new vibes). */
    CHECK_CUDA(cudaMemset(d->d_violation_count, 0, sizeof(int)));
    size_t vibe_bytes = (size_t)rc * vd * sizeof(float);
    CHECK_CUDA(cudaMemcpy(d->d_conservation_old, d->d_vibes, vibe_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d->d_conservation_new, d->d_vibes_new, vibe_bytes, cudaMemcpyDeviceToDevice));
    /* We use a single block for simplicity; could be generalized. */
    conservation_kernel<<<1, threads, 2 * threads * sizeof(float)>>>(
        d->d_conservation_old, d->d_conservation_new,
        rc, vd, d->conservation_tolerance, d->d_violation_count);
    CHECK_CUDA(cudaGetLastError());

    /* 6. Fleet reduce. */
    fleet_reduce_vibe_kernel<<<vd, threads, threads * sizeof(float)>>>(
        d->d_vibes_new, d->d_fleet_vibe, rc, vd);
    CHECK_CUDA(cudaGetLastError());
    fleet_reduce_surprise_kernel<<<1, threads, threads * sizeof(float)>>>(
        d->d_surprise, d->d_fleet_surprise, rc);
    CHECK_CUDA(cudaGetLastError());

    /* 7. Anomaly detection. */
    CHECK_CUDA(cudaMemset(d->d_anomaly, 0, rc * sizeof(uint8_t)));
    anomaly_detect_kernel<<<blocks, threads>>>(
        d->d_surprise, d->d_anomaly, rc, d->surprise_threshold, d->d_violation_count);
    CHECK_CUDA(cudaGetLastError());

    /* Update history with new vibes. */
    history_update_kernel<<<blocks, threads>>>(
        d->d_history, d->d_history_len, d->d_vibes_new, rc, vd);
    CHECK_CUDA(cudaGetLastError());

    /* Swap vibe buffers for next tick. */
    float* tmp_v = d->d_vibes;
    d->d_vibes = d->d_vibes_new;
    d->d_vibes_new = tmp_v;

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Query functions                                                    */
/* ------------------------------------------------------------------ */

int gpc_get_vibes(const GpcContext* ctx, float* out_vibes)
{
    if (!ctx || !out_vibes) return -1;
    const DeviceState* d = &ctx->dev;
    size_t bytes = (size_t)d->room_count * d->vibe_dim * sizeof(float);
    CHECK_CUDA(cudaMemcpy(out_vibes, d->d_vibes, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

int gpc_get_surprise(const GpcContext* ctx, float* out_surprise)
{
    if (!ctx || !out_surprise) return -1;
    const DeviceState* d = &ctx->dev;
    CHECK_CUDA(cudaMemcpy(out_surprise, d->d_surprise, d->room_count * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int gpc_get_fleet_state(const GpcContext* ctx, GpcFleetState* out_state)
{
    if (!ctx || !out_state) return -1;
    const DeviceState* d = &ctx->dev;
    CHECK_CUDA(cudaMemcpy(out_state->vibe, d->d_fleet_vibe, d->vibe_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&out_state->avg_surprise, d->d_fleet_surprise, sizeof(float), cudaMemcpyDeviceToHost));
    /* anomaly_count: we don't have a dedicated device counter; fetch anomaly array and count. */
    CHECK_CUDA(cudaMemcpy(ctx->h_anomaly, d->d_anomaly, d->room_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    uint32_t ac = 0;
    for (int i = 0; i < d->room_count; ++i) if (ctx->h_anomaly[i]) ac++;
    out_state->anomaly_count = ac;
    return 0;
}

int gpc_get_violations(const GpcContext* ctx, int* out_violations, int* out_count)
{
    if (!ctx || !out_violations || !out_count) return -1;
    const DeviceState* d = &ctx->dev;
    int vcount;
    CHECK_CUDA(cudaMemcpy(&vcount, d->d_violation_count, sizeof(int), cudaMemcpyDeviceToHost));
    *out_count = (vcount > 0) ? 1 : 0; /* boolean presence */
    out_violations[0] = vcount;
    return 0;
}

int gpc_get_anomalies(const GpcContext* ctx, uint8_t* out_flags)
{
    if (!ctx || !out_flags) return -1;
    const DeviceState* d = &ctx->dev;
    CHECK_CUDA(cudaMemcpy(out_flags, d->d_anomaly, d->room_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return 0;
}

int gpc_get_murmurs(const GpcContext* ctx, GpcMurmur* out_murmurs, int* out_count, int max_count)
{
    if (!ctx || !out_murmurs || !out_count) return -1;
    const DeviceState* d = &ctx->dev;
    int count;
    CHECK_CUDA(cudaMemcpy(&count, d->d_murmur_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > max_count) count = max_count;
    if (count > 0) {
        CHECK_CUDA(cudaMemcpy(out_murmurs, d->d_murmurs, count * sizeof(GpcMurmur), cudaMemcpyDeviceToHost));
    }
    *out_count = count;
    return 0;
}

int gpc_get_signals_out(const GpcContext* ctx, float* out_signals)
{
    if (!ctx || !out_signals) return -1;
    const DeviceState* d = &ctx->dev;
    CHECK_CUDA(cudaMemcpy(out_signals, d->d_signals_out, d->room_count * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Parameters                                                         */
/* ------------------------------------------------------------------ */

int gpc_set_diffusion_rate(GpcContext* ctx, float rate)
{
    if (!ctx || rate < 0.0f || rate > 1.0f) return -1;
    ctx->dev.diffusion_rate = rate;
    return 0;
}

int gpc_set_surprise_threshold(GpcContext* ctx, float threshold)
{
    if (!ctx || threshold < 0.0f) return -1;
    ctx->dev.surprise_threshold = threshold;
    return 0;
}

int gpc_set_conservation_tolerance(GpcContext* ctx, float tol)
{
    if (!ctx || tol < 0.0f) return -1;
    ctx->dev.conservation_tolerance = tol;
    return 0;
}
