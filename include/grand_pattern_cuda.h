#ifndef GRAND_PATTERN_CUDA_H
#define GRAND_PATTERN_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPC_VIBE_DIM 16
#define GPC_MAX_HISTORY 8
#define GPC_MAX_MURMURS 4096

/* Opaque handle to the CUDA context. */
typedef struct GpcContext GpcContext;

/* Edges in Compressed Sparse Row (CSR) format.
 * row_ptr has room_count+1 entries.
 * col_idx and weights have edge_count entries.
 * A NULL weights pointer means uniform weight 1.0. */
typedef struct {
    const int* row_ptr;
    const int* col_idx;
    const float* weights;
} GpcCsrEdges;

/* Murmur gossip packet. */
typedef struct {
    uint64_t id;
    uint8_t  level;      /* 0=Neighbor, 1=Zone, 2=Fleet */
    uint8_t  ttl;
    uint64_t origin_room;
    float    payload[GPC_VIBE_DIM];
} GpcMurmur;

/* Aggregated fleet state. */
typedef struct {
    float vibe[GPC_VIBE_DIM];
    float avg_surprise;
    uint32_t anomaly_count;
} GpcFleetState;

/* ---------- Lifecycle ---------- */
GpcContext* gpc_create(void);
void        gpc_destroy(GpcContext* ctx);

/* ---------- Graph setup ---------- */
int gpc_graph_create(GpcContext* ctx, int room_count, int vibe_dim);
int gpc_set_edges_csr(GpcContext* ctx, const GpcCsrEdges* edges, int edge_count);

/* ---------- Data loading ---------- */
int gpc_set_vibes(GpcContext* ctx, const float* vibes);               /* room_count * vibe_dim */
int gpc_observe_room(GpcContext* ctx, int room_idx, const float* vibe);
int gpc_inject_murmurs(GpcContext* ctx, const GpcMurmur* murmurs, int count);
int gpc_set_signals(GpcContext* ctx, const float* signals);           /* room_count floats */

/* ---------- Full tick cycle ---------- */
int gpc_tick(GpcContext* ctx);

/* ---------- Queries ---------- */
int gpc_get_vibes(const GpcContext* ctx, float* out_vibes);           /* room_count * vibe_dim */
int gpc_get_surprise(const GpcContext* ctx, float* out_surprise);     /* room_count floats */
int gpc_get_fleet_state(const GpcContext* ctx, GpcFleetState* out_state);
int gpc_get_violations(const GpcContext* ctx, int* out_violations, int* out_count);
int gpc_get_anomalies(const GpcContext* ctx, uint8_t* out_flags);     /* room_count bytes */
int gpc_get_murmurs(const GpcContext* ctx, GpcMurmur* out_murmurs, int* out_count, int max_count);
int gpc_get_signals_out(const GpcContext* ctx, float* out_signals);   /* room_count floats */

/* ---------- Parameters ---------- */
int gpc_set_diffusion_rate(GpcContext* ctx, float rate);              /* default 0.1 */
int gpc_set_surprise_threshold(GpcContext* ctx, float threshold);     /* default 1.0 */
int gpc_set_conservation_tolerance(GpcContext* ctx, float tol);       /* default 1e-3 */

#ifdef __cplusplus
}
#endif

#endif /* GRAND_PATTERN_CUDA_H */
