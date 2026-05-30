#include "grand_pattern_cuda.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#define EPS 1e-4f

static bool near(float a, float b, float eps = EPS) {
    return std::fabs(a - b) < eps;
}

/* ------------------------------------------------------------------ */
/* 1. Context create / destroy                                         */
/* ------------------------------------------------------------------ */
static void test_context_create_destroy() {
    GpcContext* ctx = gpc_create();
    assert(ctx != nullptr);
    gpc_destroy(ctx);
    printf("PASS: test_context_create_destroy\n");
}

/* ------------------------------------------------------------------ */
/* 2. Graph creation                                                   */
/* ------------------------------------------------------------------ */
static void test_create_graph() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 4, GPC_VIBE_DIM) == 0);
    gpc_destroy(ctx);
    printf("PASS: test_create_graph\n");
}

/* ------------------------------------------------------------------ */
/* 3. Set edges CSR                                                    */
/* ------------------------------------------------------------------ */
static void test_set_edges_csr() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 3, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2, 2};
    int col_idx[] = {1, 2};
    float w[] = {1.0f, 1.0f};
    GpcCsrEdges edges = {row_ptr, col_idx, w};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);
    gpc_destroy(ctx);
    printf("PASS: test_set_edges_csr\n");
}

/* ------------------------------------------------------------------ */
/* 4. Diffusion with no edges (vibes should stay almost same)          */
/* ------------------------------------------------------------------ */
static void test_diffusion_no_edges() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 0, 0};
    int col_idx[] = {};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 0) == 0);

    float vibes[2 * GPC_VIBE_DIM];
    for (int i = 0; i < 2 * GPC_VIBE_DIM; ++i) vibes[i] = (float)i;
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    float out[2 * GPC_VIBE_DIM];
    assert(gpc_get_vibes(ctx, out) == 0);
    for (int i = 0; i < 2 * GPC_VIBE_DIM; ++i) {
        assert(near(out[i], vibes[i] * 0.9f)); /* own_weight = 0.9 */
    }
    gpc_destroy(ctx);
    printf("PASS: test_diffusion_no_edges\n");
}

/* ------------------------------------------------------------------ */
/* 5. Diffusion uniform graph                                          */
/* ------------------------------------------------------------------ */
static void test_diffusion_uniform_graph() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 3, GPC_VIBE_DIM) == 0);
    /* ring: 0->1, 1->2, 2->0 */
    int row_ptr[] = {0, 1, 2, 3};
    int col_idx[] = {1, 2, 0};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 3) == 0);
    assert(gpc_set_diffusion_rate(ctx, 0.5f) == 0);

    float vibes[3 * GPC_VIBE_DIM] = {0};
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        vibes[0 * GPC_VIBE_DIM + d] = 1.0f;
        vibes[1 * GPC_VIBE_DIM + d] = 2.0f;
        vibes[2 * GPC_VIBE_DIM + d] = 3.0f;
    }
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    float out[3 * GPC_VIBE_DIM];
    assert(gpc_get_vibes(ctx, out) == 0);
    /* room0 neighbor=room1(2) => 0.5*1 + 0.5*2 = 1.5 */
    assert(near(out[0], 1.5f, 1e-3f));
    /* room1 neighbor=room2(3) => 0.5*2 + 0.5*3 = 2.5 */
    assert(near(out[GPC_VIBE_DIM], 2.5f, 1e-3f));
    /* room2 neighbor=room0(1) => 0.5*3 + 0.5*1 = 2.0 */
    assert(near(out[2 * GPC_VIBE_DIM], 2.0f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_diffusion_uniform_graph\n");
}

/* ------------------------------------------------------------------ */
/* 6. Diffusion weighted graph                                         */
/* ------------------------------------------------------------------ */
static void test_diffusion_weighted_graph() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2};
    int col_idx[] = {1, 0};
    float w[] = {2.0f, 1.0f};
    GpcCsrEdges edges = {row_ptr, col_idx, w};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);
    assert(gpc_set_diffusion_rate(ctx, 0.5f) == 0);

    float vibes[2 * GPC_VIBE_DIM] = {0};
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        vibes[0 * GPC_VIBE_DIM + d] = 0.0f;
        vibes[1 * GPC_VIBE_DIM + d] = 4.0f;
    }
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    float out[2 * GPC_VIBE_DIM];
    assert(gpc_get_vibes(ctx, out) == 0);
    /* room0 neighbor=room1 weight 2 => avg = 4 */
    /* room0 result = 0.5*0 + 0.5*4 = 2 */
    assert(near(out[0], 2.0f, 1e-3f));
    /* room1 neighbor=room0 weight 1 => avg = 0 */
    /* room1 result = 0.5*4 + 0.5*0 = 2 */
    assert(near(out[GPC_VIBE_DIM], 2.0f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_diffusion_weighted_graph\n");
}

/* ------------------------------------------------------------------ */
/* 7. JEPA predict empty history                                       */
/* ------------------------------------------------------------------ */
static void test_jepa_predict_empty_history() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float vibe[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) vibe[d] = 1.0f;
    assert(gpc_set_vibes(ctx, vibe) == 0);
    assert(gpc_tick(ctx) == 0);

    float surprise;
    assert(gpc_get_surprise(ctx, &surprise) == 0);
    /* prediction = 0, actual = 1 => mse = 1 */
    assert(near(surprise, 1.0f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_jepa_predict_empty_history\n");
}

/* ------------------------------------------------------------------ */
/* 8. JEPA predict linear extrapolation                                */
/* ------------------------------------------------------------------ */
static void test_jepa_predict_linear() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float v0[GPC_VIBE_DIM], v1[GPC_VIBE_DIM], v2[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        v0[d] = 0.0f;
        v1[d] = 1.0f;
        v2[d] = 3.0f; /* surprise = actual - predicted */
    }
    assert(gpc_observe_room(ctx, 0, v0) == 0);
    assert(gpc_observe_room(ctx, 0, v1) == 0);
    assert(gpc_set_vibes(ctx, v2) == 0);
    assert(gpc_tick(ctx) == 0);

    float surprise;
    assert(gpc_get_surprise(ctx, &surprise) == 0);
    /* predicted = 1 + (1-0) = 2, actual = 3, error = 1, mse = 1 */
    assert(near(surprise, 1.0f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_jepa_predict_linear\n");
}

/* ------------------------------------------------------------------ */
/* 9. JEPA predict moving average (flat line)                          */
/* ------------------------------------------------------------------ */
static void test_jepa_predict_moving_avg() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float v0[GPC_VIBE_DIM], v1[GPC_VIBE_DIM], v2[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        v0[d] = 2.0f;
        v1[d] = 2.0f;
        v2[d] = 2.0f;
    }
    assert(gpc_observe_room(ctx, 0, v0) == 0);
    assert(gpc_observe_room(ctx, 0, v1) == 0);
    assert(gpc_set_vibes(ctx, v2) == 0);
    assert(gpc_tick(ctx) == 0);

    float surprise;
    assert(gpc_get_surprise(ctx, &surprise) == 0);
    assert(near(surprise, 0.0f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_jepa_predict_moving_avg\n");
}

/* ------------------------------------------------------------------ */
/* 10. Surprise computation exact                                      */
/* ------------------------------------------------------------------ */
static void test_surprise_computation() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float v0[GPC_VIBE_DIM], v1[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        v0[d] = 0.0f;
        v1[d] = (float)d;
    }
    assert(gpc_observe_room(ctx, 0, v0) == 0);
    assert(gpc_set_vibes(ctx, v1) == 0);
    assert(gpc_tick(ctx) == 0);

    float surprise;
    assert(gpc_get_surprise(ctx, &surprise) == 0);
    float expected = 0.0f;
    for (int d = 0; d < GPC_VIBE_DIM; ++d) expected += (float)(d * d);
    expected /= GPC_VIBE_DIM;
    assert(near(surprise, expected, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_surprise_computation\n");
}

/* ------------------------------------------------------------------ */
/* 11. Murmur inject                                                   */
/* ------------------------------------------------------------------ */
static void test_murmur_inject() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    GpcMurmur m;
    memset(&m, 0, sizeof(m));
    m.id = 42;
    m.ttl = 3;
    m.origin_room = 0;
    for (int d = 0; d < GPC_VIBE_DIM; ++d) m.payload[d] = 1.0f;
    assert(gpc_inject_murmurs(ctx, &m, 1) == 0);

    GpcMurmur out[4];
    int count;
    assert(gpc_get_murmurs(ctx, out, &count, 4) == 0);
    assert(count == 1);
    assert(out[0].id == 42);
    gpc_destroy(ctx);
    printf("PASS: test_murmur_inject\n");
}

/* ------------------------------------------------------------------ */
/* 12. Murmur TTL decay                                                */
/* ------------------------------------------------------------------ */
static void test_murmur_ttl_decay() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2};
    int col_idx[] = {1, 0};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);

    GpcMurmur m;
    memset(&m, 0, sizeof(m));
    m.id = 7;
    m.ttl = 1;
    m.origin_room = 0;
    assert(gpc_inject_murmurs(ctx, &m, 1) == 0);

    assert(gpc_tick(ctx) == 0);

    GpcMurmur out[64];
    int count;
    assert(gpc_get_murmurs(ctx, out, &count, 64) == 0);
    /* ttl=1 => after hop becomes 0, so packet dies, no forwarding */
    assert(count == 0);
    gpc_destroy(ctx);
    printf("PASS: test_murmur_ttl_decay\n");
}

/* ------------------------------------------------------------------ */
/* 13. Murmur gossip propagation                                       */
/* ------------------------------------------------------------------ */
static void test_murmur_gossip_propagation() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 3, GPC_VIBE_DIM) == 0);
    /* star centered at 0: 0->1, 0->2 */
    int row_ptr[] = {0, 2, 2, 2};
    int col_idx[] = {1, 2};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);

    GpcMurmur m;
    memset(&m, 0, sizeof(m));
    m.id = 99;
    m.ttl = 3;
    m.origin_room = 0;
    assert(gpc_inject_murmurs(ctx, &m, 1) == 0);

    assert(gpc_tick(ctx) == 0);

    GpcMurmur out[64];
    int count;
    assert(gpc_get_murmurs(ctx, out, &count, 64) == 0);
    assert(count == 2);
    assert(out[0].ttl == 2);
    assert(out[1].ttl == 2);
    gpc_destroy(ctx);
    printf("PASS: test_murmur_gossip_propagation\n");
}

/* ------------------------------------------------------------------ */
/* 14. Conservation pass                                               */
/* ------------------------------------------------------------------ */
static void test_conservation_pass() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2};
    int col_idx[] = {1, 0};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);
    assert(gpc_set_conservation_tolerance(ctx, 10.0f) == 0); /* generous */

    float vibes[2 * GPC_VIBE_DIM] = {0};
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        vibes[d] = 1.0f;
        vibes[GPC_VIBE_DIM + d] = 1.0f;
    }
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    int violations[1];
    int vcount;
    assert(gpc_get_violations(ctx, violations, &vcount) == 0);
    assert(vcount == 0);
    gpc_destroy(ctx);
    printf("PASS: test_conservation_pass\n");
}

/* ------------------------------------------------------------------ */
/* 15. Conservation fail                                               */
/* ------------------------------------------------------------------ */
static void test_conservation_fail() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2};
    int col_idx[] = {1, 0};
    float w[] = {10.0f, 10.0f};
    GpcCsrEdges edges = {row_ptr, col_idx, w};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);
    assert(gpc_set_diffusion_rate(ctx, 0.9f) == 0);
    assert(gpc_set_conservation_tolerance(ctx, 1e-6f) == 0); /* strict */

    float vibes[2 * GPC_VIBE_DIM] = {0};
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        vibes[d] = 1.0f;
        vibes[GPC_VIBE_DIM + d] = 10.0f;
    }
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    int violations[1];
    int vcount;
    assert(gpc_get_violations(ctx, violations, &vcount) == 0);
    /* Since weighted diffusion changes total L1 sum, conservation fails. */
    assert(vcount > 0);
    gpc_destroy(ctx);
    printf("PASS: test_conservation_fail\n");
}

/* ------------------------------------------------------------------ */
/* 16. Fleet reduce                                                    */
/* ------------------------------------------------------------------ */
static void test_fleet_reduce() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 4, GPC_VIBE_DIM) == 0);
    float vibes[4 * GPC_VIBE_DIM] = {0};
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        vibes[0 * GPC_VIBE_DIM + d] = 0.0f;
        vibes[1 * GPC_VIBE_DIM + d] = 2.0f;
        vibes[2 * GPC_VIBE_DIM + d] = 4.0f;
        vibes[3 * GPC_VIBE_DIM + d] = 6.0f;
    }
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    GpcFleetState fs;
    assert(gpc_get_fleet_state(ctx, &fs) == 0);
    /* Average of 0,2,4,6 after diffusion (no edges) => each *= 0.9, avg = 2.7 */
    assert(near(fs.vibe[0], 2.7f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_fleet_reduce\n");
}

/* ------------------------------------------------------------------ */
/* 17. Anomaly detect none                                             */
/* ------------------------------------------------------------------ */
static void test_anomaly_detect_none() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    assert(gpc_set_surprise_threshold(ctx, 1000.0f) == 0);
    float vibes[2 * GPC_VIBE_DIM] = {0};
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);

    uint8_t flags[2];
    assert(gpc_get_anomalies(ctx, flags) == 0);
    assert(flags[0] == 0);
    assert(flags[1] == 0);
    gpc_destroy(ctx);
    printf("PASS: test_anomaly_detect_none\n");
}

/* ------------------------------------------------------------------ */
/* 18. Anomaly detect some                                             */
/* ------------------------------------------------------------------ */
static void test_anomaly_detect_some() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    assert(gpc_set_surprise_threshold(ctx, 0.5f) == 0);
    float v0[GPC_VIBE_DIM], v1[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        v0[d] = 0.0f;
        v1[d] = 10.0f;
    }
    assert(gpc_observe_room(ctx, 0, v0) == 0);
    assert(gpc_set_vibes(ctx, v1) == 0);
    assert(gpc_tick(ctx) == 0);

    uint8_t flags[1];
    assert(gpc_get_anomalies(ctx, flags) == 0);
    assert(flags[0] == 1);
    gpc_destroy(ctx);
    printf("PASS: test_anomaly_detect_some\n");
}

/* ------------------------------------------------------------------ */
/* 19. Signal route basic                                              */
/* ------------------------------------------------------------------ */
static void test_signal_route_basic() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 3, GPC_VIBE_DIM) == 0);
    /* 0->1, 1->2, 2->0 */
    int row_ptr[] = {0, 1, 2, 3};
    int col_idx[] = {1, 2, 0};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 3) == 0);

    float sig[3] = {1.0f, 2.0f, 3.0f};
    assert(gpc_set_signals(ctx, sig) == 0);
    assert(gpc_tick(ctx) == 0);

    float out[3];
    assert(gpc_get_signals_out(ctx, out) == 0);
    /* out[0] = sig[2] = 3 */
    assert(near(out[0], 3.0f, 1e-4f));
    /* out[1] = sig[0] = 1 */
    assert(near(out[1], 1.0f, 1e-4f));
    /* out[2] = sig[1] = 2 */
    assert(near(out[2], 2.0f, 1e-4f));
    gpc_destroy(ctx);
    printf("PASS: test_signal_route_basic\n");
}

/* ------------------------------------------------------------------ */
/* 20. Signal route weighted                                           */
/* ------------------------------------------------------------------ */
static void test_signal_route_weighted() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2};
    int col_idx[] = {1, 0};
    float w[] = {2.0f, 0.5f};
    GpcCsrEdges edges = {row_ptr, col_idx, w};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);

    float sig[2] = {4.0f, 1.0f};
    assert(gpc_set_signals(ctx, sig) == 0);
    assert(gpc_tick(ctx) == 0);

    float out[2];
    assert(gpc_get_signals_out(ctx, out) == 0);
    /* edge1: src=1,dst=0,w=0.5 => out[0] = 0.5 * sig[1] = 0.5 */
    assert(near(out[0], 0.5f, 1e-4f));
    /* edge0: src=0,dst=1,w=2.0 => out[1] = 2.0 * sig[0] = 8.0 */
    assert(near(out[1], 8.0f, 1e-4f));
    gpc_destroy(ctx);
    printf("PASS: test_signal_route_weighted\n");
}

/* ------------------------------------------------------------------ */
/* 21. Full tick cycle                                                 */
/* ------------------------------------------------------------------ */
static void test_full_tick_cycle() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 3, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2, 3};
    int col_idx[] = {1, 2, 0};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 3) == 0);

    float vibes[3 * GPC_VIBE_DIM] = {0};
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        vibes[0 * GPC_VIBE_DIM + d] = 1.0f;
        vibes[1 * GPC_VIBE_DIM + d] = 2.0f;
        vibes[2 * GPC_VIBE_DIM + d] = 3.0f;
    }
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_set_signals(ctx, vibes) == 0); /* reuse as signals */

    GpcMurmur m;
    memset(&m, 0, sizeof(m));
    m.id = 1;
    m.ttl = 2;
    m.origin_room = 0;
    assert(gpc_inject_murmurs(ctx, &m, 1) == 0);

    assert(gpc_tick(ctx) == 0);

    float out_vibes[3 * GPC_VIBE_DIM];
    float out_surprise[3];
    float out_signals[3];
    GpcFleetState fs;
    uint8_t anomaly[3];
    int violations[1], vcount;

    assert(gpc_get_vibes(ctx, out_vibes) == 0);
    assert(gpc_get_surprise(ctx, out_surprise) == 0);
    assert(gpc_get_signals_out(ctx, out_signals) == 0);
    assert(gpc_get_fleet_state(ctx, &fs) == 0);
    assert(gpc_get_anomalies(ctx, anomaly) == 0);
    assert(gpc_get_violations(ctx, violations, &vcount) == 0);

    /* default diffusion_rate=0.1, room0 neighbor=room1(value 2) => 0.9*1 + 0.1*2 = 1.1 */
    assert(near(out_vibes[0], 1.1f, 1e-2f));
    assert(out_surprise[0] >= 0.0f);
    assert(anomaly[0] == 0 || anomaly[0] == 1);
    gpc_destroy(ctx);
    printf("PASS: test_full_tick_cycle\n");
}

/* ------------------------------------------------------------------ */
/* 22. Empty graph (no edges)                                          */
/* ------------------------------------------------------------------ */
static void test_empty_graph() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float vibes[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) vibes[d] = 3.0f;
    assert(gpc_set_vibes(ctx, vibes) == 0);
    assert(gpc_tick(ctx) == 0);
    float out[GPC_VIBE_DIM];
    assert(gpc_get_vibes(ctx, out) == 0);
    assert(near(out[0], 3.0f * 0.9f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_empty_graph\n");
}

/* ------------------------------------------------------------------ */
/* 23. Single room                                                     */
/* ------------------------------------------------------------------ */
static void test_single_room() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float v[GPC_VIBE_DIM];
    for (int d = 0; d < GPC_VIBE_DIM; ++d) v[d] = (float)d;
    assert(gpc_set_vibes(ctx, v) == 0);
    assert(gpc_tick(ctx) == 0);
    float out[GPC_VIBE_DIM];
    assert(gpc_get_vibes(ctx, out) == 0);
    for (int d = 0; d < GPC_VIBE_DIM; ++d) {
        assert(near(out[d], v[d] * 0.9f, 1e-3f));
    }
    gpc_destroy(ctx);
    printf("PASS: test_single_room\n");
}

/* ------------------------------------------------------------------ */
/* 24. Parameter set diffusion rate                                    */
/* ------------------------------------------------------------------ */
static void test_parameter_set_diffusion_rate() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    assert(gpc_set_diffusion_rate(ctx, 0.5f) == 0);
    assert(gpc_set_diffusion_rate(ctx, 1.0f) == 0);
    assert(gpc_set_diffusion_rate(ctx, -0.1f) == -1);
    assert(gpc_set_diffusion_rate(ctx, 1.1f) == -1);
    gpc_destroy(ctx);
    printf("PASS: test_parameter_set_diffusion_rate\n");
}

/* ------------------------------------------------------------------ */
/* 25. Parameter set surprise threshold                                */
/* ------------------------------------------------------------------ */
static void test_parameter_set_surprise_threshold() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    assert(gpc_set_surprise_threshold(ctx, 0.0f) == 0);
    assert(gpc_set_surprise_threshold(ctx, 5.0f) == 0);
    assert(gpc_set_surprise_threshold(ctx, -1.0f) == -1);
    gpc_destroy(ctx);
    printf("PASS: test_parameter_set_surprise_threshold\n");
}

/* ------------------------------------------------------------------ */
/* 26. Parameter set conservation tolerance                            */
/* ------------------------------------------------------------------ */
static void test_parameter_set_conservation_tolerance() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    assert(gpc_set_conservation_tolerance(ctx, 1e-6f) == 0);
    assert(gpc_set_conservation_tolerance(ctx, 0.0f) == 0);
    assert(gpc_set_conservation_tolerance(ctx, -0.1f) == -1);
    gpc_destroy(ctx);
    printf("PASS: test_parameter_set_conservation_tolerance\n");
}

/* ------------------------------------------------------------------ */
/* 27. Get / set vibes round-trip                                      */
/* ------------------------------------------------------------------ */
static void test_get_set_vibes() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    float vibes[2 * GPC_VIBE_DIM];
    for (int i = 0; i < 2 * GPC_VIBE_DIM; ++i) vibes[i] = (float)(i * i);
    assert(gpc_set_vibes(ctx, vibes) == 0);
    float out[2 * GPC_VIBE_DIM];
    assert(gpc_get_vibes(ctx, out) == 0);
    for (int i = 0; i < 2 * GPC_VIBE_DIM; ++i) {
        assert(near(out[i], vibes[i], 1e-4f));
    }
    gpc_destroy(ctx);
    printf("PASS: test_get_set_vibes\n");
}

/* ------------------------------------------------------------------ */
/* 28. Murmur expiration after multiple ticks                          */
/* ------------------------------------------------------------------ */
static void test_murmur_expiration() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 2, GPC_VIBE_DIM) == 0);
    int row_ptr[] = {0, 1, 2};
    int col_idx[] = {1, 0};
    GpcCsrEdges edges = {row_ptr, col_idx, nullptr};
    assert(gpc_set_edges_csr(ctx, &edges, 2) == 0);

    GpcMurmur m;
    memset(&m, 0, sizeof(m));
    m.id = 123;
    m.ttl = 2;
    m.origin_room = 0;
    assert(gpc_inject_murmurs(ctx, &m, 1) == 0);

    /* tick 1: ttl 2 -> 1, forwarded to room1 */
    assert(gpc_tick(ctx) == 0);
    GpcMurmur out[64];
    int count;
    assert(gpc_get_murmurs(ctx, out, &count, 64) == 0);
    assert(count == 1);
    assert(out[0].ttl == 1);
    assert(out[0].origin_room == 1);

    /* tick 2: ttl 1 -> 0, dies */
    assert(gpc_tick(ctx) == 0);
    assert(gpc_get_murmurs(ctx, out, &count, 64) == 0);
    assert(count == 0);
    gpc_destroy(ctx);
    printf("PASS: test_murmur_expiration\n");
}

/* ------------------------------------------------------------------ */
/* 29. Edge case zero rooms (should fail gracefully)                   */
/* ------------------------------------------------------------------ */
static void test_edge_case_zero_rooms() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 0, GPC_VIBE_DIM) == -1);
    gpc_destroy(ctx);
    printf("PASS: test_edge_case_zero_rooms\n");
}

/* ------------------------------------------------------------------ */
/* 30. Multiple observe / history rollover                             */
/* ------------------------------------------------------------------ */
static void test_history_rollover() {
    GpcContext* ctx = gpc_create();
    assert(gpc_graph_create(ctx, 1, GPC_VIBE_DIM) == 0);
    float v[GPC_VIBE_DIM];
    for (int t = 0; t < GPC_MAX_HISTORY + 4; ++t) {
        for (int d = 0; d < GPC_VIBE_DIM; ++d) v[d] = (float)t;
        assert(gpc_observe_room(ctx, 0, v) == 0);
    }
    /* Set vibes to the predicted next value to get near-zero surprise. */
    for (int d = 0; d < GPC_VIBE_DIM; ++d) v[d] = (float)(GPC_MAX_HISTORY + 4);
    assert(gpc_set_vibes(ctx, v) == 0);
    assert(gpc_tick(ctx) == 0);
    float surprise;
    assert(gpc_get_surprise(ctx, &surprise) == 0);
    assert(near(surprise, 0.0f, 1e-3f));
    gpc_destroy(ctx);
    printf("PASS: test_history_rollover\n");
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main() {
    printf("Running grand-pattern-cuda tests...\n");
    test_context_create_destroy();
    test_create_graph();
    test_set_edges_csr();
    test_diffusion_no_edges();
    test_diffusion_uniform_graph();
    test_diffusion_weighted_graph();
    test_jepa_predict_empty_history();
    test_jepa_predict_linear();
    test_jepa_predict_moving_avg();
    test_surprise_computation();
    test_murmur_inject();
    test_murmur_ttl_decay();
    test_murmur_gossip_propagation();
    test_conservation_pass();
    test_conservation_fail();
    test_fleet_reduce();
    test_anomaly_detect_none();
    test_anomaly_detect_some();
    test_signal_route_basic();
    test_signal_route_weighted();
    test_full_tick_cycle();
    test_empty_graph();
    test_single_room();
    test_parameter_set_diffusion_rate();
    test_parameter_set_surprise_threshold();
    test_parameter_set_conservation_tolerance();
    test_get_set_vibes();
    test_murmur_expiration();
    test_edge_case_zero_rooms();
    test_history_rollover();
    printf("All 30 tests passed!\n");
    return 0;
}
