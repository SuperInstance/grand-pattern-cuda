//
// tests.cu — CUDA kernel tests
// Grand Pattern Fibonacci Dual-Direction Architecture
//

#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

static int test_cosine_similarity() {
    printf("=== Test: Cosine Similarity (CUDA) ===\n");
    const int n = 4, dim = 8;

    float h_a[n*dim], h_b[n*dim], h_r[n];
    // Pair 0: identical → 1.0
    for (int j = 0; j < dim; j++) { h_a[j] = 1.0f; h_b[j] = 1.0f; }
    // Pair 1: orthogonal → 0
    for (int j = 0; j < dim; j++) { h_a[dim+j] = (j<4)?1:0; h_b[dim+j] = (j<4)?0:1; }
    // Pair 2: opposite → -1
    for (int j = 0; j < dim; j++) { h_a[2*dim+j] = 1.0f; h_b[2*dim+j] = -1.0f; }
    // Pair 3: general
    for (int j = 0; j < dim; j++) { h_a[3*dim+j] = (float)(j+1); h_b[3*dim+j] = (float)(j+2); }

    float *d_a, *d_b, *d_r;
    CUDA_CHECK(cudaMalloc(&d_a, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n*dim*sizeof(float), cudaMemcpyHostToDevice));

    int smem = 3 * 256 * sizeof(float);
    cosine_similarity_kernel<<<n, 256, smem>>>(d_a, d_b, d_r, n, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_r, d_r, n*sizeof(float), cudaMemcpyDeviceToHost));

    int pass = 1;
    if (fabsf(h_r[0]-1.0f) > 0.01f) { printf("FAIL pair 0: %f\n", h_r[0]); pass=0; }
    if (fabsf(h_r[1]) > 0.01f) { printf("FAIL pair 1: %f\n", h_r[1]); pass=0; }
    if (fabsf(h_r[2]+1.0f) > 0.01f) { printf("FAIL pair 2: %f\n", h_r[2]); pass=0; }
    printf("Pair 3: %f\n", h_r[3]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    printf("Test Cosine Similarity: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_batch_predict() {
    printf("=== Test: Batch Predict (CUDA) ===\n");
    const int n = 3, dim = 4;
    float delta = 0.5f;

    float h_p[n*dim], h_t[n*dim], h_r[n*dim];
    for (int i = 0; i < n*dim; i++) { h_p[i] = 1.0f; h_t[i] = 3.0f; }

    float *d_p, *d_t, *d_r;
    CUDA_CHECK(cudaMalloc(&d_p, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_p, h_p, n*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t, h_t, n*dim*sizeof(float), cudaMemcpyHostToDevice));

    batch_predict_kernel<<<n, 256>>>(d_p, d_t, d_r, n, dim, delta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_r, d_r, n*dim*sizeof(float), cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < n*dim; i++) {
        if (fabsf(h_r[i]-2.0f) > 0.01f) { printf("FAIL [%d]=%f\n",i,h_r[i]); pass=0; }
    }

    cudaFree(d_p); cudaFree(d_t); cudaFree(d_r);
    printf("Test Batch Predict: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_balance_check() {
    printf("=== Test: Balance Check (CUDA) ===\n");
    const int n = 6;
    unsigned int h_in[n], h_out[n], h_res[n], h_count = 0;

    h_in[0]=5; h_out[0]=5; h_in[1]=10; h_out[1]=10; h_in[2]=3; h_out[2]=3;
    h_in[3]=5; h_out[3]=3; h_in[4]=0; h_out[4]=1; h_in[5]=7; h_out[5]=7;

    unsigned int *d_in, *d_out, *d_res, *d_count;
    CUDA_CHECK(cudaMalloc(&d_in, n*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_res, n*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, n*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    balance_check_kernel<<<1, n>>>(d_in, d_out, d_res, d_count, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_res, d_res, n*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int pass = 1;
    if (h_res[0]||h_res[1]||h_res[2]) { printf("FAIL: balanced\n"); pass=0; }
    if (!h_res[3]||!h_res[4]) { printf("FAIL: imbalanced\n"); pass=0; }
    if (h_res[5]) { printf("FAIL: room 5\n"); pass=0; }
    if (h_count != 2) { printf("FAIL: count=%u\n", h_count); pass=0; }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_res); cudaFree(d_count);
    printf("Test Balance Check: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_decay() {
    printf("=== Test: Decay (CUDA) ===\n");
    const int n = 8;
    float h_s[n], h_a[n];
    for (int i = 0; i < n; i++) { h_s[i] = 1.0f; h_a[i] = (float)i; }

    float *d_s, *d_a;
    CUDA_CHECK(cudaMalloc(&d_s, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_s, h_s, n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice));

    decay_kernel<<<1, n>>>(d_s, d_a, n, 0.1f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_s, d_s, n*sizeof(float), cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < n; i++) {
        float expected = expf(-0.1f*(float)i);
        if (fabsf(h_s[i]-expected) > 0.01f) { printf("FAIL [%d]=%f exp=%f\n",i,h_s[i],expected); pass=0; }
    }
    for (int i = 1; i < n; i++) {
        if (h_s[i] >= h_s[i-1]) { printf("FAIL monotonic\n"); pass=0; }
    }

    cudaFree(d_s); cudaFree(d_a);
    printf("Test Decay: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_vibe() {
    printf("=== Test: Vibe Compute (CUDA) ===\n");
    const int n = 2, dim = 4;
    float dt = 1.0f;

    float h_e[n*dim] = {1,0,0,0, 3,4,0,0};
    float h_v[n*dim] = {0,1,0,0, 0,0,1,0};
    float h_vb[n*dim];

    float *d_e, *d_v, *d_vb;
    CUDA_CHECK(cudaMalloc(&d_e, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vb, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_e, h_e, n*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, n*dim*sizeof(float), cudaMemcpyHostToDevice));

    int smem = 256 * sizeof(float);
    vibe_compute_kernel<<<n, 256, smem>>>(d_e, d_v, d_vb, n, dim, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_vb, d_vb, n*dim*sizeof(float), cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int r = 0; r < n; r++) {
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += h_vb[r*dim+j]*h_vb[r*dim+j];
        if (fabsf(norm-1.0f) > 0.01f) { printf("FAIL room %d norm=%f\n",r,norm); pass=0; }
    }

    cudaFree(d_e); cudaFree(d_v); cudaFree(d_vb);
    printf("Test Vibe Compute: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_correlation() {
    printf("=== Test: Correlation Matrix (CUDA) ===\n");
    const int nr = 3, dim = 4;

    float h_v[nr*dim] = {1,0,0,0, 0,1,0,0, 1,0,0,0};
    float h_m[nr*nr];

    float *d_v, *d_m;
    CUDA_CHECK(cudaMalloc(&d_v, nr*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, nr*nr*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, nr*dim*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(nr, nr);
    correlation_matrix_kernel<<<grid, 256>>>(d_v, d_m, nr, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_m, d_m, nr*nr*sizeof(float), cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < nr; i++) {
        if (fabsf(h_m[i*nr+i]-1.0f) > 0.01f) { printf("FAIL diag[%d]=%f\n",i,h_m[i*nr+i]); pass=0; }
    }
    if (fabsf(h_m[0*nr+1]) > 0.01f) { printf("FAIL sim01=%f\n",h_m[1]); pass=0; }
    if (fabsf(h_m[0*nr+2]-1.0f) > 0.01f) { printf("FAIL sim02=%f\n",h_m[2]); pass=0; }

    cudaFree(d_v); cudaFree(d_m);
    printf("Test Correlation Matrix: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_merge() {
    printf("=== Test: Merge Candidates (CUDA) ===\n");
    const int n = 4, dim = 4;
    float threshold = 0.9f;

    float h_e[n*dim] = {1,0,0,0, 1,0.01f,0,0, 0,1,0,0, 0,1,0.01f,0};
    float h_s[n] = {1,1,1,1};
    unsigned int h_c[n], h_count = 0;

    float *d_e, *d_s;
    unsigned int *d_c, *d_count;
    CUDA_CHECK(cudaMalloc(&d_e, n*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_e, h_e, n*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s, h_s, n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    merge_candidates_kernel<<<1, n>>>(d_e, d_s, d_c, d_count, n, dim, threshold);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, n*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int pass = 1;
    if (!h_c[0]) { printf("FAIL pair 0-1 should merge\n"); pass=0; }
    if (h_c[1]) { printf("FAIL pair 1-2 should not merge\n"); pass=0; }
    if (!h_c[2]) { printf("FAIL pair 2-3 should merge\n"); pass=0; }

    cudaFree(d_e); cudaFree(d_s); cudaFree(d_c); cudaFree(d_count);
    printf("Test Merge Candidates: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

static int test_e2e() {
    printf("=== Test: End-to-End CUDA (Tick → Predict → Balance → GC) ===\n");
    const int nr = 4, dim = 8;

    float h_emb[nr*dim], h_vel[nr*dim], h_pred[nr*dim], h_vibes[nr*dim];
    float h_str[nr], h_ages[nr];
    unsigned int h_zin[nr], h_zout[nr], h_bal[nr], h_imb = 0;
    unsigned int h_mc[nr], h_mcount = 0;

    for (int i = 0; i < nr; i++) {
        h_str[i] = 1.0f; h_ages[i] = 0.0f;
        h_zin[i] = 5; h_zout[i] = 5;
        for (int j = 0; j < dim; j++) {
            h_emb[i*dim+j] = sinf((float)(i*dim+j));
            h_vel[i*dim+j] = 0.1f * cosf((float)(i*dim+j));
            h_pred[i*dim+j] = h_emb[i*dim+j] + 0.5f;
        }
    }
    h_zout[2] = 3;

    float *d_emb, *d_vel, *d_pred, *d_vibes, *d_str, *d_ages;
    unsigned int *d_zin, *d_zout, *d_bal, *d_imb, *d_mc, *d_mcount;
    size_t eb = nr*dim*sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_emb, eb)); CUDA_CHECK(cudaMemcpy(d_emb, h_emb, eb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_vel, eb)); CUDA_CHECK(cudaMemcpy(d_vel, h_vel, eb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pred, eb)); CUDA_CHECK(cudaMemcpy(d_pred, h_pred, eb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_vibes, eb));
    CUDA_CHECK(cudaMalloc(&d_str, nr*sizeof(float))); CUDA_CHECK(cudaMemcpy(d_str, h_str, nr*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ages, nr*sizeof(float))); CUDA_CHECK(cudaMemcpy(d_ages, h_ages, nr*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_zin, nr*sizeof(unsigned int))); CUDA_CHECK(cudaMemcpy(d_zin, h_zin, nr*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_zout, nr*sizeof(unsigned int))); CUDA_CHECK(cudaMemcpy(d_zout, h_zout, nr*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_bal, nr*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_imb, sizeof(unsigned int))); CUDA_CHECK(cudaMemset(d_imb, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_mc, nr*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_mcount, sizeof(unsigned int))); CUDA_CHECK(cudaMemset(d_mcount, 0, sizeof(unsigned int)));

    // Step 1: Decay
    decay_kernel<<<1, nr>>>(d_str, d_ages, nr, 0.1f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Predict
    batch_predict_kernel<<<nr, 256>>>(d_emb, d_pred, d_pred, nr, dim, 0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Vibe
    vibe_compute_kernel<<<nr, 256, 256*sizeof(float)>>>(d_emb, d_vel, d_vibes, nr, dim, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Balance
    balance_check_kernel<<<1, nr>>>(d_zin, d_zout, d_bal, d_imb, nr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Merge
    merge_candidates_kernel<<<1, nr>>>(d_emb, d_str, d_mc, d_mcount, nr, dim, 0.9f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    CUDA_CHECK(cudaMemcpy(h_str, d_str, nr*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vibes, d_vibes, eb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bal, d_bal, nr*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_imb, d_imb, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < nr; i++) {
        if (fabsf(h_str[i]-1.0f) > 0.01f) { printf("FAIL str[%d]=%f\n",i,h_str[i]); pass=0; }
    }
    if (h_imb != 1) { printf("FAIL imb=%u\n",h_imb); pass=0; }
    if (!h_bal[2]) { printf("FAIL room 2 not flagged\n"); pass=0; }
    for (int i = 0; i < nr; i++) {
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += h_vibes[i*dim+j]*h_vibes[i*dim+j];
        if (fabsf(norm-1.0f) > 0.01f) { printf("FAIL vibe norm[%d]=%f\n",i,norm); pass=0; }
    }

    cudaFree(d_emb); cudaFree(d_vel); cudaFree(d_pred); cudaFree(d_vibes);
    cudaFree(d_str); cudaFree(d_ages); cudaFree(d_zin); cudaFree(d_zout);
    cudaFree(d_bal); cudaFree(d_imb); cudaFree(d_mc); cudaFree(d_mcount);

    printf("Test End-to-End: %s\n", pass?"PASSED":"FAILED");
    return pass?0:-1;
}

// Public runner
int run_all_tests() {
    int passed = 0, failed = 0;

    if (test_cosine_similarity() == 0) passed++; else failed++;
    if (test_batch_predict() == 0) passed++; else failed++;
    if (test_balance_check() == 0) passed++; else failed++;
    if (test_decay() == 0) passed++; else failed++;
    if (test_vibe() == 0) passed++; else failed++;
    if (test_correlation() == 0) passed++; else failed++;
    if (test_merge() == 0) passed++; else failed++;
    if (test_e2e() == 0) passed++; else failed++;

    printf("\n=================================\n");
    printf("Results: %d/8 passed, %d failed\n", passed, failed);
    return failed;
}
