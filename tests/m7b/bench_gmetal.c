/*
 * tests/m7b/bench_gmetal.c — informational CPU-vs-Metal microbench for
 * the GLM Metal backend (c/backend_gmetal.mm) at the real GLM-5.3-Flash
 * shapes. NOT a gate: prints timings only.
 *
 * Decode GEMV (M=1) and a small-prefill GEMM (M=32), effective weight
 * bytes/s. On unified memory both sides stream the same DRAM; the V4 m7b
 * finding (GPU at parity on GEMV shapes, win = dense compute off the
 * CPU) is expected to hold.
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
/* tok/s leg: the full GLM stack (m5g container) */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_GKDA_IMPLEMENTATION
#define APUS_GDSA_IMPLEMENTATION
#define APUS_GCACHE_IMPLEMENTATION
#define APUS_GMODEL_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bf16.h"
#include "fp8blk.h"
#include "gmodel.h"
#include "backend_gmetal.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint16_t rng_bf16(void) {
    float f = ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 8.0
               - 4.0);
    return apus_bf16_bits(f);
}

static void bench_bf16(const char *name, size_t M, size_t O, size_t K,
                       int reps) {
    uint16_t *w = malloc(O * K * sizeof(uint16_t));
    uint16_t *x = malloc(M * K * sizeof(uint16_t));
    uint16_t *y = malloc(M * O * sizeof(uint16_t));
    float *xf = malloc(M * K * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16();
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16();
    double wb = (double)O * K * sizeof(uint16_t);
    /* warmup + CPU */
    apus_bf16_gemm_mt(w, x, xf, y, M, O, K);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        apus_bf16_gemm_mt(w, x, xf, y, M, O, K);
    double tc = (now_s() - t0) / reps;
    /* Metal */
    if (apus_gmetal_bf16_gemm(w, x, y, M, O, K)) {
        printf("  %-28s  metal unsupported\n", name);
        free(w); free(x); free(y); free(xf);
        return;
    }
    t0 = now_s();
    for (int r = 0; r < reps; r++)
        apus_gmetal_bf16_gemm(w, x, y, M, O, K);
    double tm = (now_s() - t0) / reps;
    printf("  %-28s  cpu %7.3f ms (%5.1f GB/s)  metal %7.3f ms "
           "(%5.1f GB/s)  x%.2f\n", name,
           tc * 1e3, wb / tc / 1e9, tm * 1e3, wb / tm / 1e9, tc / tm);
    free(w); free(x); free(y); free(xf);
}

static void bench_fp8(const char *name, size_t M, size_t O, size_t K,
                      int reps) {
    size_t nkb = apus_fp8blk_nblocks(K);
    size_t nsb = apus_fp8blk_nblocks(O) * nkb;
    uint8_t *codes = malloc(O * K);
    float *scales = malloc(nsb * sizeof(float));
    uint16_t *x = malloc(M * K * sizeof(uint16_t));
    uint16_t *y = malloc(M * O * sizeof(uint16_t));
    uint16_t *wbuf = malloc(O * K * sizeof(uint16_t));
    float *xf = malloc(M * K * sizeof(float));
    for (size_t i = 0; i < O * K; i++) codes[i] = (uint8_t)rng_u64();
    for (size_t i = 0; i < nsb; i++) scales[i] = 0.01f;
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16();
    double wb = (double)O * K;   /* fp8 bytes streamed by the fused GPU
                                    path (the CPU path streams 2x after
                                    dequant) */
    /* CPU composition: dequant + gemm */
    apus_fp8blk_dequant(codes, scales, wbuf, O, K);
    apus_bf16_gemm_mt(wbuf, x, xf, y, M, O, K);
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        apus_fp8blk_dequant(codes, scales, wbuf, O, K);
        apus_bf16_gemm_mt(wbuf, x, xf, y, M, O, K);
    }
    double tc = (now_s() - t0) / reps;
    if (apus_gmetal_fp8blk_linear(codes, scales, x, y, M, O, K)) {
        printf("  %-28s  metal unsupported\n", name);
        free(codes); free(scales); free(x); free(y); free(wbuf); free(xf);
        return;
    }
    t0 = now_s();
    for (int r = 0; r < reps; r++)
        apus_gmetal_fp8blk_linear(codes, scales, x, y, M, O, K);
    double tm = (now_s() - t0) / reps;
    printf("  %-28s  cpu %7.3f ms (%5.1f GB/s fp8)  metal %7.3f ms "
           "(%5.1f GB/s fp8)  x%.2f\n", name,
           tc * 1e3, wb / tc / 1e9, tm * 1e3, wb / tm / 1e9, tc / tm);
    free(codes); free(scales); free(x); free(y); free(wbuf); free(xf);
}

/* decode tok/s on the m5g mini container, CPU vs Metal (hooks; floor 0
 * so the toy shapes offload — dispatch overhead dominates there, the
 * point of the measurement) */
static void bench_toks(const char *goldendir) {
    char err[256], merr[256], cpath[512], cfgpath[512];
    apus_gmetal_disable();
    setenv("APUS_GMETAL_MIN_KB", "0", 1);
    if (apus_gmetal_enable(merr, sizeof merr)) {
        fprintf(stderr, "bench: gmetal re-enable: %s\n", merr);
        return;
    }
    snprintf(cpath, sizeof cpath, "%s/container", goldendir);
    snprintf(cfgpath, sizeof cfgpath, "%s/config.json", goldendir);
    ApusGmodel *m = apus_gmodel_open(cpath, cfgpath, err, sizeof err);
    if (!m) {
        fprintf(stderr, "bench: gmodel open: %s\n", err);
        return;
    }
    size_t V = (size_t)apus_gmodel_config(m)->vocab_size;
    int32_t ids[24];
    for (int i = 0; i < 24; i++) ids[i] = (int32_t)(rng_u64() % V);
    uint16_t *logits = malloc(24 * V * sizeof(uint16_t));
    uint16_t *dlg = malloc(V * sizeof(uint16_t));
    ApusGmetalHooks saved = apus_gmetal_hooks;

    double res[2];
    for (int backend = 0; backend < 2; backend++) {
        if (backend) apus_gmetal_hooks = saved;
        else memset(&apus_gmetal_hooks, 0, sizeof apus_gmetal_hooks);
        ApusGmodelState *st = apus_gmodel_state_new(m, 24 + 48);
        apus_gmodel_prefill(m, st, ids, 24, logits, NULL, NULL);
        int32_t t = 0;
        double t0 = now_s();
        int n = 0;
        for (int step = 0; step < 48; step++) {
            const uint16_t *row = step ? dlg : logits + 23 * V;
            float best = -1e30f;
            int32_t bi = 0;
            for (size_t v = 0; v < V; v++) {
                float lv = apus_bf16_f32(row[v]);
                if (lv > best) { best = lv; bi = (int32_t)v; }
            }
            t = bi;
            apus_gmodel_decode_step(m, st, t, dlg, NULL, NULL);
            n++;
        }
        res[backend] = n / (now_s() - t0);
        apus_gmodel_state_free(st);
    }
    apus_gmetal_hooks = saved;
    printf("  m5g mini-model greedy decode:  cpu %.1f tok/s   metal %.1f "
           "tok/s  (x%.2f; dispatch-bound at toy shapes)\n",
           res[0], res[1], res[0] / res[1]);
    free(logits);
    free(dlg);
    apus_gmodel_close(m);
}

int main(void) {
    char err[256];
    if (apus_gmetal_enable(err, sizeof err)) {
        printf("bench_gmetal: Metal unavailable (%s)\n", err);
        return 0;
    }
    printf("bench_gmetal: GLM shapes, decode GEMV (M=1) + prefill GEMM "
           "(M=32); informational, not a gate\n");
    bench_bf16("expert gate/up 2048x4096 M1", 1, 2048, 4096, 20);
    bench_bf16("expert down 4096x2048 M1", 1, 4096, 2048, 20);
    bench_bf16("dense MLP 12288x4096 M1", 1, 12288, 4096, 10);
    bench_bf16("router 288x4096 M1", 1, 288, 4096, 20);
    bench_bf16("head 40960x4096 M1", 1, 40960, 4096, 5);
    bench_bf16("expert gate/up 2048x4096 M32", 32, 2048, 4096, 10);
    bench_bf16("head 40960x4096 M32", 32, 40960, 4096, 3);
    bench_fp8("dsa q_b fused 16384x1536 M1", 1, 16384, 1536, 10);
    bench_fp8("dsa o_proj fused 4096x16384 M1", 1, 4096, 16384, 5);
    bench_fp8("dsa o_proj fused 4096x16384 M32", 32, 4096, 16384, 3);
    bench_toks("tests/m5g/golden");
    apus_gmetal_disable();
    return 0;
}
