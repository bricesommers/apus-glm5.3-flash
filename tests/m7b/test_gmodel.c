/*
 * tests/m7b/test_gmodel.c — the GLM Metal model gate: CPU-vs-Metal on the
 * m5g synthetic container (tests/m5g/golden, regenerate with golden-m5g).
 *
 * THE GATE IS BITWISE (c/backend_gmetal.h): the shaders reproduce the
 * pinned CPU kernels' sequential-k two-rounding order exactly, so the
 * FULL logit stream — prefill [s, V] + a greedy decode chain — must be
 * memcmp-identical with and without the Metal hooks, in BOTH the eager
 * and the tiered (1-slot cache, eviction churn + payload recycling at
 * every layer_end) wirings. No teacher-forcing, no tolerance tier: the
 * only measured divergence class (fp32-subnormal intermediates, the
 * denormal flush — test_gkernels.c) is unreachable from fixture or
 * normative data.
 *
 * Legs (all on the same deterministic prompt):
 *   A. eager CPU vs eager Metal — bitwise; offload counters prove the
 *      GPU path actually ran (bf16 AND fp8blk — the fixture has a DSA
 *      layer).
 *   B. tiered 1-slot CPU vs tiered Metal — bitwise (exercises the
 *      ephemeral-wrap expert path across gcache recycling; Metal holds
 *      no buffer across layer_end by construction).
 *   C. Metal tiered run twice — bitwise (GPU determinism).
 *   D. eager CPU == tiered CPU — bitwise (the m6g neutrality, re-asserted
 *      through this binary).
 *
 * APUS_GMETAL_MIN_KB=0 is set in-process so the tiny fixture shapes are
 * offloaded (the floor is a perf heuristic; the bitwise contract is
 * size-independent).
 *
 * Usage: test_gmodel [golden-dir]   (default tests/m5g/golden)
 * Prints one digest line per leg (the Makefile diffs APUS_THREADS=1/4/8).
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_GKDA_IMPLEMENTATION
#define APUS_GDSA_IMPLEMENTATION
#define APUS_GCACHE_IMPLEMENTATION
#define APUS_GMODEL_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gmodel.h"
#include "backend_gmetal.h"

static int failures = 0;
static long checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

#define PRE_N 70   /* crosses the 64-token chunked-KDA boundary */
#define DEC_N 12

/* deterministic prompt ids (splitmix64) */
static void make_ids(int32_t *ids, size_t n, int32_t V) {
    uint64_t s = 0xB5297A4D3F84D5B5ull;
    for (size_t i = 0; i < n; i++) {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        ids[i] = (int32_t)((z ^ (z >> 31)) % (uint64_t)V);
    }
}

static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

/* prefill PRE_N ids + DEC_N greedy steps; logits_out [(PRE_N+DEC_N)*V]
 * codes. Returns the stream digest. */
static uint64_t run_stream(ApusGmodel *m, const int32_t *ids,
                           uint16_t *logits_out, size_t *n_out) {
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    size_t V = (size_t)c->vocab_size;
    ApusGmodelState *st = apus_gmodel_state_new(m, PRE_N + DEC_N);
    uint16_t *lg = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));
    if (apus_gmodel_prefill(m, st, ids, PRE_N, lg, NULL, NULL)) {
        CHECK(0, "prefill failed");
        exit(1);
    }
    int32_t t = 0;
    for (size_t i = 0; i < DEC_N; i++) {
        const uint16_t *row = lg + (PRE_N - 1 + i) * V;
        float best = -1e30f;
        int32_t bi = 0;
        for (size_t v = 0; v < V; v++) {
            float lv = apus_bf16_f32(row[v]);
            if (lv > best) { best = lv; bi = (int32_t)v; }
        }
        t = bi;
        if (apus_gmodel_decode_step(m, st, t, lg + (PRE_N + i) * V,
                                    NULL, NULL)) {
            CHECK(0, "decode step %zu failed", i);
            exit(1);
        }
    }
    memcpy(logits_out, lg, (PRE_N + DEC_N) * V * sizeof(uint16_t));
    *n_out = PRE_N + DEC_N;
    uint64_t h = fnv1a(lg, (PRE_N + DEC_N) * V * sizeof(uint16_t),
                       0xCBF29CE484222325ull);
    free(lg);
    apus_gmodel_state_free(st);
    return h;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/m5g/golden";
    char err[256], cpath[512], cfgpath[512];
    snprintf(cpath, sizeof cpath, "%s/container", dir);
    snprintf(cfgpath, sizeof cfgpath, "%s/config.json", dir);
    printf("test_gmodel: M7b GLM Metal model gate (CPU vs Metal, %s)\n",
           dir);

    char merr[256];
    if (apus_gmetal_enable(merr, sizeof merr)) {
        printf("  Metal unavailable (%s) — skipping (not a failure on "
               "non-GPU hosts)\n", merr);
        return 0;
    }
    apus_gmetal_disable();   /* enabled per-leg below */

    /* the tiny fixture shapes must be offloaded: floor 0, M-gate off
     * (both read at enable, per-leg) */
    setenv("APUS_GMETAL_MIN_KB", "0", 1);
    setenv("APUS_GMETAL_MAX_M", "0", 1);

    /* ---- eager open ---- */
    ApusGmodel *me = apus_gmodel_open(cpath, cfgpath, err, sizeof err);
    if (!me) { CHECK(0, "eager open: %s", err); return 1; }
    size_t V = (size_t)apus_gmodel_config(me)->vocab_size;
    int32_t ids[PRE_N];
    make_ids(ids, PRE_N, (int32_t)V);
    size_t nlog = 0;
    uint16_t *cpu_e = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));
    uint16_t *gpu_e = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));

    /* A. eager CPU vs eager Metal */
    uint64_t d_cpu_e = run_stream(me, ids, cpu_e, &nlog);
    CHECK(apus_gmetal_enable(merr, sizeof merr) == 0,
          "metal enable: %s", merr);
    uint64_t d_gpu_e = run_stream(me, ids, gpu_e, &nlog);
    uint64_t off_b = apus_gmetal_offload_bf16();
    uint64_t off_f = apus_gmetal_offload_fp8blk();
    CHECK(off_b > 0, "eager metal: no bf16 ops offloaded");
    CHECK(off_f > 0, "eager metal: no fp8blk ops offloaded (DSA layer?)");
    CHECK(memcmp(cpu_e, gpu_e, nlog * V * sizeof(uint16_t)) == 0,
          "eager: CPU vs Metal logit stream differs");
    printf("  eager:   cpu %016llx  metal %016llx  (%llu bf16 + %llu "
           "fp8blk ops offloaded, %llu dispatches)\n",
           (unsigned long long)d_cpu_e, (unsigned long long)d_gpu_e,
           (unsigned long long)off_b, (unsigned long long)off_f,
           (unsigned long long)apus_gmetal_dispatches());

    /* ---- tiered open (1 slot/layer: eviction churn + payload recycling,
     * synchronous I/O for determinism of the wiring) ---- */
    ApusGmodelTierCfg tc;
    memset(&tc, 0, sizeof tc);
    tc.tiered = 1;
    tc.cache_bytes = (size_t)4 << 20;
    tc.slots_per_layer = 1;
    tc.io_threads = -1;
    ApusGmodel *mt = apus_gmodel_open2(cpath, cfgpath, &tc, err,
                                       sizeof err);
    if (!mt) { CHECK(0, "tiered open: %s", err); return 1; }
    uint16_t *cpu_t = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));
    uint16_t *gpu_t = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));
    uint16_t *gpu_t2 = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));

    /* B. tiered CPU vs tiered Metal */
    apus_gmetal_disable();
    uint64_t d_cpu_t = run_stream(mt, ids, cpu_t, &nlog);
    CHECK(apus_gmetal_enable(merr, sizeof merr) == 0,
          "metal re-enable: %s", merr);
    uint64_t d_gpu_t = run_stream(mt, ids, gpu_t, &nlog);
    CHECK(memcmp(cpu_t, gpu_t, nlog * V * sizeof(uint16_t)) == 0,
          "tiered 1-slot: CPU vs Metal logit stream differs");
    CHECK(apus_gmetal_offload_bf16() > 0,
          "tiered metal: no expert bf16 ops offloaded");

    /* C. Metal determinism */
    uint64_t d_gpu_t2 = run_stream(mt, ids, gpu_t2, &nlog);
    CHECK(memcmp(gpu_t, gpu_t2, nlog * V * sizeof(uint16_t)) == 0,
          "tiered metal: repeated run differs (GPU nondeterminism)");
    printf("  tiered1: cpu %016llx  metal %016llx  rerun %016llx\n",
           (unsigned long long)d_cpu_t, (unsigned long long)d_gpu_t,
           (unsigned long long)d_gpu_t2);

    /* D. eager CPU == tiered CPU (the m6g neutrality, re-asserted) */
    CHECK(memcmp(cpu_e, cpu_t, nlog * V * sizeof(uint16_t)) == 0,
          "eager CPU != tiered CPU (m6 neutrality regression)");
    printf("  neutral: eager-cpu %016llx == tiered-cpu %016llx\n",
           (unsigned long long)d_cpu_e, (unsigned long long)d_cpu_t);

    /* E. P4 persistent wraps: open WITH Metal enabled -> the model-owned
     * dense weights are registered at open (c/gmodel.h apus_gm_reg ->
     * apus_gmetal_register_region) and every op reuses the persistent
     * zero-copy wrap. The logit stream stays memcmp-identical; the model
     * is closed while the backend is still enabled (exercises unregister
     * before free). */
    ApusGmodel *mr = apus_gmodel_open2(cpath, cfgpath, &tc, err,
                                       sizeof err);
    if (!mr) { CHECK(0, "registered open: %s", err); return 1; }
    uint64_t pinned = apus_gmetal_bytes_pinned();
    CHECK(pinned > 0, "P4: no weight regions registered at metal-open");
    uint16_t *gpu_r = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));
    uint64_t d_gpu_r = run_stream(mr, ids, gpu_r, &nlog);
    CHECK(memcmp(cpu_t, gpu_r, nlog * V * sizeof(uint16_t)) == 0,
          "P4 registered-wrap stream != tiered CPU stream");
    apus_gmodel_close(mr);
    CHECK(apus_gmetal_bytes_pinned() == 0,
          "P4: registered wraps not released at close (%llu B left)",
          (unsigned long long)apus_gmetal_bytes_pinned());
    printf("  persist: metal-open %016llx (%.1f MiB pinned wraps, "
           "released at close)\n", (unsigned long long)d_gpu_r,
           (double)pinned / 1048576.0);
    free(gpu_r);

    /* F. P4 M-gate at its DEFAULT (APUS_GMETAL_MAX_M=1): decode GEMVs
     * offload, the prefill GEMMs (M=70 here) run the pinned CPU kernels;
     * the stream stays memcmp-identical. */
    apus_gmetal_disable();
    setenv("APUS_GMETAL_MAX_M", "1", 1);
    CHECK(apus_gmetal_enable(merr, sizeof merr) == 0,
          "metal re-enable (M-gate): %s", merr);
    uint64_t off_f0 = apus_gmetal_offload_bf16();
    uint16_t *gpu_m1 = malloc((PRE_N + DEC_N) * V * sizeof(uint16_t));
    uint64_t d_gpu_m1 = run_stream(mt, ids, gpu_m1, &nlog);
    CHECK(memcmp(cpu_t, gpu_m1, nlog * V * sizeof(uint16_t)) == 0,
          "P4 M-gate default: stream != tiered CPU stream");
    CHECK(apus_gmetal_offload_bf16() > off_f0,
          "P4 M-gate default: no decode GEMVs offloaded");
    printf("  m-gate:  max_m=1 %016llx (%llu decode GEMVs offloaded)\n",
           (unsigned long long)d_gpu_m1,
           (unsigned long long)(apus_gmetal_offload_bf16() - off_f0));
    free(gpu_m1);
    apus_gmetal_disable();
    setenv("APUS_GMETAL_MAX_M", "0", 1);   /* restore for a re-enable */

    printf("  digest:  %016llx (all legs)\n", (unsigned long long)d_cpu_e);

    apus_gmetal_disable();
    apus_gmodel_close(me);
    apus_gmodel_close(mt);
    free(cpu_e); free(gpu_e); free(cpu_t); free(gpu_t); free(gpu_t2);
    printf("test_gmodel: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
