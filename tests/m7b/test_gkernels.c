/*
 * tests/m7b/test_gkernels.c — kernel-level verification of the GLM Metal
 * backend (c/backend_gmetal.mm) against the pinned CPU kernels
 * (c/bf16.h, c/fp8blk.h).
 *
 * THE GATE IS BITWISE (c/backend_gmetal.h numerics contract): the GLM CPU
 * kernels accumulate sequentially over k with two IEEE fp32 roundings per
 * element (product, then add — NO FMA), which a fast-math-off GPU thread
 * reproduces exactly; unlike the V4 fp8 path there is no SIMD canonical
 * order to mirror.
 *
 *   1. BF16 GEMV/GEMM (apus_gmetal_bf16_gemm) vs apus_bf16_gemv_mt /
 *      apus_bf16_gemm_mt: BITWISE on every output. Battery: the real GLM
 *      shapes (experts 2048x4096 / 4096x2048, dense MLP 12288x4096, DSA
 *      q_b 16384x1536, o_proj 4096x16384, router 288x4096, head slice
 *      8192x4096) + odd/partial shapes (K not a multiple of 32/128,
 *      O in {1,3}, M in {1,2,3,5,9}) + a scalar-anchor cross-check.
 *   2. Fused FP8-block linear (apus_gmetal_fp8blk_linear) vs the CPU
 *      composition apus_fp8blk_dequant + apus_bf16_gemm_mt: BITWISE,
 *      incl. ceil-shaped partial 128-blocks, all-256-code coverage
 *      (E4M3 subnormals + the NaN codes, which decode as +-480), and
 *      F32 scales across a wide exponent range.
 *   3. Edge values: zero rows, +-inf propagation (0*inf -> NaN parity),
 *      bf16-subnormal inputs, and a fp32-subnormal-accumulator probe
 *      (the one place GPU denormal flush could bite — measured and
 *      reported; normative data never lands there).
 *   4. APUS_GMETAL_MIN_KB floor: with a huge floor the HOOK declines
 *      (the direct entry point still runs); the hooked mt call then
 *      falls back and produces the pure-CPU bits.
 *   5. Instrumentation: zero-copy engaged, dispatches counted.
 *
 * Exit 0 iff all checks pass. Run from the repository root.
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bf16.h"
#include "fp8blk.h"
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

/* ---- deterministic PRNG (splitmix64) ---- */
static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rng_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_float(void) {   /* uniform in (-4, 4) */
    return ((double)(rng_u64() >> 40) / (double)(1ull << 24) * 8.0 - 4.0);
}
static uint8_t rng_byte(void) { return (uint8_t)(rng_u64() >> 56); }

/* random bf16 code: narrow a uniform float (realistic magnitudes) */
static uint16_t rng_bf16(void) { return apus_bf16_bits(rng_float()); }

/* ---- 1. BF16 GEMM battery: GPU vs CPU mt, BITWISE --------------------- */

static int g_bf16_case(size_t M, size_t O, size_t K) {
    uint16_t *w = malloc(O * K * sizeof(uint16_t));
    uint16_t *x = malloc(M * K * sizeof(uint16_t));
    uint16_t *yc = malloc(M * O * sizeof(uint16_t));
    uint16_t *yg = malloc(M * O * sizeof(uint16_t));
    float *xf = malloc(M * K * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16();
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16();
    memset(yc, 0xAA, M * O * sizeof(uint16_t));
    memset(yg, 0x55, M * O * sizeof(uint16_t));
    apus_bf16_gemm_mt(w, x, xf, yc, M, O, K);
    int rc = apus_gmetal_bf16_gemm(w, x, yg, M, O, K);
    int bad = rc != 0;
    size_t nbad = 0;
    if (!bad)
        for (size_t i = 0; i < M * O; i++)
            if (yc[i] != yg[i]) { nbad++; bad = 1; }
    CHECK(!bad, "bf16 gemm M%zu O%zu K%zu: rc=%d mismatches=%zu/%zu",
          M, O, K, rc, nbad, M * O);
    /* GEMV entry point must agree with the M=1 GEMM (CPU side pinned) */
    if (M == 1 && !rc) {
        uint16_t *yv = malloc(O * sizeof(uint16_t));
        apus_bf16_gemv_mt(w, x, xf, yv, O, K);
        CHECK(memcmp(yv, yg, O * sizeof(uint16_t)) == 0,
              "bf16 gemv O%zu K%zu != gemm M=1", O, K);
        free(yv);
    }
    free(w); free(x); free(yc); free(yg); free(xf);
    return !bad;
}

static void test_bf16_battery(void) {
    /* real GLM shapes (expert gate/up/down, dense MLP, DSA q_b/o_proj,
     * router gate, head slice) at M=1 (decode) and M=5 (prefill-ish) */
    static const size_t real_shapes[][2] = {
        { 2048, 4096 },   /* expert gate/up */
        { 4096, 2048 },   /* expert down */
        { 12288, 4096 },  /* dense MLP gate/up */
        { 4096, 12288 },  /* dense MLP down */
        { 16384, 1536 },  /* DSA q_b */
        { 4096, 16384 },  /* DSA o_proj */
        { 1536, 4096 },   /* DSA q_a */
        { 512, 4096 },    /* DSA kv_a */
        { 288, 4096 },    /* router gate */
        { 8192, 4096 },   /* LM head slice */
    };
    for (size_t i = 0; i < sizeof real_shapes / sizeof real_shapes[0]; i++) {
        g_bf16_case(1, real_shapes[i][0], real_shapes[i][1]);
        g_bf16_case(5, real_shapes[i][0], real_shapes[i][1]);
    }
    /* odd / partial shapes: K not a multiple of the 32-staging chunk or
     * the 128-scale block, degenerate O, assorted M */
    static const size_t odd[][3] = {
        { 1, 1, 1 }, { 1, 3, 1 }, { 1, 1, 7 }, { 2, 5, 31 },
        { 3, 17, 33 }, { 5, 33, 100 }, { 9, 7, 4095 }, { 1, 4096, 100 },
        { 4, 129, 128 }, { 2, 127, 129 }, { 7, 2048, 4097 },
    };
    for (size_t i = 0; i < sizeof odd / sizeof odd[0]; i++)
        g_bf16_case(odd[i][0], odd[i][1], odd[i][2]);
    /* scalar-anchor cross-check on two shapes (mt == scalar pinned in
     * m3g; here GPU == scalar directly) */
    for (int s = 0; s < 2; s++) {
        size_t M = s ? 3 : 1, O = s ? 77 : 129, K = s ? 300 : 1027;
        uint16_t *w = malloc(O * K * sizeof(uint16_t));
        uint16_t *x = malloc(M * K * sizeof(uint16_t));
        uint16_t *yc = malloc(M * O * sizeof(uint16_t));
        uint16_t *yg = malloc(M * O * sizeof(uint16_t));
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16();
        for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16();
        apus_bf16_gemm_scalar(w, x, yc, M, O, K);
        int rc = apus_gmetal_bf16_gemm(w, x, yg, M, O, K);
        CHECK(rc == 0 && memcmp(yc, yg, M * O * sizeof(uint16_t)) == 0,
              "bf16 gemm vs scalar anchor M%zu O%zu K%zu (rc=%d)",
              M, O, K, rc);
        free(w); free(x); free(yc); free(yg);
    }
}

/* ---- 2. fused FP8-block linear: GPU vs CPU composition, BITWISE ------- */

static int g_fp8_case(size_t M, size_t O, size_t K, int all_codes) {
    size_t nkb = apus_fp8blk_nblocks(K), nsb = apus_fp8blk_nblocks(O) * nkb;
    uint8_t *codes = malloc(O * K);
    float *scales = malloc(nsb * sizeof(float));
    uint16_t *x = malloc(M * K * sizeof(uint16_t));
    uint16_t *wbuf = malloc(O * K * sizeof(uint16_t));
    uint16_t *yc = malloc(M * O * sizeof(uint16_t));
    uint16_t *yg = malloc(M * O * sizeof(uint16_t));
    float *xf = malloc(M * K * sizeof(float));
    for (size_t i = 0; i < O * K; i++)
        codes[i] = all_codes ? (uint8_t)(i & 0xFF) : rng_byte();
    for (size_t i = 0; i < nsb; i++) {
        /* wide exponent range, positive/negative, incl. tiny (not 0/inf) */
        float m = 0.25f + 3.75f * (float)(rng_u64() >> 40)
                              / (float)(1ull << 24);
        int e = (int)(rng_u64() % 41) - 24;      /* 2^-24 .. 2^16 */
        scales[i] = ldexpf(m, e);
        if (rng_u64() & 1) scales[i] = -scales[i];
    }
    for (size_t i = 0; i < M * K; i++) x[i] = rng_bf16();
    memset(yc, 0xAA, M * O * sizeof(uint16_t));
    memset(yg, 0x55, M * O * sizeof(uint16_t));
    apus_fp8blk_dequant(codes, scales, wbuf, O, K);
    apus_bf16_gemm_mt(wbuf, x, xf, yc, M, O, K);
    int rc = apus_gmetal_fp8blk_linear(codes, scales, x, yg, M, O, K);
    int bad = rc != 0;
    size_t nbad = 0;
    if (!bad)
        for (size_t i = 0; i < M * O; i++)
            if (yc[i] != yg[i]) { nbad++; bad = 1; }
    CHECK(!bad, "fp8blk linear M%zu O%zu K%zu codes=%d: rc=%d "
          "mismatches=%zu/%zu", M, O, K, all_codes, rc, nbad, M * O);
    free(codes); free(scales); free(x); free(wbuf); free(yc); free(yg);
    free(xf);
    return !bad;
}

static void test_fp8blk_battery(void) {
    /* real GLM FP8 shapes (DSA linears) + ceil-shaped partial blocks */
    static const size_t shapes[][3] = {
        { 1, 1536, 4096 }, { 5, 1536, 4096 },     /* q_a */
        { 1, 16384, 1536 }, { 3, 16384, 1536 },   /* q_b */
        { 1, 512, 4096 },                          /* kv_a */
        { 1, 4096, 16384 }, { 2, 4096, 16384 },   /* o_proj */
        { 1, 200, 300 }, { 2, 200, 300 },         /* partial blocks */
        { 5, 129, 127 }, { 1, 128, 128 }, { 3, 127, 129 },
        { 1, 2048, 4096 },                        /* expert-sized */
    };
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        g_fp8_case(shapes[i][0], shapes[i][1], shapes[i][2], 0);
    /* all-256-code coverage (E4M3 subnormals + NaN codes -> +-480) on a
     * shape that cycles the whole code space per row */
    g_fp8_case(1, 512, 512, 1);
    g_fp8_case(2, 300, 768, 1);
}

/* ---- 3. edge values ----------------------------------------------------- */

static void test_edges(void) {
    size_t O = 64, K = 256;
    uint16_t *w = calloc(O * K, sizeof(uint16_t));
    uint16_t *x = calloc(K, sizeof(uint16_t));
    uint16_t *yc = malloc(O * sizeof(uint16_t));
    uint16_t *yg = malloc(O * sizeof(uint16_t));
    float *xf = malloc(K * sizeof(float));
    /* rows: all-zero weights; zero activations; +-inf weights (0*inf ->
     * NaN parity); bf16-subnormal inputs */
    uint16_t inf = 0x7F80, ninf = 0xFF80, sub = 0x0001, nsub = 0x8001;
    for (size_t k = 0; k < K; k++) {
        w[1 * K + k] = inf;
        w[2 * K + k] = ninf;
        w[3 * K + k] = (k & 1) ? inf : ninf;
        w[4 * K + k] = sub;
        w[5 * K + k] = nsub;
        w[6 * K + k] = (k & 1) ? sub : rng_bf16();
        x[k] = rng_bf16();
    }
    x[0] = 0x0000;   /* a zero activation against the inf rows */
    apus_bf16_gemv_mt(w, x, xf, yc, O, K);
    int rc = apus_gmetal_bf16_gemm(w, x, yg, 1, O, K);
    /* rows 0-3, 6+ (zero, +-inf, NaN parity, mixed subnormal+normal):
     * BITWISE. */
    int same = rc == 0;
    for (size_t i = 0; i < O && same; i++) {
        if (i == 4 || i == 5) continue;
        if (yc[i] != yg[i]) same = 0;
    }
    CHECK(same,
          "edges: inf/zero/mixed rows (rc=%d; y[1]=%04x/%04x y[3]=%04x/%04x)",
          rc, yc[1], yg[1], yc[3], yg[3]);
    CHECK((yg[1] & 0x7F80) == 0x7F80 && (yg[1] & 0x7F) != 0,
          "edges: 0*inf row must be NaN (%04x)", yg[1]);
    /* rows 4/5 (PURE bf16-subnormal weights): every product is
     * fp32-subnormal — the GPU flushes them to zero, the CPU accumulates
     * them (the measured DENORMAL-FLUSH class, tests/m7b/README.md:
     * |Δ| bounded by one bf16-subnormal ulp of output, ~2e-39 here;
     * unreachable from normative data, where no fp32-subnormal product
     * occurs). Measured, not gated bitwise. */
    double d4 = fabs((double)apus_bf16_f32(yc[4]) - apus_bf16_f32(yg[4]));
    double d5 = fabs((double)apus_bf16_f32(yc[5]) - apus_bf16_f32(yg[5]));
    printf("  denormal-flush class: pure-subnormal rows |Δ| = %g, %g "
           "(cpu %04x/%04x, gpu %04x/%04x)\n", d4, d5, yc[4], yc[5],
           yg[4], yg[5]);
    CHECK(rc == 0 && d4 <= 1e-37 && d5 <= 1e-37,
          "denormal-flush class exceeds its bound");

    /* fp32-subnormal accumulator probe: products ~2^-140 land below the
     * fp32 normal range; GPU denormal handling is implementation-defined
     * — MEASURE and report (normative data never reaches here: bf16
     * normals start at 2^-126 and a subnormal product needs two factors
     * below ~2^-63). */
    uint16_t tiny = 0x0001;   /* bf16 subnormal ~2^-133 */
    for (size_t k = 0; k < K; k++) { w[k] = tiny; x[k] = tiny; }
    apus_bf16_gemv_mt(w, x, xf, yc, 1, K);
    rc = apus_gmetal_bf16_gemm(w, x, yg, 1, 1, K);
    int bit_same = rc == 0 && yc[0] == yg[0];
    double dc = apus_bf16_f32(yc[0]), dg = apus_bf16_f32(yg[0]);
    printf("  fp32-subnormal probe: cpu=%g gpu=%g (%s)\n", dc, dg,
           bit_same ? "bitwise" : "diverges — denormal flush class");
    CHECK(bit_same || fabs(dc - dg) <= 1e-38,
          "subnormal probe divergence exceeds the flush bound");
    free(w); free(x); free(yc); free(yg); free(xf);
}

/* ---- 4. APUS_GMETAL_MIN_KB floor fail-soft ------------------------------ */

static void test_floor(void) {
    char err[256];
    apus_gmetal_disable();
    setenv("APUS_GMETAL_MIN_KB", "100000000", 1);   /* ~100 GB */
    CHECK(apus_gmetal_enable(err, sizeof err) == 0,
          "floor: re-enable: %s", err);
    size_t O = 512, K = 4096;
    uint16_t *w = malloc(O * K * sizeof(uint16_t));
    uint16_t *x = malloc(K * sizeof(uint16_t));
    uint16_t *y1 = malloc(O * sizeof(uint16_t));
    uint16_t *y2 = malloc(O * sizeof(uint16_t));
    float *xf = malloc(K * sizeof(float));
    for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16();
    for (size_t i = 0; i < K; i++) x[i] = rng_bf16();
    /* the hook must decline (below the floor)... */
    CHECK(apus_gmetal_hooks.bf16_gemm(w, x, y1, 1, O, K) != 0,
          "floor: oversized floor must decline the hook");
    /* ...and the hooked mt entry then produces the pure-CPU bits */
    uint64_t before = apus_gmetal_offload_bf16();
    apus_bf16_gemv_mt(w, x, xf, y1, O, K);
    CHECK(apus_gmetal_offload_bf16() == before,
          "floor: offload counter moved under the floor");
    ApusGmetalHooks saved = apus_gmetal_hooks;
    memset(&apus_gmetal_hooks, 0, sizeof apus_gmetal_hooks);
    apus_bf16_gemv_mt(w, x, xf, y2, O, K);
    CHECK(memcmp(y1, y2, O * sizeof(uint16_t)) == 0,
          "floor: hooked fallback != pure CPU");
    apus_gmetal_hooks = saved;
    /* the direct entry point ignores the floor */
    CHECK(apus_gmetal_bf16_gemm(w, x, y1, 1, O, K) == 0
          && memcmp(y1, y2, O * sizeof(uint16_t)) == 0,
          "floor: direct entry != CPU");
    /* restore the default for anything running after us */
    unsetenv("APUS_GMETAL_MIN_KB");
    apus_gmetal_disable();
    CHECK(apus_gmetal_hooks.bf16_gemm == NULL
          && apus_gmetal_hooks.fp8blk_linear == NULL,
          "floor: hooks not cleared by disable");
    CHECK(apus_gmetal_enable(err, sizeof err) == 0,
          "floor: final re-enable: %s", err);
    free(w); free(x); free(y1); free(y2); free(xf);
}

int main(void) {
    printf("test_gkernels: M7b GLM Metal backend kernel verification\n");
    char err[256];
    if (apus_gmetal_enable(err, sizeof err)) {
        printf("  Metal unavailable (%s) — skipping (not a failure on "
               "non-GPU hosts)\n", err);
        return 0;
    }
    printf("  Metal device ready\n");
    test_bf16_battery();
    test_fp8blk_battery();
    test_edges();
    printf("  after batteries: zero-copy wrapped %.1f MB, "
           "%llu dispatches\n",
           (double)apus_gmetal_bytes_wrapped() / 1048576.0,
           (unsigned long long)apus_gmetal_dispatches());
    CHECK(apus_gmetal_bytes_wrapped() > 0,
          "zero-copy wrapping never engaged (expected on Apple Silicon)");
    CHECK(apus_gmetal_dispatches() > 0, "no dispatches recorded");
    test_floor();
    apus_gmetal_disable();
    printf("test_gkernels: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
