/*
 * tests/m3g/bench_bf16.c — P5 decode-GEMV effective-bandwidth bench
 * (informational, NOT a gate; the bench-gio pattern). Reports effective
 * weight-streaming GB/s per decode-relevant GLM-5.3-Flash shape (M=1 GEMV)
 * for every pinned CPU path — scalar anchor, single-thread SIMD (NEON /
 * AVX2), and the mt row-parallel dispatch at the ambient APUS_THREADS —
 * plus two rooflines:
 *
 *   wadd8  widen bf16->f32 + strictly-sequential fp32 adds, 8 row chains
 *          interleaved (the GEMV's dependency structure WITHOUT the
 *          multiply and the product staging): the numerics-preserving
 *          ceiling of ANY bitwise-safe kernel on this shape.
 *   read   8-way independent u32 read-sum (no FP at all): the per-process
 *          streaming-read ceiling under current machine conditions.
 *
 * Reading: when mt ~= read, the decode GEMV is memory-bound and no
 * bitwise-safe kernel change can help; when neon1 << wadd8 there is
 * per-core headroom a scheduling-only change could harvest.
 *
 * Shapes (O x K): expert gate/up + down (moe_intermediate 2048, dim 4096),
 * dense-MLP gate/up, router, lm_head slice, DSA kv_b + q_b, a KDA-scale
 * projection. Numerics are irrelevant here (no gate); the kernels are the
 * production m3g ones.
 *
 * Env: APUS_THREADS (pool lanes for the mt leg), APUS_BF16_BENCH_REPS
 * (override the per-size rep counts).
 *
 * Run from the repository root:  make bench-m3g
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#include "bf16.h"
#include "fp8blk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* Roofline: widen + strictly-sequential adds, 8 independent row chains
 * (the GEMV kernel's ILP structure) but no multiply/staging. */
static void roofline_wadd8(const uint16_t *w, uint16_t *y,
                           size_t O, size_t K) {
    size_t o = 0;
    for (; o + 8 <= O; o += 8) {
        const uint16_t *wr[8];
        float a[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int r = 0; r < 8; r++) wr[r] = w + (o + r) * K;
        for (size_t k = 0; k < K; k++)
            for (int r = 0; r < 8; r++)
                a[r] += apus_bf16_f32(wr[r][k]);
        for (int r = 0; r < 8; r++) y[o + r] = apus_bf16_bits(a[r]);
    }
    for (; o < O; o++) {
        const uint16_t *wr = w + o * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++) acc += apus_bf16_f32(wr[k]);
        y[o] = apus_bf16_bits(acc);
    }
}

/* Roofline: pure streaming read (8 independent u32 chains). */
static uint64_t roofline_read(const uint16_t *w, size_t n) {
    const uint32_t *p = (const uint32_t *)w;
    size_t nw = n / 2, i = 0;
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (; i + 8 <= nw; i += 8) {
        s0 += p[i + 0]; s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3];
        s4 += p[i + 4]; s5 += p[i + 5]; s6 += p[i + 6]; s7 += p[i + 7];
    }
    for (; i < nw; i++) s0 += p[i];
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}

typedef struct { const char *name; size_t O, K; } BenchShape;

int main(void) {
    const BenchShape shapes[] = {
        { "expert gate/up", 2048, 4096 },
        { "expert down   ", 4096, 2048 },
        { "dense mlp g/u ", 12288, 4096 },
        { "router       ", 288, 4096 },
        { "kda proj     ", 8192, 4096 },
        { "dsa q_b      ", 16384, 1536 },
        { "dsa kv_b     ", 32768, 512 },
        { "head slice   ", 40960, 4096 },
    };
    const char *re = getenv("APUS_BF16_BENCH_REPS");
    int reps_ovr = re ? atoi(re) : 0;
    printf("bench_bf16: decode GEMV (M=1) effective GB/s, APUS_THREADS=%d "
           "(informational, not a gate)\n", apus_pool_threads());
    printf("  %-14s %12s  %7s %7s %7s %7s %8s\n",
           "shape", "dims", "scal", "simd1", "mt", "wadd8", "read");
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t O = shapes[s].O, K = shapes[s].K;
        uint16_t *w = malloc(O * K * sizeof(uint16_t));
        uint16_t *x = malloc(K * sizeof(uint16_t));
        uint16_t *y = malloc(O * sizeof(uint16_t));
        float *xf = malloc(K * sizeof(float));
        if (!w || !x || !y || !xf) { fprintf(stderr, "oom\n"); return 1; }
        for (size_t i = 0; i < O * K; i++) w[i] = rng_bf16();
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16();
        double wb = (double)O * K * 2.0;
        int reps = reps_ovr > 0 ? reps_ovr
                 : O * K > (size_t)64 << 20 ? 3
                 : O * K > (size_t)8 << 20 ? 10 : 30;
        double t0, t_scal, t_simd = -1.0, t_mt, t_wadd, t_read;

        apus_bf16_gemv_scalar(w, x, y, O, K);   /* warm */
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            apus_bf16_gemv_scalar(w, x, y, O, K);
        t_scal = (now_s() - t0) / reps;

#ifdef __ARM_NEON
        apus_bf16_gemv_neon(w, x, xf, y, O, K);
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            apus_bf16_gemv_neon(w, x, xf, y, O, K);
        t_simd = (now_s() - t0) / reps;
#elif APUS_X86
        if (apus_x86_have_avx2()) {
            apus_bf16_gemv_avx2(w, x, xf, y, O, K);
            t0 = now_s();
            for (int r = 0; r < reps; r++)
                apus_bf16_gemv_avx2(w, x, xf, y, O, K);
            t_simd = (now_s() - t0) / reps;
        }
#endif

        apus_bf16_gemv_mt(w, x, xf, y, O, K);
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            apus_bf16_gemv_mt(w, x, xf, y, O, K);
        t_mt = (now_s() - t0) / reps;

        roofline_wadd8(w, y, O, K);
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            roofline_wadd8(w, y, O, K);
        t_wadd = (now_s() - t0) / reps;

        uint64_t guard = 0;
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            guard += roofline_read(w, O * K);
        t_read = (now_s() - t0) / reps;
        if (guard == 42) printf("  (impossible)\n");

        printf("  %-14s %6zux%-5zu  %7.1f %7.1f %7.1f %7.1f %8.1f\n",
               shapes[s].name, O, K,
               wb / t_scal / 1e9,
               t_simd > 0 ? wb / t_simd / 1e9 : -1.0,
               wb / t_mt / 1e9,
               wb / t_wadd / 1e9, wb / t_read / 1e9);
        free(w); free(x); free(y); free(xf);
    }

    /* P5: the DSA FP8 projections (dequant -> BF16 -> GEMV per token in CPU
     * mode). deq1 = single-thread dispatch (the pre-P5 path; still what the
     * gcache I/O workers use), deq_mt = the P5 threaded dequant. */
    const BenchShape fshapes[] = {
        { "dsa q_a ", 1536, 4096 },
        { "dsa q_b ", 16384, 1536 },
        { "dsa kv_a", 512, 4096 },
        { "dsa o_pr", 4096, 16384 },
    };
    printf("  fp8-linear legs (fp8-read GB/s; gemv col = bf16 GB/s):\n");
    printf("  %-14s %12s  %7s %7s %7s\n", "shape", "dims", "deq1",
           "deq_mt", "gemv_mt");
    for (size_t s = 0; s < sizeof fshapes / sizeof fshapes[0]; s++) {
        size_t O = fshapes[s].O, K = fshapes[s].K;
        size_t nsb = apus_fp8blk_nblocks(O) * apus_fp8blk_nblocks(K);
        uint8_t *codes = malloc(O * K);
        float *scales = malloc(nsb * sizeof(float));
        uint16_t *wbuf = malloc(O * K * sizeof(uint16_t));
        uint16_t *x = malloc(K * sizeof(uint16_t));
        uint16_t *y = malloc(O * sizeof(uint16_t));
        float *xf = malloc(K * sizeof(float));
        if (!codes || !scales || !wbuf || !x || !y || !xf) {
            fprintf(stderr, "oom\n"); return 1;
        }
        memset(codes, 0x31, O * K);
        for (size_t i = 0; i < nsb; i++) scales[i] = 0.01f;
        for (size_t i = 0; i < K; i++) x[i] = rng_bf16();
        int reps = reps_ovr > 0 ? reps_ovr
                 : O * K > (size_t)32 << 20 ? 5 : 15;
        double fb = (double)O * K, t0;

        apus_fp8blk_dequant(codes, scales, wbuf, O, K);
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            apus_fp8blk_dequant(codes, scales, wbuf, O, K);
        double td1 = (now_s() - t0) / reps;

        apus_fp8blk_dequant_mt(codes, scales, wbuf, O, K);
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            apus_fp8blk_dequant_mt(codes, scales, wbuf, O, K);
        double tdm = (now_s() - t0) / reps;

        apus_bf16_gemv_mt(wbuf, x, xf, y, O, K);
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            apus_bf16_gemv_mt(wbuf, x, xf, y, O, K);
        double tg = (now_s() - t0) / reps;

        printf("  %-14s %6zux%-5zu  %7.1f %7.1f %7.1f\n",
               fshapes[s].name, O, K,
               fb / td1 / 1e9, fb / tdm / 1e9, 2 * fb / tg / 1e9);
        free(codes); free(scales); free(wbuf); free(x); free(y); free(xf);
    }
    return 0;
}
