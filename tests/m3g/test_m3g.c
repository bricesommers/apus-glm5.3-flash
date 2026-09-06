/*
 * tests/m3g/test_m3g.c — M3 (GLM) kernel hard gate: c/fp8blk.h (E4M3 + F32
 * weight_scale_inv 128x128-block dequant -> BF16) and c/bf16.h (BF16
 * GEMV/GEMM, fp32 sequential accumulate), scalar vs NEON vs AVX2 vs mt,
 * all bitwise, against the numpy-oracle goldens (tests/m3g/gen_golden.py,
 * which uses tools/oracle.py's own codec — the M0-pinned semantics).
 *
 * Run from the repository root (golden fixtures under tests/m3g/golden/).
 * Prints FNV-1a digests; the Makefile diffs full output across
 * APUS_THREADS=1/4/8 (thread-count independence).
 */
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#include "bf16.h"
#include "fp8blk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0, failures = 0;
#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

/* --- file IO --------------------------------------------------------------*/

static unsigned char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(sz ? (size_t)sz : 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return NULL;
    }
    fclose(f);
    *out_len = sz;
    return buf;
}

/* manifest.txt key=value lookup (keys unique) */
static const char *man_find(const char *man, const char *key) {
    size_t kl = strlen(key);
    const char *p = man;
    while ((p = strstr(p, key)) != NULL) {
        if ((p == man || p[-1] == '\n') && p[kl] == '=')
            return p + kl + 1;
        p += kl;
    }
    printf("manifest key %s missing\n", key);
    exit(1);
}
static long man_long(const char *man, const char *key) {
    return strtol(man_find(man, key), NULL, 10);
}

/* --- FNV-1a 64 (the oracle's digest convention) ----------------------------*/

static uint64_t fnv1a(const void *data, size_t n) {
    const unsigned char *p = data;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t digest_hex(const char *s) {
    return strtoull(s, NULL, 16);
}

/* --- splitmix64 for the in-test shape sweep --------------------------------*/

static uint64_t sm64_state;
static uint64_t sm64(void) {
    uint64_t z = (sm64_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* ---------------------------------------------------------------------------*/

/* 1. BF16 widen/narrow exhaustive (all 65536 codes): widen is the exact
 *    <<16; narrow(widen(code)) == code; narrow is NaN-passthrough. */
static void test_bf16_roundtrip(void) {
    for (uint32_t c = 0; c < 65536; c++) {
        uint16_t code = (uint16_t)c;
        float f = apus_bf16_f32(code);
        uint32_t bits;
        memcpy(&bits, &f, 4);
        if (bits != (c << 16)) {
            CHECK(0, "widen %04x: got %08x", c, bits);
            continue;
        }
        uint16_t back = apus_bf16_bits(f);
        if (back != code)
            CHECK(0, "narrow(widen %04x): got %04x", c, back);
    }
    CHECK(1, "bf16 widen/narrow exhaustive done");
}

/* 2. E4M3 scalar decode exhaustive vs the ldexpf formula (bitwise), and
 *    NaN codes decode as +-480 (documented out-of-contract behavior — the
 *    SIMD expansions must match it). */
static void test_e4m3_decode(void) {
    for (uint32_t c = 0; c < 256; c++) {
        int e = (c >> 3) & 0xF, m = c & 7;
        float want = (e != 0) ? ldexpf((float)(8 + m), e - 10)
                              : ldexpf((float)m, -9);
        if (c & 0x80) want = -want;
        float got = apus_e4m3_dequant_f32((uint8_t)c);
        uint32_t gb, wb;
        memcpy(&gb, &got, 4);
        memcpy(&wb, &want, 4);
        if (gb != wb)
            CHECK(0, "e4m3 decode %02x: got %a want %a", c, got, want);
    }
    CHECK(1, "e4m3 decode exhaustive done (NaN codes -> +-480 pinned)");
}

/* 3. Dequant exhaustive: every code at every position of a 2-block row
 *    (K=256) x scale corners; scalar vs NEON vs dispatch bitwise. */
static void test_dequant_exhaustive(void) {
    static const struct { float s; const char *name; } scales[] = {
        { 1.0f, "1" }, { 0.5f, "0.5" }, { -2.0f, "-2" }, { 0.0f, "0" },
        { 0x1p-126f, "2^-126" }, { 0x1p-127f, "2^-127(sub)" },
        { 0x1p40f, "2^40" }, { 448.0f, "448" }, { 0x1p-9f, "2^-9" },
        { 3.14159e-30f, "3.14e-30" },
    };
    enum { K = 256, O = 4 };
    uint8_t codes[O * K];
    for (int o = 0; o < O; o++)
        for (int k = 0; k < K; k++)
            codes[o * K + k] = (uint8_t)((o * 64 + k) & 0xFF); /* all 256 */
    for (size_t si = 0; si < sizeof(scales) / sizeof(scales[0]); si++) {
        float ws[2 * 1] = { scales[si].s, scales[si].s };  /* nkb=2, nbo=1 */
        uint16_t ref[O * K], out_s[O * K], out_d[O * K];
        for (int o = 0; o < O; o++)
            for (int k = 0; k < K; k++)
                ref[o * K + k] = apus_bf16_bits(
                    apus_e4m3_dequant_f32(codes[o * K + k]) * scales[si].s);
        apus_fp8blk_dequant_scalar(codes, ws, out_s, O, K);
        apus_fp8blk_dequant(codes, ws, out_d, O, K);
        uint16_t out_m[O * K];
        apus_fp8blk_dequant_mt(codes, ws, out_m, O, K);
        for (int i = 0; i < O * K; i++) {
            if (out_s[i] != ref[i])
                CHECK(0, "deq scalar scale=%s i=%d code=%02x: %04x vs %04x",
                      scales[si].name, i, codes[i], out_s[i], ref[i]);
            if (out_d[i] != ref[i])
                CHECK(0, "deq dispatch scale=%s i=%d code=%02x: %04x vs %04x",
                      scales[si].name, i, codes[i], out_d[i], ref[i]);
            if (out_m[i] != ref[i])
                CHECK(0, "deq mt scale=%s i=%d code=%02x: %04x vs %04x",
                      scales[si].name, i, codes[i], out_m[i], ref[i]);
        }
#ifdef __ARM_NEON
        uint16_t out_n[O * K];
        apus_fp8blk_dequant_neon(codes, ws, out_n, O, K);
        for (int i = 0; i < O * K; i++)
            if (out_n[i] != ref[i])
                CHECK(0, "deq NEON scale=%s i=%d code=%02x: %04x vs %04x",
                      scales[si].name, i, codes[i], out_n[i], ref[i]);
#endif
    }
    CHECK(1, "dequant exhaustive done");
}

/* --- golden-driven checks ---------------------------------------------------*/

static char g_man[65536];

static void golden_dequant(const char *dir) {
    long ndeq = man_long(g_man, "ndeq");
    for (long i = 0; i < ndeq; i++) {
        char key[64], path[256];
        snprintf(key, sizeof key, "deq_%ld_O", i);
        long O = man_long(g_man, key);
        snprintf(key, sizeof key, "deq_%ld_K", i);
        long K = man_long(g_man, key);
        snprintf(key, sizeof key, "deq_%ld_digest", i);
        uint64_t want_digest = digest_hex(man_find(g_man, key));
        size_t nbo = (size_t)(O + 127) / 128, nbk = (size_t)(K + 127) / 128;
        long l1, l2, l3;
        snprintf(path, sizeof path, "%s/deq_%ld_codes.bin", dir, i);
        uint8_t *codes = (uint8_t *)read_file(path, &l1);
        snprintf(path, sizeof path, "%s/deq_%ld_scales.bin", dir, i);
        float *ws = (float *)read_file(path, &l2);
        snprintf(path, sizeof path, "%s/deq_%ld_out.bin", dir, i);
        uint16_t *want = (uint16_t *)read_file(path, &l3);
        CHECK(l1 == O * K && l2 == (long)(nbo * nbk * 4)
              && l3 == O * K * 2, "deq_%ld sizes", i);
        size_t n = (size_t)O * (size_t)K;
        uint16_t *s = malloc(n * 2), *d = malloc(n * 2), *mt = malloc(n * 2);
        apus_fp8blk_dequant_scalar(codes, ws, s, (size_t)O, (size_t)K);
        apus_fp8blk_dequant(codes, ws, d, (size_t)O, (size_t)K);
        apus_fp8blk_dequant_mt(codes, ws, mt, (size_t)O, (size_t)K);
        int ok_s = !memcmp(s, want, n * 2), ok_d = !memcmp(d, want, n * 2);
        CHECK(ok_s, "deq_%ld scalar vs oracle (%ldx%ld)", i, O, K);
        CHECK(ok_d, "deq_%ld dispatch vs oracle", i);
        CHECK(!memcmp(mt, want, n * 2), "deq_%ld mt vs oracle", i);
#ifdef __ARM_NEON
        uint16_t *nn = malloc(n * 2);
        apus_fp8blk_dequant_neon(codes, ws, nn, (size_t)O, (size_t)K);
        CHECK(!memcmp(nn, want, n * 2), "deq_%ld NEON vs oracle", i);
        free(nn);
#endif
        CHECK(fnv1a(s, n * 2) == want_digest, "deq_%ld digest", i);
        free(codes); free(ws); free(want); free(s); free(d); free(mt);
    }
}

/* Run one gemm case through every path; all outputs must be bitwise equal
 * to the oracle golden y. Returns the mt output in out_mt (for the thread
 * digest), or NULL on alloc failure. */
static uint16_t *gemm_one(const char *dir, long i, int is_linear) {
    char key[64], path[256];
    const char *pfx = is_linear ? "lin" : "gem";
    snprintf(key, sizeof key, "%s_%ld_M", pfx, i);
    long M = man_long(g_man, key);
    snprintf(key, sizeof key, "%s_%ld_O", pfx, i);
    long O = man_long(g_man, key);
    snprintf(key, sizeof key, "%s_%ld_K", pfx, i);
    long K = man_long(g_man, key);
    snprintf(key, sizeof key, "%s_%ld_digest", pfx, i);
    uint64_t want_digest = digest_hex(man_find(g_man, key));
    long l1, l2, l3, l4;
    uint16_t *wb;
    if (is_linear) {
        snprintf(path, sizeof path, "%s/lin_%ld_codes.bin", dir, i);
        uint8_t *codes = (uint8_t *)read_file(path, &l4);
        snprintf(path, sizeof path, "%s/lin_%ld_scales.bin", dir, i);
        float *ws = (float *)read_file(path, &l1);
        wb = malloc((size_t)O * (size_t)K * 2);
        apus_fp8blk_dequant(codes, ws, wb, (size_t)O, (size_t)K);
        free(codes); free(ws);
    } else {
        snprintf(path, sizeof path, "%s/gem_%ld_w.bin", dir, i);
        wb = (uint16_t *)read_file(path, &l4);
    }
    snprintf(path, sizeof path, "%s/%s_%ld_x.bin", dir, pfx, i);
    uint16_t *x = (uint16_t *)read_file(path, &l2);
    snprintf(path, sizeof path, "%s/%s_%ld_y.bin", dir, pfx, i);
    uint16_t *want = (uint16_t *)read_file(path, &l3);
    CHECK(l2 == M * K * 2 && l3 == M * O * 2, "%s_%ld sizes", pfx, i);

    size_t n = (size_t)M * (size_t)O;
    uint16_t *y_s = malloc(n * 2), *y_mt = malloc(n * 2);
    float *xf = malloc((size_t)M * (size_t)K * 4);
    apus_bf16_gemm_scalar(wb, x, y_s, (size_t)M, (size_t)O, (size_t)K);
    apus_bf16_gemm_mt(wb, x, xf, y_mt, (size_t)M, (size_t)O, (size_t)K);
    CHECK(!memcmp(y_s, want, n * 2), "%s_%ld scalar gemm vs oracle (%ld,%ld,%ld)",
          pfx, i, M, O, K);
    CHECK(!memcmp(y_mt, want, n * 2), "%s_%ld mt gemm vs oracle", pfx, i);
    CHECK(fnv1a(y_mt, n * 2) == want_digest, "%s_%ld digest", pfx, i);

    /* M-independence: row m of the GEMM is bitwise the GEMV of row m */
    if (M > 1 && !is_linear) {
        uint16_t *yv = malloc((size_t)O * 2);
        for (long m = 0; m < M; m++) {
            apus_bf16_gemv_scalar(wb, x + m * K, yv, (size_t)O, (size_t)K);
            if (memcmp(yv, y_s + m * O, (size_t)O * 2))
                CHECK(0, "%s_%ld M-independence row %ld", pfx, i, m);
        }
        free(yv);
    }
#ifdef __ARM_NEON
    uint16_t *y_n = malloc(n * 2);
    apus_bf16_gemm_neon(wb, x, xf, y_n, (size_t)M, (size_t)O, (size_t)K);
    CHECK(!memcmp(y_n, want, n * 2), "%s_%ld NEON gemm vs oracle", pfx, i);
    /* NEON GEMV bitwise == scalar GEMV */
    {
        uint16_t *yv_s = malloc((size_t)O * 2), *yv_n = malloc((size_t)O * 2);
        apus_bf16_gemv_scalar(wb, x, yv_s, (size_t)O, (size_t)K);
        apus_bf16_gemv_neon(wb, x, xf, yv_n, (size_t)O, (size_t)K);
        CHECK(!memcmp(yv_n, yv_s, (size_t)O * 2),
              "%s_%ld NEON gemv vs scalar", pfx, i);
        free(yv_s); free(yv_n);
    }
    free(y_n);
#endif
#if APUS_X86
    if (apus_x86_have_avx2()) {
        uint16_t *y_a = malloc(n * 2);
        apus_bf16_gemm_avx2(wb, x, xf, y_a, (size_t)M, (size_t)O, (size_t)K);
        CHECK(!memcmp(y_a, want, n * 2), "%s_%ld AVX2 gemm vs oracle", pfx, i);
        free(y_a);
    }
#endif
    free(wb); free(x); free(want); free(y_s); free(xf);
    return y_mt;
}

/* In-test shape sweep: scalar vs NEON/dispatch vs mt bitwise over random
 * bf16 inputs across tail shapes (chunk 32, row groups 8/4, M groups 4). */
static void test_shape_sweep(void) {
    static const size_t shapes[][3] = {  /* M, O, K */
        {1, 1, 1}, {1, 9, 17}, {3, 5, 31}, {2, 11, 32}, {5, 12, 33},
        {4, 8, 63}, {7, 16, 64}, {1, 24, 65}, {6, 13, 96}, {2, 40, 127},
        {8, 17, 128}, {3, 25, 129}, {1, 100, 200}, {5, 33, 257},
        {4, 64, 384}, {2, 128, 96}, {9, 7, 41}, {1, 200, 37},
    };
    sm64_state = 0x12345;
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        size_t M = shapes[si][0], O = shapes[si][1], K = shapes[si][2];
        uint16_t *w = malloc(O * K * 2), *x = malloc(M * K * 2);
        for (size_t i = 0; i < O * K; i++) w[i] = (uint16_t)sm64();
        for (size_t i = 0; i < M * K; i++) x[i] = (uint16_t)sm64();
        /* keep inputs finite: clamp to non-inf/non-NaN bf16 codes */
        for (size_t i = 0; i < O * K; i++)
            if ((w[i] & 0x7F80) == 0x7F80) w[i] = 0x3F80;
        for (size_t i = 0; i < M * K; i++)
            if ((x[i] & 0x7F80) == 0x7F80) x[i] = 0x3F80;
        uint16_t *y_s = malloc(M * O * 2), *y_mt = malloc(M * O * 2);
        float *xf = malloc(M * K * 4);
        apus_bf16_gemm_scalar(w, x, y_s, M, O, K);
        apus_bf16_gemm_mt(w, x, xf, y_mt, M, O, K);
        CHECK(!memcmp(y_mt, y_s, M * O * 2),
              "sweep mt==scalar M=%zu O=%zu K=%zu", M, O, K);
#ifdef __ARM_NEON
        uint16_t *y_n = malloc(M * O * 2);
        apus_bf16_gemm_neon(w, x, xf, y_n, M, O, K);
        CHECK(!memcmp(y_n, y_s, M * O * 2),
              "sweep NEON==scalar M=%zu O=%zu K=%zu", M, O, K);
        free(y_n);
#endif
#if APUS_X86
        if (apus_x86_have_avx2()) {
            uint16_t *y_a = malloc(M * O * 2);
            apus_bf16_gemm_avx2(w, x, xf, y_a, M, O, K);
            CHECK(!memcmp(y_a, y_s, M * O * 2),
                  "sweep AVX2==scalar M=%zu O=%zu K=%zu", M, O, K);
            free(y_a);
        }
#endif
        free(w); free(x); free(y_s); free(y_mt); free(xf);
    }
    CHECK(1, "shape sweep done");
}

int main(void) {
    const char *dir = "tests/m3g/golden";
    long mlen;
    unsigned char *man = read_file("tests/m3g/golden/manifest.txt", &mlen);
    CHECK(mlen < (long)sizeof(g_man), "manifest too big");
    memcpy(g_man, man, (size_t)mlen);
    g_man[mlen < (long)sizeof(g_man) ? mlen : (long)sizeof(g_man) - 1] = 0;
    free(man);

    printf("== m3g: GLM kernels (fp8blk dequant + bf16 gemv/gemm) ==\n");
#if APUS_X86
    printf("platform: x86-64 (AVX2 %s)\n",
           apus_x86_have_avx2() ? "available" : "NOT available — scalar");
#elif defined(__ARM_NEON)
    printf("platform: ARM NEON\n");
#else
    printf("platform: scalar only\n");
#endif

    test_bf16_roundtrip();
    test_e4m3_decode();
    test_dequant_exhaustive();

    golden_dequant(dir);

    /* gemm cases + composition cases; collect mt outputs for the thread
     * digest (FNV over all outputs, printed — diffed across APUS_THREADS) */
    long ngem = man_long(g_man, "ngem");
    long nlin = man_long(g_man, "nlin");
    uint64_t h = 14695981039346656037ull;
    for (long i = 0; i < ngem; i++) {
        uint16_t *y = gemm_one(dir, i, 0);
        size_t n;
        {
            char key[64];
            snprintf(key, sizeof key, "gem_%ld_M", i);
            long m = man_long(g_man, key);
            snprintf(key, sizeof key, "gem_%ld_O", i);
            n = (size_t)m * (size_t)man_long(g_man, key);
        }
        uint64_t hy = fnv1a(y, n * 2);
        h ^= hy; h *= 1099511628211ull;
        free(y);
    }
    for (long i = 0; i < nlin; i++) {
        uint16_t *y = gemm_one(dir, i, 1);
        size_t n;
        {
            char key[64];
            snprintf(key, sizeof key, "lin_%ld_M", i);
            long m = man_long(g_man, key);
            snprintf(key, sizeof key, "lin_%ld_O", i);
            n = (size_t)m * (size_t)man_long(g_man, key);
        }
        uint64_t hy = fnv1a(y, n * 2);
        h ^= hy; h *= 1099511628211ull;
        free(y);
    }

    test_shape_sweep();

#if APUS_X86
    printf("x86 avx2 path hits: %lu\n", apus_x86_avx2_hits());
#endif
    printf("mt output digest: %016llx\n", (unsigned long long)h);
    printf("test_m3g: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
