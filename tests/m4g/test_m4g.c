/*
 * tests/m4g/test_m4g.c — M4a (GLM) sublayer hard gate: c/gmhc.h (mHC maps,
 * Sinkhorn-20, collapse/expand/head) and c/gmoe.h (sigmoid router,
 * swiglu experts, MoE forward) against the numpy-oracle goldens
 * (tests/m4g/gen_golden.py — the M0 oracle's own sublayer functions in
 * f32-faithful mode).
 *
 * Gate tiers (tests/m4g/README.md):
 *   BITWISE   — host expf == numpy float32 exp (probed at runtime):
 *               every golden compared memcmp-bitwise (f32 goldens incl.
 *               the pre-round values; BF16 outputs widened — equal iff the
 *               codes are equal; selections always bitwise).
 *   TOLERANCE — otherwise: f32 goldens rel err <= 1e-5 (the documented
 *               host-transcendental class), selections STILL bitwise
 *               (fixtures hold a >= 1e-4 top-k boundary margin).
 *
 * Run from the repository root (golden fixtures under tests/m4g/golden/).
 * Prints FNV-1a digests of the C outputs; the Makefile diffs full output
 * across APUS_THREADS=1/4/8 (thread-count independence).
 */
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#include "gmhc.h"
#include "gmoe.h"
#include "bf16.h"
#include "fp8blk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NORM_EPS 1e-5f   /* rms_norm_eps (mHC norm-before-fn) */
#define HC_EPS   1e-6f   /* hc_eps */
#define ITERS    20      /* hc_sinkhorn_iters */
#define RSCALE   2.5f    /* routed_scaling_factor */
#define TOL      1e-5f   /* tolerance-tier relative bound */

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

static float *load_f32(const char *dir, const char *name, size_t n) {
    char path[512];
    long len;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    unsigned char *b = read_file(path, &len);
    if ((size_t)len != n * sizeof(float)) {
        printf("%s: size mismatch (%ld != %zu)\n", path, len,
               n * sizeof(float));
        exit(1);
    }
    return (float *)b;
}

static uint16_t *load_u16(const char *dir, const char *name, size_t n) {
    char path[512];
    long len;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    unsigned char *b = read_file(path, &len);
    if ((size_t)len != n * sizeof(uint16_t)) {
        printf("%s: size mismatch (%ld != %zu)\n", path, len,
               n * sizeof(uint16_t));
        exit(1);
    }
    return (uint16_t *)b;
}

static uint8_t *load_u8(const char *dir, const char *name, size_t n) {
    char path[512];
    long len;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    unsigned char *b = read_file(path, &len);
    if ((size_t)len != n) {
        printf("%s: size mismatch (%ld != %zu)\n", path, len, n);
        exit(1);
    }
    return b;
}

static int32_t *load_i32(const char *dir, const char *name, size_t n) {
    return (int32_t *)load_f32(dir, name, n);   /* same 4-byte size check */
}

/* --- rolling FNV-1a 64 over the C outputs ---------------------------------*/

static uint64_t dg_state = 14695981039346656037ull;
static void dg_add(const void *data, size_t n) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++) {
        dg_state ^= p[i];
        dg_state *= 1099511628211ull;
    }
}

/* --- comparison (gate tier) ------------------------------------------------*/

static int bitwise_mode = 1;

/* f32 golden comparison. Returns max rel err; one CHECK per array. */
static void cmp_f32(const char *what, const float *got, const float *want,
                    size_t n) {
    if (bitwise_mode) {
        if (memcmp(got, want, n * sizeof(float)) != 0) {
            size_t bad = 0;
            for (size_t i = 0; i < n; i++)
                if (memcmp(got + i, want + i, 4) != 0) bad++;
            CHECK(0, "%s: %zu/%zu f32 elements differ (bitwise)", what,
                  bad, n);
        } else {
            CHECK(1, "%s", what);
        }
        return;
    }
    double maxrel = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        double rel = d / fmax(1e-30, fabs((double)want[i]));
        if (rel > maxrel) maxrel = rel;
        if (rel > TOL) bad++;
    }
    CHECK(bad == 0, "%s: %zu/%zu over tol (max rel %.3g)", what, bad, n,
          maxrel);
}

/* BF16-code output vs an f32 (bf16-valued) golden: widen and compare. */
static void cmp_codes(const char *what, const uint16_t *got,
                      const float *want, size_t n) {
    float *w = calloc(n ? n : 1, sizeof(float));
    for (size_t i = 0; i < n; i++) w[i] = apus_bf16_f32(got[i]);
    cmp_f32(what, w, want, n);
    dg_add(w, n * sizeof(float));
    free(w);
}

static void cmp_i32(const char *what, const int32_t *got,
                    const int32_t *want, size_t n) {
    /* selections are bitwise in BOTH tiers (margin-protected) */
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[i]) bad++;
    CHECK(bad == 0, "%s: %zu/%zu i32 selections differ", what, bad, n);
    dg_add(got, n * sizeof(int32_t));
}

static void dg_f32(const float *v, size_t n) {
    dg_add(v, n * sizeof(float));
}

/* --- exp probe (host-transcendental detection) ----------------------------*/

static void run_probe(const char *dir, const char *man) {
    size_t n = (size_t)man_long(man, "probe_n");
    float *x = load_f32(dir, "probe_x", n);
    float *y = load_f32(dir, "probe_y", n);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        float e = expf(x[i]);
        if (memcmp(&e, y + i, 4) != 0) bad++;
    }
    bitwise_mode = (bad == 0);
    printf("probe: expf vs numpy f32 exp — %zu/%zu differ -> %s tier\n",
           bad, n, bitwise_mode ? "BITWISE" : "TOLERANCE");
    free(x);
    free(y);
}

/* --- mHC ------------------------------------------------------------------*/

static void test_mhc(const char *dir, const char *man) {
    long nc = man_long(man, "nmhc");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64];
        snprintf(key, sizeof key, "mhc%ld_s", c);
        size_t s = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "mhc%ld_d", c);
        size_t d = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "mhc%ld_n", c);
        size_t n = (size_t)man_long(man, key);
        size_t nmix = (2 + n) * n, nx = n * d;

        snprintf(nm, sizeof nm, "mhc%ld_h", c);
        float *h = load_f32(dir, nm, s * nx);
        snprintf(nm, sizeof nm, "mhc%ld_fn", c);
        uint16_t *fn_c = load_u16(dir, nm, nmix * nx);
        snprintf(nm, sizeof nm, "mhc%ld_scale", c);
        float *scale = load_f32(dir, nm, 3);
        snprintf(nm, sizeof nm, "mhc%ld_base", c);
        float *base = load_f32(dir, nm, nmix);
        snprintf(nm, sizeof nm, "mhc%ld_mixes", c);
        float *g_mix = load_f32(dir, nm, s * nmix);
        snprintf(nm, sizeof nm, "mhc%ld_pre", c);
        float *g_pre = load_f32(dir, nm, s * n);
        snprintf(nm, sizeof nm, "mhc%ld_post", c);
        float *g_post = load_f32(dir, nm, s * n);
        snprintf(nm, sizeof nm, "mhc%ld_comb", c);
        float *g_comb = load_f32(dir, nm, s * n * n);
        snprintf(nm, sizeof nm, "mhc%ld_y", c);
        float *g_y = load_f32(dir, nm, s * d);

        float *fn = malloc(nmix * nx * sizeof(float));
        for (size_t i = 0; i < nmix * nx; i++)
            fn[i] = apus_bf16_f32(fn_c[i]);
        float *mixes = malloc(nmix * sizeof(float));
        float *pre = malloc(n * sizeof(float));
        float *post = malloc(n * sizeof(float));
        float *comb = malloc(n * n * sizeof(float));
        uint16_t *y = malloc(d * sizeof(uint16_t));
        char what[96];

        for (size_t t = 0; t < s; t++) {
            apus_gmhc_maps(h + t * nx, d, n, fn, scale, base,
                           NORM_EPS, HC_EPS, ITERS, pre, post, comb, mixes);
            snprintf(what, sizeof what, "mhc%ld[t%zu] mixes", c, t);
            cmp_f32(what, mixes, g_mix + t * nmix, nmix);
            dg_f32(mixes, nmix);
            snprintf(what, sizeof what, "mhc%ld[t%zu] pre", c, t);
            cmp_f32(what, pre, g_pre + t * n, n);
            dg_f32(pre, n);
            snprintf(what, sizeof what, "mhc%ld[t%zu] post", c, t);
            cmp_f32(what, post, g_post + t * n, n);
            dg_f32(post, n);
            snprintf(what, sizeof what, "mhc%ld[t%zu] comb", c, t);
            cmp_f32(what, comb, g_comb + t * n * n, n * n);
            dg_f32(comb, n * n);
            apus_gmhc_collapse(h + t * nx, pre, y, d, n);
            snprintf(what, sizeof what, "mhc%ld[t%zu] collapse", c, t);
            cmp_codes(what, y, g_y + t * d, d);
        }
        free(y);
        free(comb);
        free(post);
        free(pre);
        free(mixes);
        free(fn);
        free(g_y);
        free(g_comb);
        free(g_post);
        free(g_pre);
        free(g_mix);
        free(base);
        free(scale);
        free(fn_c);
        free(h);
    }
}

static void test_expand(const char *dir, const char *man) {
    long nc = man_long(man, "nexpand");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64];
        snprintf(key, sizeof key, "exp%ld_s", c);
        size_t s = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "exp%ld_d", c);
        size_t d = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "exp%ld_n", c);
        size_t n = (size_t)man_long(man, key);

        snprintf(nm, sizeof nm, "exp%ld_x", c);
        uint16_t *x = load_u16(dir, nm, s * d);
        snprintf(nm, sizeof nm, "exp%ld_res", c);
        uint16_t *res = load_u16(dir, nm, s * n * d);
        snprintf(nm, sizeof nm, "exp%ld_post", c);
        float *post = load_f32(dir, nm, s * n);
        snprintf(nm, sizeof nm, "exp%ld_comb", c);
        float *comb = load_f32(dir, nm, s * n * n);
        snprintf(nm, sizeof nm, "exp%ld_y4", c);
        float *g_y4 = load_f32(dir, nm, s * n * d);

        uint16_t *y4 = malloc(n * d * sizeof(uint16_t));
        char what[96];
        for (size_t t = 0; t < s; t++) {
            apus_gmhc_expand(x + t * d, res + t * n * d, post + t * n,
                             comb + t * n * n, y4, d, n);
            snprintf(what, sizeof what, "exp%ld[t%zu] expand", c, t);
            cmp_codes(what, y4, g_y4 + t * n * d, n * d);
        }
        free(y4);
        free(g_y4);
        free(comb);
        free(post);
        free(res);
        free(x);
    }
}

static void test_head(const char *dir, const char *man) {
    long nc = man_long(man, "nhead");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64];
        snprintf(key, sizeof key, "head%ld_s", c);
        size_t s = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "head%ld_d", c);
        size_t d = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "head%ld_n", c);
        size_t n = (size_t)man_long(man, key);

        snprintf(nm, sizeof nm, "head%ld_x4", c);
        uint16_t *x4 = load_u16(dir, nm, s * n * d);
        snprintf(nm, sizeof nm, "head%ld_y", c);
        float *g_y = load_f32(dir, nm, s * d);

        uint16_t *y = malloc(d * sizeof(uint16_t));
        char what[96];
        for (size_t t = 0; t < s; t++) {
            apus_gmhc_head(x4 + t * n * d, y, d, n);
            snprintf(what, sizeof what, "head%ld[t%zu] head", c, t);
            cmp_codes(what, y, g_y + t * d, d);
        }
        free(y);
        free(g_y);
        free(x4);
    }
}

/* --- MoE ------------------------------------------------------------------*/

static void test_router(const char *dir, const char *man) {
    long nc = man_long(man, "nrouter");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64];
        snprintf(key, sizeof key, "rtr%ld_s", c);
        size_t s = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "rtr%ld_E", c);
        int E = (int)man_long(man, key);
        snprintf(key, sizeof key, "rtr%ld_d", c);
        size_t d = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "rtr%ld_topk", c);
        int topk = (int)man_long(man, key);

        snprintf(nm, sizeof nm, "rtr%ld_x", c);
        uint16_t *x = load_u16(dir, nm, s * d);
        snprintf(nm, sizeof nm, "rtr%ld_w", c);
        uint16_t *gw = load_u16(dir, nm, (size_t)E * d);
        snprintf(nm, sizeof nm, "rtr%ld_bias", c);
        float *bias = load_f32(dir, nm, (size_t)E);
        snprintf(nm, sizeof nm, "rtr%ld_scores", c);
        float *g_sc = load_f32(dir, nm, s * (size_t)E);
        snprintf(nm, sizeof nm, "rtr%ld_biased", c);
        float *g_bi = load_f32(dir, nm, s * (size_t)E);
        snprintf(nm, sizeof nm, "rtr%ld_idx", c);
        int32_t *g_idx = load_i32(dir, nm, s * (size_t)topk);
        snprintf(nm, sizeof nm, "rtr%ld_wgt", c);
        float *g_w = load_f32(dir, nm, s * (size_t)topk);

        ApusGmoeRouterW rw = { E, topk, d, RSCALE, gw, bias };
        float *scores = malloc((size_t)E * sizeof(float));
        float *biased = malloc((size_t)E * sizeof(float));
        float *wgt = malloc((size_t)topk * sizeof(float));
        int32_t *idx = malloc((size_t)topk * sizeof(int32_t));
        char what[96];
        for (size_t t = 0; t < s; t++) {
            apus_gmoe_router(&rw, x + t * d, scores, idx, wgt, biased);
            snprintf(what, sizeof what, "rtr%ld[t%zu] scores", c, t);
            cmp_f32(what, scores, g_sc + t * (size_t)E, (size_t)E);
            dg_f32(scores, (size_t)E);
            snprintf(what, sizeof what, "rtr%ld[t%zu] biased", c, t);
            cmp_f32(what, biased, g_bi + t * (size_t)E, (size_t)E);
            dg_f32(biased, (size_t)E);
            snprintf(what, sizeof what, "rtr%ld[t%zu] idx", c, t);
            cmp_i32(what, idx, g_idx + t * (size_t)topk, (size_t)topk);
            snprintf(what, sizeof what, "rtr%ld[t%zu] wgt", c, t);
            cmp_f32(what, wgt, g_w + t * (size_t)topk, (size_t)topk);
            dg_f32(wgt, (size_t)topk);
        }
        free(idx);
        free(wgt);
        free(biased);
        free(scores);
        free(g_w);
        free(g_idx);
        free(g_bi);
        free(g_sc);
        free(bias);
        free(gw);
        free(x);
    }
}

/* dequant one fp8 weight to bf16 codes */
static uint16_t *deq(const char *dir, const char *base, size_t O, size_t K) {
    char nm[96];
    snprintf(nm, sizeof nm, "%s_codes", base);
    uint8_t *codes = load_u8(dir, nm, O * K);
    snprintf(nm, sizeof nm, "%s_scales", base);
    float *scales = load_f32(dir, nm,
                             apus_fp8blk_nblocks(O) * apus_fp8blk_nblocks(K));
    uint16_t *w = malloc(O * K * sizeof(uint16_t));
    apus_fp8blk_dequant(codes, scales, w, O, K);
    free(codes);
    free(scales);
    return w;
}

static void test_expert(const char *dir, const char *man) {
    long nc = man_long(man, "nexpert");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64], base[64];
        snprintf(key, sizeof key, "xp%ld_s", c);
        size_t s = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "xp%ld_d", c);
        size_t d = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "xp%ld_inter", c);
        size_t inter = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "xp%ld_limit", c);
        float limit = (float)strtod(man_find(man, key), NULL);

        snprintf(nm, sizeof nm, "xp%ld_x", c);
        uint16_t *x = load_u16(dir, nm, s * d);
        snprintf(base, sizeof base, "xp%ld_g", c);
        uint16_t *wg = deq(dir, base, inter, d);
        snprintf(base, sizeof base, "xp%ld_u", c);
        uint16_t *wu = deq(dir, base, inter, d);
        snprintf(base, sizeof base, "xp%ld_d", c);
        uint16_t *wd = deq(dir, base, d, inter);
        snprintf(nm, sizeof nm, "xp%ld_go", c);
        float *g_go = load_f32(dir, nm, s * inter);
        snprintf(nm, sizeof nm, "xp%ld_uo", c);
        float *g_uo = load_f32(dir, nm, s * inter);
        snprintf(nm, sizeof nm, "xp%ld_ho", c);
        float *g_ho = load_f32(dir, nm, s * inter);
        snprintf(nm, sizeof nm, "xp%ld_out", c);
        float *g_out = load_f32(dir, nm, s * d);

        uint16_t *g = malloc(inter * sizeof(uint16_t));
        uint16_t *u = malloc(inter * sizeof(uint16_t));
        uint16_t *h = malloc(inter * sizeof(uint16_t));
        uint16_t *out = malloc(d * sizeof(uint16_t));
        size_t maxk = d > inter ? d : inter;
        float *xf = malloc(maxk * sizeof(float));
        char what[96];
        for (size_t t = 0; t < s; t++) {
            apus_gmoe_expert(wg, wu, wd, x + t * d, d, inter, limit,
                             out, g, u, h, xf);
            snprintf(what, sizeof what, "xp%ld[t%zu] gate", c, t);
            cmp_codes(what, g, g_go + t * inter, inter);
            snprintf(what, sizeof what, "xp%ld[t%zu] up", c, t);
            cmp_codes(what, u, g_uo + t * inter, inter);
            snprintf(what, sizeof what, "xp%ld[t%zu] h", c, t);
            cmp_codes(what, h, g_ho + t * inter, inter);
            snprintf(what, sizeof what, "xp%ld[t%zu] out", c, t);
            cmp_codes(what, out, g_out + t * d, d);
        }
        free(xf);
        free(out);
        free(h);
        free(u);
        free(g);
        free(g_out);
        free(g_ho);
        free(g_uo);
        free(g_go);
        free(wd);
        free(wu);
        free(wg);
        free(x);
    }
}

static void test_moe(const char *dir, const char *man) {
    long nc = man_long(man, "nmoe");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64], base[64];
        snprintf(key, sizeof key, "moe%ld_s", c);
        size_t s = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "moe%ld_E", c);
        int E = (int)man_long(man, key);
        snprintf(key, sizeof key, "moe%ld_d", c);
        size_t d = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "moe%ld_topk", c);
        int topk = (int)man_long(man, key);
        snprintf(key, sizeof key, "moe%ld_inter", c);
        size_t inter = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "moe%ld_sinter", c);
        size_t sinter = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "moe%ld_limit", c);
        float limit = (float)strtod(man_find(man, key), NULL);

        snprintf(nm, sizeof nm, "moe%ld_x", c);
        uint16_t *x = load_u16(dir, nm, s * d);
        snprintf(nm, sizeof nm, "moe%ld_gate_w", c);
        uint16_t *gw = load_u16(dir, nm, (size_t)E * d);
        snprintf(nm, sizeof nm, "moe%ld_gate_bias", c);
        float *bias = load_f32(dir, nm, (size_t)E);

        uint16_t **eg = malloc((size_t)E * sizeof(uint16_t *));
        uint16_t **eu = malloc((size_t)E * sizeof(uint16_t *));
        uint16_t **ed = malloc((size_t)E * sizeof(uint16_t *));
        for (int e = 0; e < E; e++) {
            snprintf(base, sizeof base, "moe%ld_e%d_g", c, e);
            eg[e] = deq(dir, base, inter, d);
            snprintf(base, sizeof base, "moe%ld_e%d_u", c, e);
            eu[e] = deq(dir, base, inter, d);
            snprintf(base, sizeof base, "moe%ld_e%d_d", c, e);
            ed[e] = deq(dir, base, d, inter);
        }
        snprintf(base, sizeof base, "moe%ld_sg", c);
        uint16_t *sg = deq(dir, base, sinter, d);
        snprintf(base, sizeof base, "moe%ld_su", c);
        uint16_t *su = deq(dir, base, sinter, d);
        snprintf(base, sizeof base, "moe%ld_sd", c);
        uint16_t *sd = deq(dir, base, d, sinter);

        snprintf(nm, sizeof nm, "moe%ld_scores", c);
        float *g_sc = load_f32(dir, nm, s * (size_t)E);
        snprintf(nm, sizeof nm, "moe%ld_biased", c);
        float *g_bi = load_f32(dir, nm, s * (size_t)E);
        snprintf(nm, sizeof nm, "moe%ld_idx", c);
        int32_t *g_idx = load_i32(dir, nm, s * (size_t)topk);
        snprintf(nm, sizeof nm, "moe%ld_wgt", c);
        float *g_w = load_f32(dir, nm, s * (size_t)topk);
        snprintf(nm, sizeof nm, "moe%ld_routed", c);
        float *g_ro = load_f32(dir, nm, s * d);
        snprintf(nm, sizeof nm, "moe%ld_shared", c);
        float *g_sh = load_f32(dir, nm, s * d);
        snprintf(nm, sizeof nm, "moe%ld_out", c);
        float *g_out = load_f32(dir, nm, s * d);

        ApusGmoeW W;
        W.router = (ApusGmoeRouterW){ E, topk, d, RSCALE, gw, bias };
        W.inter = inter;
        W.eg = (const uint16_t *const *)eg;
        W.eu = (const uint16_t *const *)eu;
        W.ed = (const uint16_t *const *)ed;
        W.sg = sg;
        W.su = su;
        W.sd = sd;
        W.sinter = sinter;
        W.limit = limit;

        ApusGmoeInterm interm;
        interm.router_scores = malloc(s * (size_t)E * sizeof(float));
        interm.router_biased = malloc(s * (size_t)E * sizeof(float));
        interm.router_w = malloc(s * (size_t)topk * sizeof(float));
        interm.router_idx = malloc(s * (size_t)topk * sizeof(int32_t));
        interm.moe_routed = malloc(s * d * sizeof(float));
        interm.moe_shared = malloc(s * d * sizeof(float));
        uint16_t *out = malloc(s * d * sizeof(uint16_t));

        apus_gmoe_forward(&W, x, s, out, &interm);

        char what[96];
        snprintf(what, sizeof what, "moe%ld scores", c);
        cmp_f32(what, interm.router_scores, g_sc, s * (size_t)E);
        dg_f32(interm.router_scores, s * (size_t)E);
        snprintf(what, sizeof what, "moe%ld biased", c);
        cmp_f32(what, interm.router_biased, g_bi, s * (size_t)E);
        dg_f32(interm.router_biased, s * (size_t)E);
        snprintf(what, sizeof what, "moe%ld idx", c);
        cmp_i32(what, interm.router_idx, g_idx, s * (size_t)topk);
        snprintf(what, sizeof what, "moe%ld wgt", c);
        cmp_f32(what, interm.router_w, g_w, s * (size_t)topk);
        dg_f32(interm.router_w, s * (size_t)topk);
        snprintf(what, sizeof what, "moe%ld routed", c);
        cmp_f32(what, interm.moe_routed, g_ro, s * d);
        dg_f32(interm.moe_routed, s * d);
        snprintf(what, sizeof what, "moe%ld shared", c);
        cmp_f32(what, interm.moe_shared, g_sh, s * d);
        dg_f32(interm.moe_shared, s * d);
        snprintf(what, sizeof what, "moe%ld out", c);
        cmp_codes(what, out, g_out, s * d);

        free(out);
        free(interm.moe_shared);
        free(interm.moe_routed);
        free(interm.router_idx);
        free(interm.router_w);
        free(interm.router_biased);
        free(interm.router_scores);
        free(g_out);
        free(g_sh);
        free(g_ro);
        free(g_w);
        free(g_idx);
        free(g_bi);
        free(g_sc);
        free(sd);
        free(su);
        free(sg);
        for (int e = 0; e < E; e++) {
            free(ed[e]);
            free(eu[e]);
            free(eg[e]);
        }
        free(ed);
        free(eu);
        free(eg);
        free(bias);
        free(gw);
        free(x);
    }
}

int main(void) {
    const char *dir = "tests/m4g/golden";
    long mlen;
    char mpath[512];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    char *man = (char *)read_file(mpath, &mlen);

    run_probe(dir, man);
    /* Per-suite progress + fflush: stdout is block-buffered under CI
     * pipes/redirects; without the flush a hang prints nothing (M15y). */
    test_mhc(dir, man);
    printf("m4g: mhc done\n");    fflush(stdout);
    test_expand(dir, man);
    printf("m4g: expand done\n"); fflush(stdout);
    test_head(dir, man);
    printf("m4g: head done\n");   fflush(stdout);
    test_router(dir, man);
    printf("m4g: router done\n"); fflush(stdout);
    test_expert(dir, man);
    printf("m4g: expert done\n"); fflush(stdout);
    test_moe(dir, man);
    printf("m4g: moe done\n");    fflush(stdout);

    printf("digest: %016llx\n", (unsigned long long)dg_state);
    printf("%s: %d checks, %d failures\n",
           failures ? "FAIL" : "PASS", checks, failures);
    free(man);
    return failures ? 1 : 0;
}
