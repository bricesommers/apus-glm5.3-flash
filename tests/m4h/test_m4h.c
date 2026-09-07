/*
 * tests/m4h/test_m4h.c — M4b (GLM) sublayer hard gate: c/gkda.h (KDA
 * chunked-prefill AND recurrent-decode orderings, conv, gates, o_norm)
 * and c/gdsa.h (MLA pure-NoPE + Lightning indexer + sparse attention)
 * against the numpy-oracle goldens (tests/m4h/gen_golden.py — the M0
 * oracle's own sublayer functions in f32-faithful mode).
 *
 * Gate tiers (tests/m4h/README.md):
 *   BITWISE   — host expf == numpy float32 exp (probed at runtime):
 *               every golden compared memcmp-bitwise (f32 goldens; BF16
 *               outputs widened — equal iff the codes are equal; indexer
 *               top-k selections; states).
 *   TOLERANCE — otherwise: f32 goldens rel <= 1e-5 (or abs <= 1e-6; the
 *               host-transcendental class), BF16-code goldens rel <= 1e-2
 *               or abs <= 1e-6 (the +-1-code bf16 boundary-flip class),
 *               indexer pool scores rel <= 1e-5 or abs <= 5e-3 (the
 *               pool-prob bf16-flip class), indexer selections STILL
 *               bitwise EXCEPT the manifest's fragile queries (structural
 *               relu-clip zero ties + sub-margin gaps — skipped).
 *
 * Run from the repository root (golden fixtures under tests/m4h/golden/).
 * Prints FNV-1a digests of the C outputs; the Makefile diffs full output
 * across APUS_THREADS=1/4/8 (thread-count independence).
 */
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_GKDA_IMPLEMENTATION
#define APUS_GDSA_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#include "gmhc.h"
#include "gmoe.h"
#include "gkda.h"
#include "gdsa.h"
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
static double man_double(const char *man, const char *key) {
    return strtod(man_find(man, key), NULL);
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
        printf("%s: size mismatch (%ld != %zu)\n", path, len, n);
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

#define TOL_F32   1e-5f   /* f32 host-transcendental class */
#define TOL_ABS   1e-6f   /* near-zero absolute clause */
#define TOL_CODE  1e-2f   /* bf16 +-1-code boundary-flip class (relative) */
#define TOL_SCORE 5e-3f   /* indexer pool-score abs clause (pool-prob
                             bf16-flip class) */

/* f32 golden comparison (rel <= TOL_F32 or abs <= TOL_ABS in the
 * tolerance tier). */
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
    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        double rel = d / fmax(1e-30, fabs((double)want[i]));
        if (rel > worst) worst = rel;
        if (rel > TOL_F32 && d > TOL_ABS) bad++;
    }
    CHECK(bad == 0, "%s: %zu/%zu over tol (max rel %.3g)", what, bad, n,
          worst);
}

/* Indexer pool scores: rel <= TOL_F32 or abs <= TOL_SCORE (the pool-prob
 * bf16-flip class). Rows listed in skip[] are excluded in the tolerance
 * tier (fragile queries). */
static void cmp_scores(const char *what, const float *got,
                       const float *want, size_t s, size_t nfull,
                       const unsigned char *skip) {
    if (bitwise_mode) {
        cmp_f32(what, got, want, s * nfull);
        return;
    }
    size_t bad = 0;
    double worst = 0.0;
    for (size_t t = 0; t < s; t++) {
        if (skip && skip[t]) continue;
        for (size_t i = 0; i < nfull; i++) {
            size_t k = t * nfull + i;
            double d = fabs((double)got[k] - (double)want[k]);
            double rel = d / fmax(1e-30, fabs((double)want[k]));
            if (rel > worst) worst = rel;
            if (rel > TOL_F32 && d > TOL_SCORE) bad++;
        }
    }
    CHECK(bad == 0, "%s: %zu over tol (max rel %.3g)", what, bad, worst);
}

/* BF16-code output vs an f32 (bf16-valued) golden: bitwise -> codes
 * equal; tolerance -> widened rel <= TOL_CODE or abs <= TOL_ABS. Rows may
 * be skipped in the tolerance tier (fragile queries); row size rc. */
static void cmp_codes_rows(const char *what, const uint16_t *got,
                           const float *want, size_t rows, size_t rc,
                           const unsigned char *skip) {
    size_t n = rows * rc;
    float *w = calloc(n ? n : 1, sizeof(float));
    for (size_t i = 0; i < n; i++) w[i] = apus_bf16_f32(got[i]);
    if (bitwise_mode) {
        if (memcmp(w, want, n * sizeof(float)) != 0) {
            size_t bad = 0;
            for (size_t i = 0; i < n; i++)
                if (memcmp(w + i, want + i, 4) != 0) bad++;
            CHECK(0, "%s: %zu/%zu bf16 codes differ (bitwise)", what,
                  bad, n);
        } else {
            CHECK(1, "%s", what);
        }
    } else {
        size_t bad = 0;
        double worst = 0.0;
        for (size_t t = 0; t < rows; t++) {
            if (skip && skip[t]) continue;
            for (size_t i = 0; i < rc; i++) {
                size_t k = t * rc + i;
                double d = fabs((double)w[k] - (double)want[k]);
                double rel = d / fmax(1e-30, fabs((double)want[k]));
                if (rel > worst) worst = rel;
                if (rel > TOL_CODE && d > TOL_ABS) bad++;
            }
        }
        CHECK(bad == 0, "%s: %zu over tol (max rel %.3g)", what, bad,
              worst);
    }
    dg_add(w, n * sizeof(float));
    free(w);
}

static void cmp_codes(const char *what, const uint16_t *got,
                      const float *want, size_t n) {
    cmp_codes_rows(what, got, want, 1, n, NULL);
}

/* Indexer top-k selections: ALWAYS exact, except fragile queries in the
 * tolerance tier (skipped). */
static void cmp_topk(const char *what, const int32_t *got,
                     const int32_t *want, size_t s, size_t width,
                     const unsigned char *skip) {
    size_t bad = 0, skipped = 0;
    for (size_t t = 0; t < s; t++) {
        if (!bitwise_mode && skip && skip[t]) {
            skipped++;
            continue;
        }
        for (size_t i = 0; i < width; i++)
            if (got[t * width + i] != want[t * width + i]) bad++;
    }
    CHECK(bad == 0, "%s: %zu mismatched indices (%zu rows skipped)", what,
          bad, skipped);
    dg_add(got, s * width * sizeof(int32_t));
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

/* --- dims ------------------------------------------------------------------*/

static size_t DIM, H, D, DR, QL, KL, IH, ID, ITOPK, KPOOL, WIDTH;
static float EPS, LB;

static void load_dims(const char *man) {
    DIM = (size_t)man_long(man, "dim_dim");
    H = (size_t)man_long(man, "dim_H");
    D = (size_t)man_long(man, "dim_D");
    DR = (size_t)man_long(man, "dim_Dr");
    QL = (size_t)man_long(man, "dim_ql");
    KL = (size_t)man_long(man, "dim_kl");
    IH = (size_t)man_long(man, "dim_IH");
    ID = (size_t)man_long(man, "dim_ID");
    ITOPK = (size_t)man_long(man, "dim_itopk");
    KPOOL = (size_t)man_long(man, "dim_kpool");
    WIDTH = (size_t)man_long(man, "dim_width");
    EPS = (float)man_double(man, "dim_eps");
    LB = (float)man_double(man, "dim_lb");
}

/* --- KDA ------------------------------------------------------------------*/

static ApusGkdaW kdaW;

static void load_kda_weights(const char *dir) {
    size_t qkv = H * D;
    kdaW.dim = DIM;
    kdaW.H = H;
    kdaW.D = D;
    kdaW.Dr = DR;
    kdaW.q_w = load_u16(dir, "kda_q_w", qkv * DIM);
    kdaW.k_w = load_u16(dir, "kda_k_w", qkv * DIM);
    kdaW.v_w = load_u16(dir, "kda_v_w", qkv * DIM);
    kdaW.conv_w = load_u16(dir, "kda_conv_w", 3 * qkv * APUS_GKDA_CONV_K);
    kdaW.f_a = load_u16(dir, "kda_f_a", DR * DIM);
    kdaW.f_b = load_u16(dir, "kda_f_b", qkv * DR);
    kdaW.A_log = load_f32(dir, "kda_A_log", H);
    kdaW.dt_bias = load_f32(dir, "kda_dt_bias", qkv);
    kdaW.b_w = load_u16(dir, "kda_b_w", H * DIM);
    kdaW.g_a = load_u16(dir, "kda_g_a", DR * DIM);
    kdaW.g_b = load_u16(dir, "kda_g_b", qkv * DR);
    kdaW.o_norm = load_u16(dir, "kda_o_norm", D);
    kdaW.o_w = load_u16(dir, "kda_o_w", DIM * qkv);
    kdaW.lower_bound = LB;
    kdaW.eps = EPS;
}

static void test_kda(const char *dir, const char *man) {
    size_t qkv = H * D;
    long nc = man_long(man, "nkda");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64], what[128];
        snprintf(key, sizeof key, "k%ld_calls", c);
        size_t calls = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "k%ld_npre", c);
        size_t npre = (size_t)man_long(man, key);
        printf("kda case k%ld: %zu calls (%zu prefill)\n", c, calls, npre);
        fflush(stdout);   /* CI progress: stdout is block-buffered under
                             pipes/redirects — flush so a hang bisects to
                             the case (M15y, the windows-latest m4h hang) */

        ApusGkdaState st;
        st.conv_state = calloc(3 * qkv * (APUS_GKDA_CONV_K - 1),
                               sizeof(uint16_t));
        st.rec_state = calloc(H * D * D, sizeof(float));

        for (size_t ci = 0; ci < calls; ci++) {
            snprintf(key, sizeof key, "k%ld_c%zu_s", c, ci);
            size_t s = (size_t)man_long(man, key);
            int decode = ci >= npre;
            snprintf(nm, sizeof nm, "k%ld_c%zu_x", c, ci);
            uint16_t *x = load_u16(dir, nm, s * DIM);
            snprintf(nm, sizeof nm, "k%ld_c%zu_mixed", c, ci);
            float *g_mixed = load_f32(dir, nm, s * 3 * qkv);
            snprintf(nm, sizeof nm, "k%ld_c%zu_g", c, ci);
            float *g_g = load_f32(dir, nm, s * qkv);
            snprintf(nm, sizeof nm, "k%ld_c%zu_beta", c, ci);
            float *g_beta = load_f32(dir, nm, s * H);
            snprintf(nm, sizeof nm, "k%ld_c%zu_core", c, ci);
            float *g_core = load_f32(dir, nm, s * qkv);
            snprintf(nm, sizeof nm, "k%ld_c%zu_gate", c, ci);
            float *g_gate = load_f32(dir, nm, s * qkv);
            snprintf(nm, sizeof nm, "k%ld_c%zu_onorm", c, ci);
            float *g_onorm = load_f32(dir, nm, s * qkv);
            snprintf(nm, sizeof nm, "k%ld_c%zu_out", c, ci);
            float *g_out = load_f32(dir, nm, s * DIM);

            ApusGkdaInterm im;
            im.mixed = malloc(s * 3 * qkv * sizeof(uint16_t));
            im.g = malloc(s * qkv * sizeof(float));
            im.beta = malloc(s * H * sizeof(uint16_t));
            im.core = malloc(s * qkv * sizeof(uint16_t));
            im.gate = malloc(s * qkv * sizeof(uint16_t));
            im.onorm = malloc(s * qkv * sizeof(uint16_t));
            uint16_t *out = malloc(s * DIM * sizeof(uint16_t));

            apus_gkda_forward(&kdaW, x, s, &st, decode, out, &im);

            snprintf(what, sizeof what, "k%ld.c%zu%s mixed", c, ci,
                     decode ? " (dec)" : "");
            cmp_codes(what, im.mixed, g_mixed, s * 3 * qkv);
            snprintf(what, sizeof what, "k%ld.c%zu g", c, ci);
            cmp_f32(what, im.g, g_g, s * qkv);
            dg_f32(im.g, s * qkv);
            snprintf(what, sizeof what, "k%ld.c%zu beta", c, ci);
            cmp_codes(what, im.beta, g_beta, s * H);
            snprintf(what, sizeof what, "k%ld.c%zu core", c, ci);
            cmp_codes(what, im.core, g_core, s * qkv);
            snprintf(what, sizeof what, "k%ld.c%zu gate", c, ci);
            cmp_codes(what, im.gate, g_gate, s * qkv);
            snprintf(what, sizeof what, "k%ld.c%zu onorm", c, ci);
            cmp_codes(what, im.onorm, g_onorm, s * qkv);
            snprintf(what, sizeof what, "k%ld.c%zu out", c, ci);
            cmp_codes(what, out, g_out, s * DIM);

            free(out);
            free(im.onorm);
            free(im.gate);
            free(im.core);
            free(im.beta);
            free(im.g);
            free(im.mixed);
            free(g_out);
            free(g_onorm);
            free(g_gate);
            free(g_core);
            free(g_beta);
            free(g_g);
            free(g_mixed);
            free(x);
        }
        /* states: conv_state (engine channel-major [3qkv][K-1] vs the
         * oracle's [K-1][3qkv]) and rec_state [H,D,D] fp32 */
        snprintf(nm, sizeof nm, "k%ld_conv_state", c);
        float *g_conv = load_f32(dir, nm, 3 * 3 * qkv);
        float *got_conv = malloc(3 * 3 * qkv * sizeof(float));
        for (size_t ch = 0; ch < 3 * qkv; ch++)
            for (size_t j = 0; j < APUS_GKDA_CONV_K - 1; j++)
                got_conv[j * 3 * qkv + ch] =
                    apus_bf16_f32(st.conv_state[
                        ch * (APUS_GKDA_CONV_K - 1) + j]);
        snprintf(what, sizeof what, "k%ld conv_state", c);
        cmp_f32(what, got_conv, g_conv, 3 * 3 * qkv);
        dg_f32(got_conv, 3 * 3 * qkv);
        snprintf(nm, sizeof nm, "k%ld_rec_state", c);
        float *g_rec = load_f32(dir, nm, H * D * D);
        snprintf(what, sizeof what, "k%ld rec_state", c);
        cmp_f32(what, st.rec_state, g_rec, H * D * D);
        dg_f32(st.rec_state, H * D * D);
        free(g_rec);
        free(got_conv);
        free(g_conv);
        free(st.rec_state);
        free(st.conv_state);
    }
}

/* --- DSA ------------------------------------------------------------------*/

static ApusGdsaW dsaW;

static void load_dsa_weights(const char *dir) {
    size_t qd = D, vd = D;
    dsaW.dim = DIM;
    dsaW.H = H;
    dsaW.qd = qd;
    dsaW.vd = vd;
    dsaW.ql = QL;
    dsaW.kl = KL;
    dsaW.q_a_c = load_u8(dir, "dsa_q_a_codes", QL * DIM);
    dsaW.q_a_s = load_f32(dir, "dsa_q_a_scales",
                          apus_fp8blk_nblocks(QL) * apus_fp8blk_nblocks(DIM));
    dsaW.q_a_norm = load_u16(dir, "dsa_q_a_norm", QL);
    dsaW.q_b_c = load_u8(dir, "dsa_q_b_codes", H * qd * QL);
    dsaW.q_b_s = load_f32(dir, "dsa_q_b_scales",
                          apus_fp8blk_nblocks(H * qd)
                          * apus_fp8blk_nblocks(QL));
    dsaW.kv_a_c = load_u8(dir, "dsa_kv_a_codes", KL * DIM);
    dsaW.kv_a_s = load_f32(dir, "dsa_kv_a_scales",
                           apus_fp8blk_nblocks(KL)
                           * apus_fp8blk_nblocks(DIM));
    dsaW.kv_a_norm = load_u16(dir, "dsa_kv_a_norm", KL);
    dsaW.kv_b = load_u16(dir, "dsa_kv_b", H * (qd + vd) * KL);
    dsaW.o_c = load_u8(dir, "dsa_o_codes", DIM * H * vd);
    dsaW.o_s = load_f32(dir, "dsa_o_scales",
                        apus_fp8blk_nblocks(DIM)
                        * apus_fp8blk_nblocks(H * vd));
    dsaW.eps = EPS;
    dsaW.IH = IH;
    dsaW.ID = ID;
    dsaW.topk = ITOPK;
    dsaW.kpool = KPOOL;
    dsaW.tail = 1;
    dsaW.idx_wq_b = load_u16(dir, "dsa_idx_wq_b", IH * ID * QL);
    dsaW.idx_wk = load_u16(dir, "dsa_idx_wk", ID * DIM);
    dsaW.idx_knorm_w = load_u16(dir, "dsa_idx_knorm_w", ID);
    dsaW.idx_knorm_b = load_u16(dir, "dsa_idx_knorm_b", ID);
    dsaW.idx_wproj = load_u16(dir, "dsa_idx_wproj", IH * DIM);
    dsaW.idx_ape = load_u16(dir, "dsa_idx_ape", KPOOL * ID);
    dsaW.idx_gate = load_u16(dir, "dsa_idx_gate", ID * DIM);
}

/* parse the fragile list "9,11,..." or "-" into a bool array [s].
 * '\r' terminates too: a CRLF manifest (a text-mode Python write on
 * Windows) otherwise leaves p on '\r' after the last number, strtol
 * performs no conversion, p never advances — an infinite loop (the
 * 2026-09-06 windows-latest CI hang). The generators also pin LF. */
static unsigned char *parse_fragile(const char *man, const char *key,
                                    size_t s) {
    unsigned char *fr = calloc(s ? s : 1, 1);
    const char *p = man_find(man, key);
    if (*p == '-') return fr;
    while (*p && *p != '\n' && *p != '\r') {
        long t = strtol(p, (char **)&p, 10);
        if (t >= 0 && (size_t)t < s) fr[t] = 1;
        if (*p == ',') p++;
    }
    return fr;
}

static void test_dsa(const char *dir, const char *man) {
    size_t qd = D, vd = D;
    long nc = man_long(man, "ndsa");
    for (long c = 0; c < nc; c++) {
        char key[64], nm[64], what[128];
        snprintf(key, sizeof key, "d%ld_calls", c);
        size_t calls = (size_t)man_long(man, key);
        snprintf(key, sizeof key, "d%ld_n", c);
        size_t cap = (size_t)man_long(man, key);
        printf("dsa case d%ld: %zu calls (cap %zu)\n", c, calls, cap);
        fflush(stdout);   /* CI progress (see the kda-case note above) */

        ApusGdsaState st;
        st.k_cache = calloc(H * cap * qd, sizeof(uint16_t));
        st.v_cache = calloc(H * cap * vd, sizeof(uint16_t));
        st.idx_k = calloc(cap * ID, sizeof(uint16_t));
        st.idx_gate = calloc(cap * ID, sizeof(uint16_t));
        st.cap = cap;
        st.n = 0;

        for (size_t ci = 0; ci < calls; ci++) {
            snprintf(key, sizeof key, "d%ld_c%zu_s", c, ci);
            size_t s = (size_t)man_long(man, key);
            snprintf(key, sizeof key, "d%ld_c%zu_fragile", c, ci);
            unsigned char *fragile = parse_fragile(man, key, s);
            size_t base = st.n;
            size_t n = base + s;
            size_t n_full = n / KPOOL;

            snprintf(nm, sizeof nm, "d%ld_c%zu_x", c, ci);
            uint16_t *x = load_u16(dir, nm, s * DIM);
            snprintf(nm, sizeof nm, "d%ld_c%zu_q_resid", c, ci);
            float *g_qres = load_f32(dir, nm, s * QL);
            snprintf(nm, sizeof nm, "d%ld_c%zu_q", c, ci);
            float *g_q = load_f32(dir, nm, s * H * qd);
            snprintf(nm, sizeof nm, "d%ld_c%zu_k_new", c, ci);
            float *g_kn = load_f32(dir, nm, s * H * qd);
            snprintf(nm, sizeof nm, "d%ld_c%zu_v_new", c, ci);
            float *g_vn = load_f32(dir, nm, s * H * vd);
            snprintf(nm, sizeof nm, "d%ld_c%zu_idx_knew", c, ci);
            float *g_ikn = load_f32(dir, nm, s * ID);
            snprintf(nm, sizeof nm, "d%ld_c%zu_idx_gnew", c, ci);
            float *g_ign = load_f32(dir, nm, s * ID);
            snprintf(nm, sizeof nm, "d%ld_c%zu_idx_scores", c, ci);
            float *g_sc = n_full ? load_f32(dir, nm, s * n_full) : NULL;
            snprintf(nm, sizeof nm, "d%ld_c%zu_idx_topk", c, ci);
            int32_t *g_tk = load_i32(dir, nm, s * WIDTH);
            snprintf(nm, sizeof nm, "d%ld_c%zu_probs", c, ci);
            float *g_pr = load_f32(dir, nm, s * H * n);
            snprintf(nm, sizeof nm, "d%ld_c%zu_attn", c, ci);
            float *g_at = load_f32(dir, nm, s * H * vd);
            snprintf(nm, sizeof nm, "d%ld_c%zu_out", c, ci);
            float *g_out = load_f32(dir, nm, s * DIM);

            ApusGdsaInterm im;
            im.q_resid = malloc(s * QL * sizeof(uint16_t));
            im.q = malloc(s * H * qd * sizeof(uint16_t));
            im.k_new = malloc(s * H * qd * sizeof(uint16_t));
            im.v_new = malloc(s * H * vd * sizeof(uint16_t));
            im.idx_knew = malloc(s * ID * sizeof(uint16_t));
            im.idx_gnew = malloc(s * ID * sizeof(uint16_t));
            im.idx_scores = malloc(s * (n_full ? n_full : 1)
                                   * sizeof(float));
            im.idx_topk = malloc(s * WIDTH * sizeof(int32_t));
            im.probs = malloc(s * H * n * sizeof(uint16_t));
            im.attn = malloc(s * H * vd * sizeof(uint16_t));
            uint16_t *out = malloc(s * DIM * sizeof(uint16_t));

            apus_gdsa_forward(&dsaW, x, s, &st, out, &im);

            snprintf(what, sizeof what, "d%ld.c%zu q_resid", c, ci);
            cmp_codes(what, im.q_resid, g_qres, s * QL);
            snprintf(what, sizeof what, "d%ld.c%zu q", c, ci);
            cmp_codes(what, im.q, g_q, s * H * qd);
            snprintf(what, sizeof what, "d%ld.c%zu k_new", c, ci);
            cmp_codes(what, im.k_new, g_kn, s * H * qd);
            snprintf(what, sizeof what, "d%ld.c%zu v_new", c, ci);
            cmp_codes(what, im.v_new, g_vn, s * H * vd);
            snprintf(what, sizeof what, "d%ld.c%zu idx_knew", c, ci);
            cmp_codes(what, im.idx_knew, g_ikn, s * ID);
            snprintf(what, sizeof what, "d%ld.c%zu idx_gnew", c, ci);
            cmp_codes(what, im.idx_gnew, g_ign, s * ID);
            snprintf(what, sizeof what, "d%ld.c%zu idx_scores", c, ci);
            if (n_full) {
                cmp_scores(what, im.idx_scores, g_sc, s, n_full, fragile);
                dg_f32(im.idx_scores, s * n_full);
            }
            snprintf(what, sizeof what, "d%ld.c%zu idx_topk", c, ci);
            cmp_topk(what, im.idx_topk, g_tk, s, WIDTH, fragile);
            snprintf(what, sizeof what, "d%ld.c%zu probs", c, ci);
            const size_t fh = s * (size_t)H;
            unsigned char *fragile_h = calloc(fh ? fh : 1, 1);
            for (size_t t = 0; t < s; t++)
                if (fragile[t])
                    for (size_t h = 0; h < H; h++)
                        fragile_h[t * H + h] = 1;
            cmp_codes_rows(what, im.probs, g_pr, s * H, n, fragile_h);
            free(fragile_h);
            snprintf(what, sizeof what, "d%ld.c%zu attn", c, ci);
            cmp_codes_rows(what, im.attn, g_at, s, H * vd, fragile);
            snprintf(what, sizeof what, "d%ld.c%zu out", c, ci);
            cmp_codes_rows(what, out, g_out, s, DIM, fragile);

            free(out);
            free(im.attn);
            free(im.probs);
            free(im.idx_topk);
            free(im.idx_scores);
            free(im.idx_gnew);
            free(im.idx_knew);
            free(im.v_new);
            free(im.k_new);
            free(im.q);
            free(im.q_resid);
            free(g_out);
            free(g_at);
            free(g_pr);
            free(g_tk);
            free(g_sc);
            free(g_ign);
            free(g_ikn);
            free(g_vn);
            free(g_kn);
            free(g_q);
            free(g_qres);
            free(x);
            free(fragile);
        }
        /* final caches: engine head-major [H][cap*d] vs the oracle's
         * [n, H, d] token-major; idx caches token-major [n, ID] */
        snprintf(nm, sizeof nm, "d%ld_k_cache", c);
        float *g_kc = load_f32(dir, nm, cap * H * qd);
        snprintf(nm, sizeof nm, "d%ld_v_cache", c);
        float *g_vc = load_f32(dir, nm, cap * H * vd);
        float *got = malloc(cap * H * (qd > vd ? qd : vd) * sizeof(float));
        for (size_t p = 0; p < cap; p++)
            for (size_t h = 0; h < H; h++)
                for (size_t d = 0; d < qd; d++)
                    got[(p * H + h) * qd + d] =
                        apus_bf16_f32(st.k_cache[h * cap * qd + p * qd + d]);
        snprintf(what, sizeof what, "d%ld k_cache", c);
        cmp_f32(what, got, g_kc, cap * H * qd);
        dg_f32(got, cap * H * qd);
        for (size_t p = 0; p < cap; p++)
            for (size_t h = 0; h < H; h++)
                for (size_t d = 0; d < vd; d++)
                    got[(p * H + h) * vd + d] =
                        apus_bf16_f32(st.v_cache[h * cap * vd + p * vd + d]);
        snprintf(what, sizeof what, "d%ld v_cache", c);
        cmp_f32(what, got, g_vc, cap * H * vd);
        dg_f32(got, cap * H * vd);
        snprintf(nm, sizeof nm, "d%ld_idx_k", c);
        float *g_ik = load_f32(dir, nm, cap * ID);
        for (size_t i = 0; i < cap * ID; i++)
            got[i] = apus_bf16_f32(st.idx_k[i]);
        snprintf(what, sizeof what, "d%ld idx_k cache", c);
        cmp_f32(what, got, g_ik, cap * ID);
        dg_f32(got, cap * ID);
        snprintf(nm, sizeof nm, "d%ld_idx_gate", c);
        float *g_ig = load_f32(dir, nm, cap * ID);
        for (size_t i = 0; i < cap * ID; i++)
            got[i] = apus_bf16_f32(st.idx_gate[i]);
        snprintf(what, sizeof what, "d%ld idx_gate cache", c);
        cmp_f32(what, got, g_ig, cap * ID);
        dg_f32(got, cap * ID);
        free(g_ig);
        free(g_ik);
        free(got);
        free(g_vc);
        free(g_kc);
        free(st.idx_gate);
        free(st.idx_k);
        free(st.v_cache);
        free(st.k_cache);
    }
}

int main(void) {
    const char *dir = "tests/m4h/golden";
    long mlen;
    char mpath[512];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    char *man = (char *)read_file(mpath, &mlen);

    load_dims(man);
    run_probe(dir, man);
    load_kda_weights(dir);
    load_dsa_weights(dir);
    printf("m4h: weights loaded\n");
    fflush(stdout);
    test_kda(dir, man);
    test_dsa(dir, man);

    printf("digest: %016llx\n", (unsigned long long)dg_state);
    printf("%s: %d checks, %d failures\n",
           failures ? "FAIL" : "PASS", checks, failures);
    free(man);
    return failures ? 1 : 0;
}
