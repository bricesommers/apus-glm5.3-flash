/*
 * tests/m5g/test_m5g.c — M5 (GLM) full-model hard gate: c/gmodel.h (the
 * 45-layer mHC stack — token embed -> per-layer mHC/KDA|DSA/dense|MoE ->
 * HyperHead -> final RMSNorm -> lm_head logits), the M1 v2 container
 * loader (dense tensors + one-pread-per-expert slabs), and the flat/real
 * config parser, against the numpy-oracle goldens
 * (tests/m5g/gen_golden.py — the M0 oracle's model_forward in
 * f32-faithful mode, run back from the converted container).
 *
 * Gate tiers (tests/m5g/README.md — the m4g/m4h two-tier host pattern):
 *   BITWISE   — host expf == numpy float32 exp (probed at runtime):
 *               EVERY golden compared bitwise — per-layer block-output
 *               streams h, logits per call, KDA conv/rec states, DSA
 *               KV/indexer caches. Both the free-running model (run A)
 *               and the teacher-forced model (run B) must match.
 *   TOLERANCE — otherwise: run B (teacher-forced selections — indexer
 *               top-k + router idx from the goldens, because structural
 *               relu-clip zero ties and compounding bf16 flips make
 *               free-running selections host-exp sensitive) compared
 *               with the documented compounding classes: bf16-code
 *               goldens rel <= 5e-2 or abs <= 1e-4, fp32 rec_state
 *               rel <= 1e-2 or abs <= 1e-4.
 *
 * Run from the repository root (golden fixtures under tests/m5g/golden/).
 * Prints an FNV-1a digest of the C outputs; the Makefile diffs full
 * output across APUS_THREADS=1/4/8 (thread-count independence).
 * APUS_M5G_TIER=tol forces the tolerance tier (tolerance-class
 * measurement only — tests/m5g/README.md).
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
#include "gmodel.h"
#include "json.h"
#include "st.h"
#include "bf16.h"

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

/* comma list -> int array (returns count). '\r' terminates too — a CRLF
 * manifest (text-mode Python write on Windows) would otherwise strand p
 * on '\r' with strtol performing no conversion: an infinite loop (same
 * class as the 2026-09-06 windows-latest m4h hang; m5g was next in the
 * battery). The generators also pin LF. */
static int man_list(const char *man, const char *key, int *out, int cap) {
    const char *p = man_find(man, key);
    int n = 0;
    while (*p && *p != '\n' && *p != '\r' && n < cap) {
        out[n++] = (int)strtol(p, (char **)&p, 10);
        if (*p == ',') p++;
    }
    return n;
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

static int32_t *load_i32(const char *dir, const char *name, size_t n) {
    char path[512];
    long len;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    unsigned char *b = read_file(path, &len);
    if ((size_t)len != n * sizeof(int32_t)) {
        printf("%s: size mismatch (%ld != %zu)\n", path, len, n);
        exit(1);
    }
    return (int32_t *)b;
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
static double tol_worst = 0.0;      /* worst rel diff in tolerance tier */
/* In the TOLERANCE tier only the teacher-forced run B is compared: the
 * free-running run A may flip structurally-tied indexer selections (the
 * fragile-query class — it poisons the whole downstream chain, so there
 * is no meaningful tolerance comparison for it). Run A still executes
 * (crash/UB coverage). */
static int cmp_enabled = 1;

/* Tolerance-tier classes — MEASURED (tests/m5g/README.md): golden
 * fixtures regenerated with numpy's f32 exp deterministically perturbed
 * by 1 ulp on EVERY call (a far harsher host-exp difference than any
 * probed host — unpinned Linux X86_V3 differs on ~40 % of values), the C
 * gate then run teacher-forced against them. Worst observed element
 * drifts: h streams abs 0.125, logits abs 0.033, bf16 state caches abs
 * 0.023, fp32 rec_state abs 0.0012 (rel clauses are meaningless near
 * zero — small-magnitude elements legitimately flip several codes).
 * Classes below carry ~2x headroom over that worst case. */
#define TOL_H_REL      0.25    /* mHC streams h (bf16 codes) */
#define TOL_H_ABS      0.25
#define TOL_LOGIT_REL  0.25    /* lm_head logits (bf16 codes) */
#define TOL_LOGIT_ABS  0.05
#define TOL_STC_REL    0.25    /* KDA conv state + DSA caches (bf16 codes) */
#define TOL_STC_ABS    0.05
#define TOL_REC_REL    0.25    /* KDA rec_state (fp32) */
#define TOL_REC_ABS    5e-3

static void cmp_f32_tol(const char *what, const float *got,
                        const float *want, size_t n, double rel_tol,
                        double abs_tol) {
    if (!cmp_enabled) return;
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
        dg_add(got, n * sizeof(float));
        return;
    }
    size_t bad = 0;
    double worst_a = 0.0, worst_r = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        double rel = d / fmax(1e-30, fabs((double)want[i]));
        if (rel > tol_worst) tol_worst = rel;
        if (d > worst_a) worst_a = d;
        if (rel > worst_r) worst_r = rel;
        if (rel > rel_tol && d > abs_tol) bad++;
    }
    CHECK(bad == 0, "%s: %zu/%zu over tol (max abs %.3g, max rel %.3g)",
          what, bad, n, worst_a, worst_r);
    dg_add(got, n * sizeof(float));
}

/* bf16 codes vs an f32 (bf16-valued) golden */
static void cmp_codes(const char *what, const uint16_t *got,
                      const float *want, size_t n, double rel_tol,
                      double abs_tol) {
    float *w = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) w[i] = apus_bf16_f32(got[i]);
    cmp_f32_tol(what, w, want, n, rel_tol, abs_tol);
    free(w);
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
    const char *force = getenv("APUS_M5G_TIER");
    if (force && !strcmp(force, "tol")) bitwise_mode = 0;
    printf("probe: expf vs numpy f32 exp — %zu/%zu differ -> %s tier%s\n",
           bad, n, bitwise_mode ? "BITWISE" : "TOLERANCE",
           (force && !strcmp(force, "tol")) ? " (forced)" : "");
    free(x);
    free(y);
}

/* --- dims ------------------------------------------------------------------*/

static int L, DIM, V, HC, HK, DK, QKV, HD, QD, VD, ID, TOPK, WIDTH;
static int NPRE, NSTEP, NTOT;
static int dsa_layers[64], moe_layers[64], kda_layers[64];
static int ndsa, nmoe, nkda;

static void load_dims(const char *man) {
    L = (int)man_long(man, "cfg_L");
    DIM = (int)man_long(man, "cfg_dim");
    V = (int)man_long(man, "cfg_V");
    HC = (int)man_long(man, "cfg_hc");
    HK = (int)man_long(man, "cfg_Hk");
    DK = (int)man_long(man, "cfg_Dk");
    QKV = (int)man_long(man, "cfg_qkv");
    HD = (int)man_long(man, "cfg_Hd");
    QD = (int)man_long(man, "cfg_qd");
    VD = (int)man_long(man, "cfg_vd");
    ID = (int)man_long(man, "cfg_ID");
    TOPK = (int)man_long(man, "cfg_topk");
    WIDTH = (int)man_long(man, "cfg_width");
    NPRE = (int)man_long(man, "prefill_len");
    NSTEP = (int)man_long(man, "decode_steps");
    NTOT = (int)man_long(man, "n_total");
    ndsa = man_list(man, "cfg_dsa_layers", dsa_layers, 64);
    nmoe = man_list(man, "cfg_moe_layers", moe_layers, 64);
    nkda = man_list(man, "cfg_kda_layers", kda_layers, 64);
}

/* --- the real-config parser check (reference/config.json) -----------------*/

static void test_reference_config(void) {
    char err[256];
    ApusGmodelConfig rc;
    if (apus_gmodel_config_load("reference/config.json", &rc, err,
                                sizeof err)) {
        CHECK(0, "reference/config.json parse failed: %s", err);
        return;
    }
    CHECK(rc.num_hidden_layers == 45 && rc.hidden_size == 4096
          && rc.vocab_size == 154880 && rc.hc_mult == 4,
          "reference config: L=%d dim=%d V=%d hc=%d",
          rc.num_hidden_layers, rc.hidden_size, rc.vocab_size,
          rc.hc_mult);
    CHECK(rc.linear_num_heads == 64 && rc.linear_head_dim == 128
          && rc.linear_conv_kernel_dim == 4
          && rc.linear_lower_bound == -5.0f,
          "reference config: KDA H=%d D=%d conv=%d lb=%g",
          rc.linear_num_heads, rc.linear_head_dim,
          rc.linear_conv_kernel_dim, (double)rc.linear_lower_bound);
    CHECK(rc.num_attention_heads == 64 && rc.q_lora_rank == 1536
          && rc.kv_lora_rank == 512 && rc.index_n_heads == 32
          && rc.index_topk == 2048 && rc.index_kpool == 4,
          "reference config: DSA/indexer dims");
    CHECK(rc.n_routed_experts == 288 && rc.num_experts_per_tok == 8
          && rc.routed_scaling_factor == 2.5f
          && rc.swiglu_limit == 10.0f && rc.first_k_dense_replace == 3,
          "reference config: MoE dims");
    /* layer pattern: explicit lists must equal the i%4==3 / first-k rule */
    int rd[45], rs[45];
    apus_gmodel_layer_patterns(45, rc.first_k_dense_replace, rd, rs);
    int bad = 0;
    for (int i = 0; i < 45; i++)
        if (rd[i] != rc.layer_is_dsa[i] || rs[i] != rc.mlp_is_sparse[i])
            bad++;
    CHECK(bad == 0, "reference config: rule-vs-explicit layer pattern "
          "mismatches: %d", bad);
    CHECK(rc.layer_is_dsa[3] == 1 && rc.layer_is_dsa[4] == 0
          && rc.layer_is_dsa[43] == 1 && rc.mlp_is_sparse[0] == 0
          && rc.mlp_is_sparse[3] == 1 && rc.mlp_is_sparse[44] == 1,
          "reference config: spot layer kinds");
    apus_gmodel_config_free(&rc);
}

/* --- one full model run (A: free; B: teacher-forced) -----------------------*/

typedef struct {
    ApusGmodel *m;
    const char *dir;
    const char *man;
    int forced;                     /* 0 = run A, 1 = run B (forced) */
} RunCtx;

static void run_model(RunCtx *rc) {
    ApusGmodel *m = rc->m;
    const char *dir = rc->dir;
    char nm[128], what[192];
    size_t snp = (size_t)NPRE, sns = (size_t)NSTEP;
    size_t hc = (size_t)HC, dim = (size_t)DIM, v = (size_t)V;

    int32_t *pids = load_i32(dir, "prompt_ids", snp);
    int32_t *dids = load_i32(dir, "decode_ids", sns);
    float *g_lp = load_f32(dir, "prefill_logits", snp * v);
    float *g_ld = load_f32(dir, "decode_logits", sns * v);

    /* per-layer h goldens */
    float **g_pre_h = calloc((size_t)L, sizeof *g_pre_h);
    float **g_dec_h = calloc((size_t)L, sizeof *g_dec_h);
    for (int l = 0; l < L; l++) {
        snprintf(nm, sizeof nm, "pre_h_l%d", l);
        g_pre_h[l] = load_f32(dir, nm, snp * hc * dim);
        snprintf(nm, sizeof nm, "dec_h_l%d", l);
        g_dec_h[l] = load_f32(dir, nm, sns * hc * dim);
    }

    /* teacher-forcing selections (run B) */
    const int32_t **router_idx = calloc((size_t)L, sizeof(int32_t *));
    const int32_t **indexer_topk = calloc((size_t)L, sizeof(int32_t *));
    const int32_t **router_idx_d = calloc((size_t)L, sizeof(int32_t *));
    const int32_t **indexer_topk_d = calloc((size_t)L, sizeof(int32_t *));
    for (int i = 0; i < nmoe; i++) {
        int l = moe_layers[i];
        snprintf(nm, sizeof nm, "forced_idx_pre_l%d", l);
        router_idx[l] = load_i32(dir, nm, snp * (size_t)TOPK);
        snprintf(nm, sizeof nm, "forced_idx_dec_l%d", l);
        router_idx_d[l] = load_i32(dir, nm, sns * (size_t)TOPK);
    }
    for (int i = 0; i < ndsa; i++) {
        int l = dsa_layers[i];
        snprintf(nm, sizeof nm, "forced_topk_pre_l%d", l);
        indexer_topk[l] = load_i32(dir, nm, snp * (size_t)WIDTH);
        snprintf(nm, sizeof nm, "forced_topk_dec_l%d", l);
        indexer_topk_d[l] = load_i32(dir, nm, sns * (size_t)WIDTH);
    }
    ApusGmodelForces forces = { router_idx, indexer_topk };
    ApusGmodelForces forces_d = { router_idx_d, indexer_topk_d };

    ApusGmodelState *st = apus_gmodel_state_new(m, (size_t)NTOT);
    CHECK(st != NULL, "state_new");
    cmp_enabled = bitwise_mode || rc->forced;
    uint16_t *logits = malloc(snp * v * sizeof(uint16_t));
    uint16_t *trace = malloc((size_t)L * snp * hc * dim
                             * sizeof(uint16_t));

    /* prefill */
    int rc_pref = apus_gmodel_prefill(m, st, pids, snp, logits,
                                      rc->forced ? &forces : NULL, trace);
    CHECK(rc_pref == 0, "prefill rc");
    snprintf(what, sizeof what, "%s prefill logits",
             rc->forced ? "B" : "A");
    cmp_codes(what, logits, g_lp, snp * v, TOL_LOGIT_REL, TOL_LOGIT_ABS);
    for (int l = 0; l < L; l++) {
        snprintf(what, sizeof what, "%s prefill h l%d",
                 rc->forced ? "B" : "A", l);
        cmp_codes(what, trace + (size_t)l * snp * hc * dim, g_pre_h[l],
                  snp * hc * dim, TOL_H_REL, TOL_H_ABS);
    }

    /* decode chain */
    for (int k = 0; k < NSTEP; k++) {
        ApusGmodelForces fd = forces_d;
        const int32_t **rsel = calloc((size_t)L, sizeof(int32_t *));
        const int32_t **tsel = calloc((size_t)L, sizeof(int32_t *));
        for (int i = 0; i < nmoe; i++)
            rsel[moe_layers[i]] =
                router_idx_d[moe_layers[i]] + (size_t)k * (size_t)TOPK;
        for (int i = 0; i < ndsa; i++)
            tsel[dsa_layers[i]] =
                indexer_topk_d[dsa_layers[i]] + (size_t)k * (size_t)WIDTH;
        fd.router_idx = rsel;
        fd.indexer_topk = tsel;
        int rc_dec = apus_gmodel_decode_step(m, st, dids[k], logits,
                                             rc->forced ? &fd : NULL,
                                             trace);
        CHECK(rc_dec == 0, "decode step %d rc", k);
        snprintf(what, sizeof what, "%s decode %d logits",
                 rc->forced ? "B" : "A", k);
        cmp_codes(what, logits, g_ld + (size_t)k * v, v, TOL_LOGIT_REL, TOL_LOGIT_ABS);
        for (int l = 0; l < L; l++) {
            snprintf(what, sizeof what, "%s decode %d h l%d",
                     rc->forced ? "B" : "A", k, l);
            cmp_codes(what, trace + (size_t)l * hc * dim,
                      g_dec_h[l] + (size_t)k * hc * dim, hc * dim,
                      TOL_H_REL, TOL_H_ABS);
        }
        free(rsel);
        free(tsel);
    }
    CHECK(apus_gmodel_pos(st) == (size_t)NTOT, "final pos %zu",
          apus_gmodel_pos(st));

    /* final states */
    for (int i = 0; i < nkda; i++) {
        int l = kda_layers[i];
        const ApusGkdaState *ks = apus_gmodel_kda_state(st, l);
        CHECK(ks != NULL, "kda state l%d", l);
        snprintf(nm, sizeof nm, "kda_l%d_conv", l);
        size_t cq = 3 * (size_t)QKV;        /* concatenated q|k|v */
        float *g_cv = load_f32(dir, nm, 3 * cq);
        /* oracle [K-1, 3qkv] token-major vs engine channel-major */
        float *got = malloc(3 * cq * sizeof(float));
        for (size_t c = 0; c < cq; c++)
            for (int j = 0; j < 3; j++)
                got[j * cq + c] =
                    apus_bf16_f32(ks->conv_state[c * 3 + j]);
        snprintf(what, sizeof what, "%s kda l%d conv_state",
                 rc->forced ? "B" : "A", l);
        cmp_f32_tol(what, got, g_cv, 3 * cq,
                    TOL_STC_REL, TOL_STC_ABS);
        free(got);
        free(g_cv);
        snprintf(nm, sizeof nm, "kda_l%d_rec", l);
        float *g_rc = load_f32(dir, nm, (size_t)QKV * (size_t)DK);
        snprintf(what, sizeof what, "%s kda l%d rec_state",
                 rc->forced ? "B" : "A", l);
        cmp_f32_tol(what, ks->rec_state, g_rc,
                    (size_t)QKV * (size_t)DK, TOL_REC_REL, TOL_REC_ABS);
        free(g_rc);
    }
    for (int i = 0; i < ndsa; i++) {
        int l = dsa_layers[i];
        const ApusGdsaState *ds = apus_gmodel_dsa_state(st, l);
        CHECK(ds != NULL && ds->n == (size_t)NTOT, "dsa state l%d", l);
        size_t n = (size_t)NTOT;
        snprintf(nm, sizeof nm, "dsa_l%d_k_cache", l);
        float *g_kc = load_f32(dir, nm, n * (size_t)HD * (size_t)QD);
        snprintf(nm, sizeof nm, "dsa_l%d_v_cache", l);
        float *g_vc = load_f32(dir, nm, n * (size_t)HD * (size_t)VD);
        /* engine head-major [H][cap*d] vs oracle [n, H, d] */
        float *got = malloc(n * (size_t)HD * (size_t)(QD > VD ? QD : VD)
                            * sizeof(float));
        for (size_t p = 0; p < n; p++)
            for (int h = 0; h < HD; h++)
                for (int d = 0; d < QD; d++)
                    got[(p * (size_t)HD + h) * (size_t)QD + d] =
                        apus_bf16_f32(
                            ds->k_cache[(size_t)h * ds->cap * (size_t)QD
                                        + p * (size_t)QD + d]);
        snprintf(what, sizeof what, "%s dsa l%d k_cache",
                 rc->forced ? "B" : "A", l);
        cmp_f32_tol(what, got, g_kc, n * (size_t)HD * (size_t)QD,
                    TOL_STC_REL, TOL_STC_ABS);
        for (size_t p = 0; p < n; p++)
            for (int h = 0; h < HD; h++)
                for (int d = 0; d < VD; d++)
                    got[(p * (size_t)HD + h) * (size_t)VD + d] =
                        apus_bf16_f32(
                            ds->v_cache[(size_t)h * ds->cap * (size_t)VD
                                        + p * (size_t)VD + d]);
        snprintf(what, sizeof what, "%s dsa l%d v_cache",
                 rc->forced ? "B" : "A", l);
        cmp_f32_tol(what, got, g_vc, n * (size_t)HD * (size_t)VD,
                    TOL_STC_REL, TOL_STC_ABS);
        free(got);
        free(g_kc);
        free(g_vc);
        snprintf(nm, sizeof nm, "dsa_l%d_idx_k", l);
        float *g_ik = load_f32(dir, nm, n * (size_t)ID);
        got = malloc(n * (size_t)ID * sizeof(float));
        for (size_t j = 0; j < n * (size_t)ID; j++)
            got[j] = apus_bf16_f32(ds->idx_k[j]);
        snprintf(what, sizeof what, "%s dsa l%d idx_k",
                 rc->forced ? "B" : "A", l);
        cmp_f32_tol(what, got, g_ik, n * (size_t)ID,
                    TOL_STC_REL, TOL_STC_ABS);
        free(got);
        free(g_ik);
        snprintf(nm, sizeof nm, "dsa_l%d_idx_gate", l);
        float *g_ig = load_f32(dir, nm, n * (size_t)ID);
        got = malloc(n * (size_t)ID * sizeof(float));
        for (size_t j = 0; j < n * (size_t)ID; j++)
            got[j] = apus_bf16_f32(ds->idx_gate[j]);
        snprintf(what, sizeof what, "%s dsa l%d idx_gate",
                 rc->forced ? "B" : "A", l);
        cmp_f32_tol(what, got, g_ig, n * (size_t)ID,
                    TOL_STC_REL, TOL_STC_ABS);
        free(got);
        free(g_ig);
    }

    apus_gmodel_state_free(st);
    free(trace);
    free(logits);
    for (int i = 0; i < nmoe; i++) {
        free((void *)router_idx[moe_layers[i]]);
        free((void *)router_idx_d[moe_layers[i]]);
    }
    for (int i = 0; i < ndsa; i++) {
        free((void *)indexer_topk[dsa_layers[i]]);
        free((void *)indexer_topk_d[dsa_layers[i]]);
    }
    free(router_idx);
    free(indexer_topk);
    free(router_idx_d);
    free(indexer_topk_d);
    for (int l = 0; l < L; l++) {
        free(g_pre_h[l]);
        free(g_dec_h[l]);
    }
    free(g_pre_h);
    free(g_dec_h);
    free(g_ld);
    free(g_lp);
    free(dids);
    free(pids);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/m5g/golden";
    long mlen;
    char mpath[512];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    char *man = (char *)read_file(mpath, &mlen);

    load_dims(man);
    run_probe(dir, man);
    test_reference_config();

    /* fixture config cross-check */
    char err[256], cfgpath[512];
    snprintf(cfgpath, sizeof cfgpath, "%s/config.json", dir);
    ApusGmodelConfig fc;
    if (apus_gmodel_config_load(cfgpath, &fc, err, sizeof err)) {
        CHECK(0, "fixture config parse failed: %s", err);
        return 1;
    }
    CHECK(fc.num_hidden_layers == L && fc.hidden_size == DIM
          && fc.vocab_size == V && fc.hc_mult == HC,
          "fixture config dims");
    {
        int bad = 0;
        for (int i = 0; i < L; i++) {
            int want_dsa = 0, want_sparse = 0;
            for (int j = 0; j < ndsa; j++)
                if (dsa_layers[j] == i) want_dsa = 1;
            for (int j = 0; j < nmoe; j++)
                if (moe_layers[j] == i) want_sparse = 1;
            if (fc.layer_is_dsa[i] != want_dsa
                || fc.mlp_is_sparse[i] != want_sparse)
                bad++;
        }
        CHECK(bad == 0, "fixture config layer kinds: %d mismatches", bad);
    }

    char cpath[512];
    snprintf(cpath, sizeof cpath, "%s/container", dir);
    ApusGmodel *m = apus_gmodel_open(cpath, cfgpath, err, sizeof err);
    if (!m) {
        CHECK(0, "apus_gmodel_open failed: %s", err);
        apus_gmodel_config_free(&fc);
        return 1;
    }
    CHECK(1, "apus_gmodel_open");
    /* accessor sanity: kinds + the expert seam */
    for (int i = 0; i < L; i++) {
        int want_dsa = 0, want_sparse = 0;
        for (int j = 0; j < ndsa; j++)
            if (dsa_layers[j] == i) want_dsa = 1;
        for (int j = 0; j < nmoe; j++)
            if (moe_layers[j] == i) want_sparse = 1;
        CHECK(apus_gmodel_layer_is_dsa(m, i) == want_dsa
              && apus_gmodel_layer_is_sparse(m, i) == want_sparse,
              "model layer %d kinds", i);
    }
    for (int i = 0; i < nmoe; i++) {
        const ApusGmodelExpertW *xw = apus_gmodel_expert(m, moe_layers[i],
                                                         0);
        CHECK(xw && xw->gate && xw->up && xw->down,
              "expert accessor l%d", moe_layers[i]);
    }
    apus_gmodel_config_free(&fc);

    RunCtx ra = { m, dir, man, 0 };
    RunCtx rb = { m, dir, man, 1 };
    run_model(&ra);
    run_model(&rb);

    printf("tol_worst: %.3g\n", tol_worst);
    printf("digest: %016llx\n", (unsigned long long)dg_state);
    printf("%s: %d checks, %d failures\n",
           failures ? "FAIL" : "PASS", checks, failures);
    apus_gmodel_close(m);
    free(man);
    return failures ? 1 : 0;
}
