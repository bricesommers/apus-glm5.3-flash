/*
 * tests/m8g/test_m8g.c — M8b (GLM) MTP/NextN + speculative-decoding hard
 * gate: c/gmtp.h (the MTP draft head + ApusGspec engine) and the M8b
 * c/gmodel.h surfaces (per-token-interleaved decode batch, h-out surface,
 * MTP container load) against the M8a numpy-oracle fixtures (tests/m8g/
 * golden/<case>/, cases kda_top / dsa_top — see tests/m8g/README.md).
 *
 * Legs (per case):
 *   0. model + h surface: prefill/decode logits and the hnorm input built
 *      from the M8b h-out surface vs the goldens (pre_logits, dec_logits,
 *      pair_h) — validates the engine upstream of the MTP glue.
 *   1. DECODE-BATCH INTERLEAVE (the hard blocker): 6 tokens decoded
 *      one-by-one vs ONE apus_gmodel_decode_batch call from identical
 *      state — logits and the full state digest must be BITWISE equal.
 *      C-vs-C, host-independent, unconditional (by construction).
 *   2. mtp_forward batched true-pair replay vs replay_logits /
 *      replay_out_h goldens.
 *   3. draft chain (argmax drafts on the draft's own post-norm hidden) vs
 *      chain_drafts / chain_logits / chain_out_h goldens.
 *   4. EQUIVALENCE (the hard gate): spec decode (depth 1/2/3) vs non-spec
 *      decode, greedy AND sampled (temp 0.8, top_p 0.95, fixed seed) —
 *      emitted streams BITWISE identical. Plus ROLLBACK: state digest
 *      after the spec run == digest after decoding exactly the emitted
 *      tokens non-speculatively (rejected drafts leave no trace).
 *   5. Forced draft patterns (draft_override): truth-oracle (full accept +
 *      bonus), garbage (all reject), mixed (partial) — streams and state
 *      digests still bitwise == non-spec; accept stats match the pattern.
 *   6. TIERED: the greedy depth-3 equivalence re-run with the M6 cache
 *      (slots_per_layer=1, synchronous I/O) serving the MTP slabs (store
 *      layer n_main+0) — stream + state digest bitwise == the eager run.
 *
 * Golden-compare legs (0/2/3) keep the m4g/m5g two-tier host pattern:
 *   BITWISE   — host expf == numpy float32 exp (probed): every golden
 *               compared bitwise, free-running (A) and teacher-forced (B).
 *   TOLERANCE — otherwise: only the teacher-forced run B is compared,
 *               with the measured m5g compounding classes.
 * Legs 1/4/5/6 are engine-internal equivalences: bitwise unconditionally.
 *
 * The manifest's hnorm_input/pair_lag are CHECKED against the compiled-in
 * APUS_GMTP_HNORM_SRC / APUS_GMTP_PAIR_LAG — a pin update (tools/mtp_pin.py
 * verdict) that lands differently must flip the one-line constant in
 * c/gmtp.h AND regenerate the fixtures; this gate fails loudly until then.
 *
 * Run from the repository root. Prints an FNV-1a digest of the C outputs;
 * the Makefile diffs full output across APUS_THREADS=1/4/8.
 * APUS_M8G_TIER=tol forces the tolerance tier (measurement only).
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
#define APUS_SAMPLE_IMPLEMENTATION
#define APUS_GMTP_IMPLEMENTATION
#include "gmodel.h"
#include "gmtp.h"
#include "sample.h"
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

/* --- file IO (m5g conventions) -------------------------------------------*/

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
static void man_str(const char *man, const char *key, char *out,
                    size_t cap) {
    const char *p = man_find(man, key);
    size_t i = 0;
    while (p[i] && p[i] != '\n' && i + 1 < cap) {
        out[i] = p[i];
        i++;
    }
    out[i] = 0;
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

/* --- comparison (two-tier, m5g conventions) --------------------------------*/

static int bitwise_mode = 1;
static double tol_worst = 0.0;
static int cmp_enabled = 1;

/* m5g measured classes (tests/m5g/README.md — 1-ulp-perturbed numpy exp
 * goldens, ~2x headroom). */
#define TOL_H_REL      0.25
#define TOL_H_ABS      0.25
#define TOL_LOGIT_REL  0.25
#define TOL_LOGIT_ABS  0.05

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

/* Unconditional bitwise compare (engine-internal equivalence legs). */
static void cmp_codes_hard(const char *what, const uint16_t *got,
                           const uint16_t *want, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[i]) bad++;
    CHECK(bad == 0, "%s: %zu/%zu bf16 codes differ (hard bitwise)", what,
          bad, n);
    dg_add(got, n * sizeof(uint16_t));
}

/* --- exp probe (host-transcendental detection) -----------------------------*/

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
    const char *force = getenv("APUS_M8G_TIER");
    if (force && !strcmp(force, "tol")) bitwise_mode = 0;
    printf("probe: expf vs numpy f32 exp — %zu/%zu differ -> %s tier%s\n",
           bad, n, bitwise_mode ? "BITWISE" : "TOLERANCE",
           (force && !strcmp(force, "tol")) ? " (forced)" : "");
    free(x);
    free(y);
}

/* --- per-case context --------------------------------------------------------*/

#define M8G_PREFILL 12
#define M8G_DECODE 2
#define M8G_NTOK 14
#define M8G_CHAIN 3
#define M8G_CAP 64
#define EQ_TARGET 32
#define EQ_TRUTH_STEPS 40

typedef struct {
    const char *dir;
    const char *name;
    const char *man;
    ApusGmodel *m;
    ApusGmtp mt;
    int L, dim, V, hc, topk, width;
    int s_replay;               /* manifest replay_len = n_tokens - lag */
    int32_t *pids, *dids;               /* [12] / [2] */
    float *g_pre, *g_dec;               /* [12,V] / [2,V] */
    int32_t *pair_ids;                  /* [14] */
    float *pair_h;                      /* [14, dim] */
    float *g_replay_l, *g_replay_h;     /* [14,V] / [14,dim] */
    int32_t *g_chain_d;                 /* [3] */
    float *g_chain_l, *g_chain_h;       /* [3,V] / [3,dim] */
    const int32_t **ridx_pre, **ridx_dec;   /* [L] forced router rows */
    const int32_t **itopk_pre, **itopk_dec; /* [L] forced indexer rows */
    int32_t *fidx_replay, *ftopk_replay;    /* [14,topk] / [14,width] */
    int32_t *fidx_chain, *ftopk_chain;      /* [2,topk] / [2,width] */
} CaseCtx;

static void case_load(CaseCtx *cc, const char *base, const char *name) {
    char dir[512], nm[128], err[256];
    snprintf(dir, sizeof dir, "%s/%s", base, name);
    cc->dir = malloc(strlen(dir) + 1);
    strcpy((char *)cc->dir, dir);
    cc->name = name;
    char mpath[600];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    long mlen;
    cc->man = (const char *)read_file(mpath, &mlen);

    cc->L = (int)man_long(cc->man, "cfg_L");
    cc->dim = (int)man_long(cc->man, "cfg_dim");
    cc->V = (int)man_long(cc->man, "cfg_V");
    cc->hc = (int)man_long(cc->man, "cfg_hc");
    cc->topk = (int)man_long(cc->man, "cfg_topk");
    cc->width = (int)man_long(cc->man, "cfg_width");
    CHECK((int)man_long(cc->man, "prefill_len") == M8G_PREFILL
          && (int)man_long(cc->man, "decode_steps") == M8G_DECODE
          && (int)man_long(cc->man, "n_tokens") == M8G_NTOK
          && (int)man_long(cc->man, "replay_len")
                 == M8G_NTOK - (int)man_long(cc->man, "pair_lag")
          && (int)man_long(cc->man, "chain_depth") == M8G_CHAIN,
          "%s: manifest shape constants", name);
    /* the compiled-in pin must match the fixtures' (a mismatch = flip the
     * c/gmtp.h constant AND regenerate the fixtures) */
    char pin[64];
    man_str(cc->man, "hnorm_input", pin, sizeof pin);
    CHECK(!strcmp(pin, apus_gmtp_hnorm_src_name()),
          "%s: manifest hnorm_input=%s != compiled %s", name, pin,
          apus_gmtp_hnorm_src_name());
    CHECK(man_long(cc->man, "pair_lag") == APUS_GMTP_PAIR_LAG,
          "%s: manifest pair_lag != %d", name, APUS_GMTP_PAIR_LAG);
    cc->s_replay = (int)man_long(cc->man, "replay_len");

    const size_t dim = (size_t)cc->dim, v = (size_t)cc->V;
    cc->pids = load_i32(dir, "prompt_ids", M8G_PREFILL);
    cc->dids = load_i32(dir, "decode_ids", M8G_DECODE);
    cc->g_pre = load_f32(dir, "pre_logits", M8G_PREFILL * v);
    cc->g_dec = load_f32(dir, "dec_logits", M8G_DECODE * v);
    cc->pair_ids = load_i32(dir, "pair_ids", (size_t)cc->s_replay);
    cc->pair_h = load_f32(dir, "pair_h", (size_t)cc->s_replay * dim);
    cc->g_replay_l = load_f32(dir, "replay_logits", (size_t)cc->s_replay * v);
    cc->g_replay_h = load_f32(dir, "replay_out_h", (size_t)cc->s_replay * dim);
    cc->g_chain_d = load_i32(dir, "chain_drafts", M8G_CHAIN);
    cc->g_chain_l = load_f32(dir, "chain_logits", M8G_CHAIN * v);
    cc->g_chain_h = load_f32(dir, "chain_out_h", M8G_CHAIN * dim);
    cc->fidx_replay = load_i32(dir, "forced_idx_mtp_replay",
                               (size_t)cc->s_replay * (size_t)cc->topk);
    cc->ftopk_replay = load_i32(dir, "forced_topk_mtp_replay",
                                (size_t)cc->s_replay * (size_t)cc->width);
    cc->fidx_chain = load_i32(dir, "forced_idx_mtp_chain",
                              (M8G_CHAIN - 1) * (size_t)cc->topk);
    cc->ftopk_chain = load_i32(dir, "forced_topk_mtp_chain",
                               (M8G_CHAIN - 1) * (size_t)cc->width);

    /* model (eager) + MTP bind */
    snprintf(mpath, sizeof mpath, "%s/config.json", dir);
    char cdir[600];
    snprintf(cdir, sizeof cdir, "%s/container", dir);
    cc->m = apus_gmodel_open(cdir, mpath, err, sizeof err);
    if (!cc->m) {
        CHECK(0, "%s: apus_gmodel_open failed: %s", name, err);
        exit(1);
    }
    CHECK(apus_gmodel_has_mtp(cc->m), "%s: MTP block not loaded", name);
    CHECK(apus_gmodel_mtp_layer(cc->m) == cc->L,
          "%s: mtp layer id", name);
    if (apus_gmtp_bind(&cc->mt, cc->m, err, sizeof err)) {
        CHECK(0, "%s: apus_gmtp_bind failed: %s", name, err);
        exit(1);
    }
    const ApusGmodelExpertW *xw = apus_gmodel_expert(cc->m, cc->L, 0);
    CHECK(xw && xw->gate && xw->up && xw->down,
          "%s: MTP expert accessor", name);

    /* per-layer forced selections from the parsed config's kinds */
    const ApusGmodelConfig *cfg = apus_gmodel_config(cc->m);
    cc->ridx_pre = calloc((size_t)cc->L, sizeof(int32_t *));
    cc->ridx_dec = calloc((size_t)cc->L, sizeof(int32_t *));
    cc->itopk_pre = calloc((size_t)cc->L, sizeof(int32_t *));
    cc->itopk_dec = calloc((size_t)cc->L, sizeof(int32_t *));
    for (int l = 0; l < cc->L; l++) {
        if (cfg->mlp_is_sparse[l]) {
            snprintf(nm, sizeof nm, "forced_idx_pre_l%d", l);
            cc->ridx_pre[l] = load_i32(dir, nm,
                                       M8G_PREFILL * (size_t)cc->topk);
            snprintf(nm, sizeof nm, "forced_idx_dec_l%d", l);
            cc->ridx_dec[l] = load_i32(dir, nm,
                                       M8G_DECODE * (size_t)cc->topk);
        }
        if (cfg->layer_is_dsa[l]) {
            snprintf(nm, sizeof nm, "forced_topk_pre_l%d", l);
            cc->itopk_pre[l] = load_i32(dir, nm,
                                        M8G_PREFILL * (size_t)cc->width);
            snprintf(nm, sizeof nm, "forced_topk_dec_l%d", l);
            cc->itopk_dec[l] = load_i32(dir, nm,
                                        M8G_DECODE * (size_t)cc->width);
        }
    }
}

/* --- state digest (rollback check; live rows only) --------------------------*/

static uint64_t model_state_digest(const ApusGmodel *m,
                                   const ApusGmodelState *st) {
    size_t pos = apus_gmodel_pos(st);
    uint64_t d = 14695981039346656037ull;
#define DG(p, n) do { \
    const unsigned char *pp = (const unsigned char *)(p); \
    for (size_t i = 0; i < (size_t)(n); i++) { \
        d ^= pp[i]; d *= 1099511628211ull; \
    } \
} while (0)
    DG(&pos, sizeof pos);
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    const int L = c->num_hidden_layers;
    const size_t qkv = (size_t)c->linear_num_heads
                     * (size_t)c->linear_head_dim;
    const size_t ck = (size_t)c->linear_conv_kernel_dim;
    const size_t H = (size_t)c->num_attention_heads;
    const size_t qd = (size_t)c->qk_nope_head_dim;
    const size_t vd = (size_t)c->v_head_dim;
    const size_t ID = (size_t)c->index_head_dim;
    for (int l = 0; l < L; l++) {
        const ApusGkdaState *ks = apus_gmodel_kda_state(st, l);
        if (ks) {
            DG(ks->conv_state, 3 * qkv * (ck - 1) * sizeof(uint16_t));
            DG(ks->rec_state, qkv * (size_t)c->linear_head_dim
                              * sizeof(float));
        } else {
            const ApusGdsaState *ds = apus_gmodel_dsa_state(st, l);
            DG(&ds->n, sizeof ds->n);
            for (size_t hh = 0; hh < H; hh++) {
                DG(ds->k_cache + hh * ds->cap * qd,
                   ds->n * qd * sizeof(uint16_t));
                DG(ds->v_cache + hh * ds->cap * vd,
                   ds->n * vd * sizeof(uint16_t));
            }
            DG(ds->idx_k, ds->n * ID * sizeof(uint16_t));
            DG(ds->idx_gate, ds->n * ID * sizeof(uint16_t));
        }
    }
#undef DG
    return d;
}

/* --- leg 0: model + h surface (A free / B teacher-forced) -------------------*/

static void leg_model_h(CaseCtx *cc, int forced) {
    const size_t dim = (size_t)cc->dim, v = (size_t)cc->V;
    const size_t hcd = (size_t)cc->hc * dim;
    char what[192];
    ApusGmodelForces fp = { cc->ridx_pre, cc->itopk_pre };
    ApusGmodelState *st = apus_gmodel_state_new(cc->m, M8G_CAP);
    uint16_t *lg = malloc(M8G_PREFILL * v * sizeof(uint16_t));
    uint16_t *h_out = malloc(M8G_PREFILL * hcd * sizeof(uint16_t));
    uint16_t *yn_out = malloc(M8G_PREFILL * dim * sizeof(uint16_t));
    uint16_t *ph = malloc(dim * sizeof(uint16_t));
    cmp_enabled = bitwise_mode || forced;

    int rc = apus_gmodel_prefill_h(cc->m, st, cc->pids, M8G_PREFILL, lg,
                                   forced ? &fp : NULL, h_out, yn_out);
    CHECK(rc == 0, "%s %c: prefill rc", cc->name, forced ? 'B' : 'A');
    snprintf(what, sizeof what, "%s %c prefill logits", cc->name,
             forced ? 'B' : 'A');
    cmp_codes(what, lg, cc->g_pre, M8G_PREFILL * v,
              TOL_LOGIT_REL, TOL_LOGIT_ABS);
    for (size_t t = 0; t < M8G_PREFILL; t++) {
        apus_gmtp_hnorm_input(&cc->mt, h_out + t * hcd, yn_out + t * dim,
                              ph);
        snprintf(what, sizeof what, "%s %c pair_h[%zu] (h surface)",
                 cc->name, forced ? 'B' : 'A', t);
        cmp_codes(what, ph, cc->pair_h + t * dim, dim,
                  TOL_H_REL, TOL_H_ABS);
    }
    for (int k = 0; k < M8G_DECODE; k++) {
        const int32_t **rsel = calloc((size_t)cc->L, sizeof(int32_t *));
        const int32_t **tsel = calloc((size_t)cc->L, sizeof(int32_t *));
        for (int l = 0; l < cc->L; l++) {
            if (cc->ridx_dec[l])
                rsel[l] = cc->ridx_dec[l] + (size_t)k * (size_t)cc->topk;
            if (cc->itopk_dec[l])
                tsel[l] = cc->itopk_dec[l] + (size_t)k * (size_t)cc->width;
        }
        ApusGmodelForces fd = { rsel, tsel };
        rc = apus_gmodel_decode_batch(cc->m, st, cc->dids + k, 1, lg,
                                      forced ? &fd : NULL, h_out, yn_out);
        CHECK(rc == 0, "%s %c: decode %d rc", cc->name,
              forced ? 'B' : 'A', k);
        snprintf(what, sizeof what, "%s %c decode %d logits", cc->name,
                 forced ? 'B' : 'A', k);
        cmp_codes(what, lg, cc->g_dec + (size_t)k * v, v,
                  TOL_LOGIT_REL, TOL_LOGIT_ABS);
        /* the h-surface check covers rows the fixture's pair_h carries:
         * n_tokens - pair_lag rows (at lag 1 the LAST position has no
         * pair row — (h_{p-1}, tok_p) pairs h at p-1) */
        if (M8G_PREFILL + k < cc->s_replay) {
            apus_gmtp_hnorm_input(&cc->mt, h_out, yn_out, ph);
            snprintf(what, sizeof what, "%s %c pair_h[%d] (h surface)",
                     cc->name, forced ? 'B' : 'A', M8G_PREFILL + k);
            cmp_codes(what, ph, cc->pair_h + (M8G_PREFILL + k) * dim, dim,
                      TOL_H_REL, TOL_H_ABS);
        }
        free(rsel);
        free(tsel);
    }
    CHECK(apus_gmodel_pos(st) == M8G_NTOK, "%s %c: pos", cc->name,
          forced ? 'B' : 'A');
    apus_gmodel_state_free(st);
    free(lg);
    free(h_out);
    free(yn_out);
    free(ph);
}

/* --- leg 1: decode-batch interleave (hard bitwise, host-independent) --------*/

static void leg_interleave(CaseCtx *cc) {
    const size_t v = (size_t)cc->V;
    enum { NB = 6 };
    int32_t ids[NB];
    ids[0] = cc->dids[0];
    ids[1] = cc->dids[1];
    for (int i = 0; i < 4; i++) ids[2 + i] = cc->pids[i];

    /* sequential: one-by-one decode steps (logits buffers must hold the
     * PREFILL rows too — the prefill writes M8G_PREFILL > NB rows) */
    ApusGmodelState *ss = apus_gmodel_state_new(cc->m, M8G_CAP);
    uint16_t *lg_s = malloc(M8G_PREFILL * v * sizeof(uint16_t));
    if (apus_gmodel_prefill(cc->m, ss, cc->pids, M8G_PREFILL, lg_s, NULL,
                            NULL)) {
        CHECK(0, "%s: interleave seq prefill", cc->name);
        return;
    }
    for (int t = 0; t < NB; t++) {
        int rc = apus_gmodel_decode_step(cc->m, ss, ids[t],
                                         lg_s + t * v, NULL, NULL);
        CHECK(rc == 0, "%s: interleave seq step %d", cc->name, t);
    }
    uint64_t d_seq = model_state_digest(cc->m, ss);

    /* batched: ONE decode-batch call from identical state */
    ApusGmodelState *sb = apus_gmodel_state_new(cc->m, M8G_CAP);
    uint16_t *lg_b = malloc(M8G_PREFILL * v * sizeof(uint16_t));
    if (apus_gmodel_prefill(cc->m, sb, cc->pids, M8G_PREFILL, lg_b, NULL,
                            NULL)) {
        CHECK(0, "%s: interleave batch prefill", cc->name);
        return;
    }
    {
        int rc = apus_gmodel_decode_batch(cc->m, sb, ids, NB, lg_b, NULL,
                                          NULL, NULL);
        CHECK(rc == 0, "%s: interleave batch rc", cc->name);
    }
    uint64_t d_bat = model_state_digest(cc->m, sb);

    for (int t = 0; t < NB; t++) {
        char what[128];
        snprintf(what, sizeof what, "%s interleave logits t%d", cc->name,
                 t);
        cmp_codes_hard(what, lg_b + (size_t)t * v, lg_s + (size_t)t * v,
                       v);
    }
    CHECK(d_seq == d_bat,
          "%s: interleave state digest %016llx != %016llx", cc->name,
          (unsigned long long)d_bat, (unsigned long long)d_seq);
    dg_add(&d_bat, sizeof d_bat);
    printf("  %s interleave: %d tokens batched == sequential, state %s\n",
           cc->name, NB, d_seq == d_bat ? "BITWISE" : "DIFFERS");
    apus_gmodel_state_free(ss);
    apus_gmodel_state_free(sb);
    free(lg_s);
    free(lg_b);
}

/* --- legs 2+3: MTP replay + draft chain (A free / B teacher-forced) --------*/

static void leg_mtp(CaseCtx *cc, int forced) {
    const size_t dim = (size_t)cc->dim, v = (size_t)cc->V;
    char what[192];
    const char tag = forced ? 'B' : 'A';
    cmp_enabled = bitwise_mode || forced;
    const size_t sr = (size_t)cc->s_replay;
    uint16_t *pair_h = malloc(sr * dim * sizeof(uint16_t));
    /* the replay input hidden: bf16 codes of the golden pair_h */
    for (size_t i = 0; i < sr * dim; i++)
        pair_h[i] = apus_bf16_bits(cc->pair_h[i]);

    ApusGmtpState *mst = apus_gmtp_state_new(&cc->mt, M8G_CAP);
    uint16_t *lg = malloc(sr * v * sizeof(uint16_t));
    uint16_t *oh = malloc(sr * dim * sizeof(uint16_t));
    ApusGmtpForces fr = { cc->fidx_replay, cc->ftopk_replay };
    int rc = apus_gmtp_forward(&cc->mt, mst, cc->pair_ids, pair_h,
                               sr, lg, oh, forced ? &fr : NULL);
    CHECK(rc == 0, "%s %c: mtp replay rc", cc->name, tag);
    CHECK(mst->pos == sr, "%s %c: mtp pos %zu", cc->name, tag,
          mst->pos);
    snprintf(what, sizeof what, "%s %c mtp replay logits", cc->name, tag);
    cmp_codes(what, lg, cc->g_replay_l, sr * v,
              TOL_LOGIT_REL, TOL_LOGIT_ABS);
    snprintf(what, sizeof what, "%s %c mtp replay out_h", cc->name, tag);
    cmp_codes(what, oh, cc->g_replay_h, sr * dim,
              TOL_H_REL, TOL_H_ABS);

    /* draft chain over the replay-built state: d1 = argmax of the replay's
     * last logits row, then chain_depth-1 steps (the engine flow). */
    float *row = malloc(v * sizeof(float));
    for (size_t i = 0; i < v; i++)
        row[i] = apus_bf16_f32(lg[(sr - 1) * v + i]);
    int32_t d = apus_sample_argmax(row, v);
    if (bitwise_mode && !forced)
        CHECK(d == cc->g_chain_d[0], "%s A: chain d1 %d != golden %d",
              cc->name, d, cc->g_chain_d[0]);
    if (forced)
        d = cc->g_chain_d[0];   /* teacher-forced seed (near-tie class) */
    uint16_t *cur = malloc(dim * sizeof(uint16_t));
    memcpy(cur, oh + (sr - 1) * dim, dim * sizeof(uint16_t));
    int32_t drafts[M8G_CHAIN];
    drafts[0] = d;
    for (int i = 1; i < M8G_CHAIN; i++) {
        ApusGmtpForces fc = {
            cc->fidx_chain + (size_t)(i - 1) * (size_t)cc->topk,
            cc->ftopk_chain + (size_t)(i - 1) * (size_t)cc->width
        };
        rc = apus_gmtp_forward(&cc->mt, mst, &d, cur, 1, lg, oh,
                               forced ? &fc : NULL);
        CHECK(rc == 0, "%s %c: chain step %d rc", cc->name, tag, i);
        for (size_t j = 0; j < v; j++)
            row[j] = apus_bf16_f32(lg[j]);
        int32_t dn = apus_sample_argmax(row, v);
        snprintf(what, sizeof what, "%s %c chain step %d logits", cc->name,
                 tag, i);
        cmp_codes(what, lg, cc->g_chain_l + (size_t)i * v, v,
                  TOL_LOGIT_REL, TOL_LOGIT_ABS);
        snprintf(what, sizeof what, "%s %c chain step %d out_h", cc->name,
                 tag, i);
        cmp_codes(what, oh, cc->g_chain_h + (size_t)i * dim, dim,
                  TOL_H_REL, TOL_H_ABS);
        if (forced)
            dn = cc->g_chain_d[i];  /* teacher-forced chain input */
        else if (bitwise_mode)
            CHECK(dn == cc->g_chain_d[i],
                  "%s A: chain draft %d: %d != golden %d", cc->name, i,
                  dn, cc->g_chain_d[i]);
        d = dn;
        drafts[i] = d;
        memcpy(cur, oh, dim * sizeof(uint16_t));
    }
    if (!forced && bitwise_mode) {
        int dm = 1;
        for (int i = 0; i < M8G_CHAIN; i++)
            if (drafts[i] != cc->g_chain_d[i]) dm = 0;
        CHECK(dm, "%s A: chain drafts diverge", cc->name);
    }
    free(cur);
    free(row);
    free(lg);
    free(oh);
    free(pair_h);
    apus_gmtp_state_free(mst);
}

/* --- legs 4+5: spec equivalence + rollback + forced drafts -------------------*/

/* Non-speculative reference: prefill + `steps` sample/decode iterations.
 * out gets the emitted tokens; the state ends fed through n+steps-1... n
 * prompt + steps decoded = n+steps positions (the last sampled token is
 * fed too — mirrors the spec engine's fed==emitted accounting). */
static void nonspec_run(CaseCtx *cc, ApusGmodelState *st, int steps,
                        float temp, float top_p, uint64_t seed, int *out) {
    const size_t v = (size_t)cc->V;
    uint16_t *lg16 = malloc(M8G_PREFILL * v * sizeof(uint16_t));
    float *logits = malloc(v * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size(v));
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    if (apus_gmodel_prefill(cc->m, st, cc->pids, M8G_PREFILL, lg16, NULL,
                            NULL)) {
        CHECK(0, "%s: nonspec prefill", cc->name);
        exit(1);
    }
    for (size_t i = 0; i < v; i++)
        logits[i] = apus_bf16_f32(lg16[(M8G_PREFILL - 1) * v + i]);
    for (int k = 0; k < steps; k++) {
        int t = apus_sample(logits, v, temp, top_p, &rng, scratch);
        out[k] = t;
        int32_t nx = t;
        if (apus_gmodel_decode_step(cc->m, st, nx, lg16, NULL, NULL)) {
            CHECK(0, "%s: nonspec decode %d", cc->name, k);
            exit(1);
        }
        for (size_t i = 0; i < v; i++)
            logits[i] = apus_bf16_f32(lg16[i]);
    }
    free(lg16);
    free(logits);
    free(scratch);
}

typedef struct {
    const int32_t *truth;   /* full true sequence (prompt + stream) */
    int V;
    int mode;               /* 0 = truth, 1 = garbage, 2 = mixed (d2 ok,
                               d3+ wrong) */
} DraftCtx;

static void draft_hook(void *v, int64_t q, int32_t *drafts, int n) {
    DraftCtx *dc = v;
    for (int i = 0; i < n; i++) {
        int t = (int)dc->truth[q + i];
        if (dc->mode == 1 || (dc->mode == 2 && i >= 2))
            t = (t + 1) % dc->V;    /* != truth[q+i]: always rejected */
        drafts[i] = t;
    }
}

/* Speculative run. mt non-NULL: the real MTP head drafts; NULL: dc hooks
 * forced drafts. The stream may overshoot target to step granularity;
 * *emitted is the full emitted count. */
static void spec_run(CaseCtx *cc, ApusGmodelState *st, ApusGmtpState *mst,
                     int target, int depth, float temp, float top_p,
                     uint64_t seed, const ApusGmtp *mt, DraftCtx *dc,
                     int *out, int *emitted, uint64_t *acc, uint64_t *off) {
    void *scratch = malloc(apus_sample_scratch_size((size_t)cc->V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);
    ApusGspec sp;
    apus_gspec_init(&sp, cc->m, st, mt, mt ? mst : NULL, depth, temp,
                    top_p, &rng, scratch);
    if (!mt) {
        sp.draft_override = draft_hook;
        sp.draft_ctx = dc;
    }
    if (apus_gspec_prefill(&sp, cc->pids, M8G_PREFILL)) {
        CHECK(0, "%s: spec prefill", cc->name);
        exit(1);
    }
    int buf[64], n_out = 0;
    while ((int)sp.emitted < target) {
        int ne = apus_gspec_step(&sp, buf, 64);
        if (ne <= 0) break;
        for (int i = 0; i < ne; i++) out[n_out++] = buf[i];
    }
    *emitted = n_out;
    if (acc) *acc = sp.accepted;
    if (off) *off = sp.offered;
    apus_gspec_free(&sp);
    free(scratch);
}

static void leg_equivalence(CaseCtx *cc, float temp, float top_p,
                            uint64_t seed, const char *tag) {
    int ref[EQ_TRUTH_STEPS];
    ApusGmodelState *st0 = apus_gmodel_state_new(cc->m, M8G_CAP);
    nonspec_run(cc, st0, EQ_TRUTH_STEPS, temp, top_p, seed, ref);
    apus_gmodel_state_free(st0);

    for (int depth = 1; depth <= 3; depth++) {
        ApusGmodelState *st = apus_gmodel_state_new(cc->m, M8G_CAP);
        ApusGmtpState *mst =
            apus_gmtp_state_new(&cc->mt, M8G_CAP + depth);
        int out[128];
        int emitted = 0;
        uint64_t acc = 0, off = 0;
        spec_run(cc, st, mst, EQ_TARGET, depth, temp, top_p, seed,
                 &cc->mt, NULL, out, &emitted, &acc, &off);
        int same = emitted >= EQ_TARGET;
        for (int i = 0; i < EQ_TARGET && same; i++)
            if (out[i] != ref[i]) same = 0;
        CHECK(same, "%s %s: spec depth %d stream != non-spec", cc->name,
              tag, depth);
        /* rollback: state after the spec run == state after decoding
         * exactly `emitted` tokens non-speculatively */
        int ref2[128];
        ApusGmodelState *st_ref = apus_gmodel_state_new(cc->m, M8G_CAP);
        nonspec_run(cc, st_ref, emitted, temp, top_p, seed, ref2);
        uint64_t d_spec = model_state_digest(cc->m, st);
        uint64_t d_ref = model_state_digest(cc->m, st_ref);
        CHECK(d_spec == d_ref,
              "%s %s: spec depth %d state digest %016llx != %016llx",
              cc->name, tag, depth, (unsigned long long)d_spec,
              (unsigned long long)d_ref);
        int same2 = 1;
        for (int i = 0; i < emitted; i++)
            if (out[i] != ref2[i]) same2 = 0;
        CHECK(same2, "%s %s: spec depth %d full stream != non-spec",
              cc->name, tag, depth);
        printf("  %s %s depth %d: emitted %d, accept %llu/%llu, stream %s,"
               " state %s\n", cc->name, tag, depth, emitted,
               (unsigned long long)acc, (unsigned long long)off,
               same ? "BITWISE" : "DIFFERS",
               d_spec == d_ref ? "BITWISE" : "DIFFERS");
        dg_add(out, (size_t)emitted * sizeof(int));
        dg_add(&d_spec, sizeof d_spec);
        apus_gmtp_state_free(mst);
        apus_gmodel_state_free(st);
        apus_gmodel_state_free(st_ref);
    }
}

static void leg_forced(CaseCtx *cc) {
    /* truth = prompt + the non-spec greedy stream */
    int stream[EQ_TRUTH_STEPS];
    ApusGmodelState *st0 = apus_gmodel_state_new(cc->m, M8G_CAP);
    nonspec_run(cc, st0, EQ_TRUTH_STEPS, 0.0f, 1.0f, 0, stream);
    apus_gmodel_state_free(st0);
    int32_t truth[128];
    for (int i = 0; i < M8G_PREFILL; i++) truth[i] = cc->pids[i];
    for (int i = 0; i < EQ_TRUTH_STEPS; i++)
        truth[M8G_PREFILL + i] = stream[i];

    static const struct { int mode, depth; const char *name; } pats[] = {
        { 0, 2, "truth-oracle d2 (full accept+bonus)" },
        { 0, 3, "truth-oracle d3 (full accept+bonus)" },
        { 1, 2, "garbage d2 (all reject)" },
        { 1, 3, "garbage d3 (all reject)" },
        { 2, 3, "mixed d3 (partial accept)" },
    };
    for (size_t p = 0; p < sizeof pats / sizeof pats[0]; p++) {
        DraftCtx dc = { truth, cc->V, pats[p].mode };
        ApusGmodelState *st = apus_gmodel_state_new(cc->m, M8G_CAP);
        int out[128];
        int emitted = 0;
        uint64_t acc = 0, off = 0;
        spec_run(cc, st, NULL, EQ_TARGET, pats[p].depth, 0.0f, 1.0f, 0,
                 NULL, &dc, out, &emitted, &acc, &off);
        int same = emitted >= EQ_TARGET;
        for (int i = 0; i < EQ_TARGET && same; i++)
            if (out[i] != stream[i]) same = 0;
        CHECK(same, "%s forced %s: stream != non-spec", cc->name,
              pats[p].name);
        int ref2[128];
        ApusGmodelState *st_ref = apus_gmodel_state_new(cc->m, M8G_CAP);
        nonspec_run(cc, st_ref, emitted, 0.0f, 1.0f, 0, ref2);
        uint64_t d_spec = model_state_digest(cc->m, st);
        uint64_t d_ref = model_state_digest(cc->m, st_ref);
        CHECK(d_spec == d_ref, "%s forced %s: state digest differs",
              cc->name, pats[p].name);
        if (pats[p].mode == 0) CHECK(acc == off && off > 0,
              "%s forced %s: expected 100%% accept, got %llu/%llu",
              cc->name, pats[p].name, (unsigned long long)acc,
              (unsigned long long)off);
        if (pats[p].mode == 1) CHECK(acc == 0 && off > 0,
              "%s forced %s: expected 0%% accept, got %llu/%llu",
              cc->name, pats[p].name, (unsigned long long)acc,
              (unsigned long long)off);
        if (pats[p].mode == 2) CHECK(acc > 0 && acc < off,
              "%s forced %s: expected partial accept, got %llu/%llu",
              cc->name, pats[p].name, (unsigned long long)acc,
              (unsigned long long)off);
        printf("  %s forced %s: emitted %d, accept %llu/%llu, stream %s,"
               " state %s\n", cc->name, pats[p].name, emitted,
               (unsigned long long)acc, (unsigned long long)off,
               same ? "BITWISE" : "DIFFERS",
               d_spec == d_ref ? "BITWISE" : "DIFFERS");
        dg_add(out, (size_t)emitted * sizeof(int));
        dg_add(&d_spec, sizeof d_spec);
        apus_gmodel_state_free(st);
        apus_gmodel_state_free(st_ref);
    }
}

/* --- leg 6: tiered (M6 cache serving the MTP slabs) -------------------------*/

static void leg_tiered(CaseCtx *cc) {
    char err[256], cdir[600], cpath[600];
    snprintf(cdir, sizeof cdir, "%s/container", cc->dir);
    snprintf(cpath, sizeof cpath, "%s/config.json", cc->dir);
    ApusGmodelTierCfg tc;
    memset(&tc, 0, sizeof tc);
    tc.tiered = 1;
    tc.want_mtp = 1;
    tc.slots_per_layer = 1;     /* force misses: the slabs really stream */
    tc.io_threads = -1;         /* synchronous I/O (deterministic) */
    ApusGmodel *tm = apus_gmodel_open2(cdir, cpath, &tc, err, sizeof err);
    if (!tm) {
        CHECK(0, "%s: tiered open failed: %s", cc->name, err);
        return;
    }
    CHECK(apus_gmodel_has_mtp(tm), "%s: tiered MTP not loaded", cc->name);
    ApusGmtp mt;
    if (apus_gmtp_bind(&mt, tm, err, sizeof err)) {
        CHECK(0, "%s: tiered bind failed: %s", cc->name, err);
        apus_gmodel_close(tm);
        return;
    }
    /* expert accessor resolves through the cache at store layer L */
    const ApusGmodelExpertW *xw = apus_gmodel_expert(tm, cc->L, 1);
    CHECK(xw && xw->gate && xw->up && xw->down,
          "%s: tiered MTP expert resolve", cc->name);
    apus_gcache_layer_end(apus_gmodel_cache(tm), cc->L);

    /* greedy depth-3 equivalence against a fresh non-spec run on the SAME
     * (tiered) model, then the digests cross-checked vs the eager engine */
    const int depth = 3;
    int ref[EQ_TRUTH_STEPS];
    ApusGmodelState *st0 = apus_gmodel_state_new(tm, M8G_CAP);
    {
        /* non-spec on the tiered model (the m6g neutrality contract) */
        CaseCtx tc2 = *cc;
        tc2.m = tm;
        nonspec_run(&tc2, st0, EQ_TRUTH_STEPS, 0.0f, 1.0f, 0, ref);
    }
    apus_gmodel_state_free(st0);
    ApusGmodelState *st = apus_gmodel_state_new(tm, M8G_CAP);
    ApusGmtpState *mst = apus_gmtp_state_new(&mt, M8G_CAP + depth);
    int out[128];
    int emitted = 0;
    uint64_t acc = 0, off = 0;
    {
        CaseCtx tc2 = *cc;
        tc2.m = tm;
        spec_run(&tc2, st, mst, EQ_TARGET, depth, 0.0f, 1.0f, 0, &mt,
                 NULL, out, &emitted, &acc, &off);
        int same = emitted >= EQ_TARGET;
        for (int i = 0; i < EQ_TARGET && same; i++)
            if (out[i] != ref[i]) same = 0;
        CHECK(same, "%s tiered: spec depth %d stream != non-spec",
              cc->name, depth);
        int ref2[128];
        ApusGmodelState *st_ref = apus_gmodel_state_new(tm, M8G_CAP);
        nonspec_run(&tc2, st_ref, emitted, 0.0f, 1.0f, 0, ref2);
        uint64_t d_spec = model_state_digest(tm, st);
        uint64_t d_ref = model_state_digest(tm, st_ref);
        CHECK(d_spec == d_ref, "%s tiered: state digest differs",
              cc->name);
        /* tiered == eager (the m6g neutrality contract incl. MTP slabs):
         * the eager engine's non-spec run of the same tokens must give
         * the identical state digest */
        ApusGmodelState *st_eag = apus_gmodel_state_new(cc->m, M8G_CAP);
        nonspec_run(cc, st_eag, emitted, 0.0f, 1.0f, 0, ref2);
        uint64_t d_eag = model_state_digest(cc->m, st_eag);
        CHECK(d_ref == d_eag,
              "%s tiered-vs-eager: state digest %016llx != %016llx",
              cc->name, (unsigned long long)d_ref,
              (unsigned long long)d_eag);
        apus_gmodel_state_free(st_eag);
        printf("  %s tiered depth %d: emitted %d, accept %llu/%llu, "
               "stream %s, state %s, vs-eager %s\n", cc->name, depth,
               emitted, (unsigned long long)acc, (unsigned long long)off,
               same ? "BITWISE" : "DIFFERS",
               d_spec == d_ref ? "BITWISE" : "DIFFERS",
               d_ref == d_eag ? "BITWISE" : "DIFFERS");
        dg_add(out, (size_t)emitted * sizeof(int));
        dg_add(&d_spec, sizeof d_spec);
        dg_add(&d_eag, sizeof d_eag);
        apus_gmodel_state_free(st_ref);
    }
    apus_gmtp_state_free(mst);
    apus_gmodel_state_free(st);
    apus_gmodel_close(tm);
}

/* --- per-case driver ---------------------------------------------------------*/

static void run_case(const char *base, const char *name) {
    CaseCtx cc;
    memset(&cc, 0, sizeof cc);
    case_load(&cc, base, name);
    run_probe(cc.dir, cc.man);
    printf("case %s: L=%d dim=%d V=%d hc=%d\n", name, cc.L, cc.dim, cc.V,
           cc.hc);

    leg_model_h(&cc, 0);
    leg_model_h(&cc, 1);
    leg_interleave(&cc);
    leg_mtp(&cc, 0);
    leg_mtp(&cc, 1);
    leg_equivalence(&cc, 0.0f, 1.0f, 0, "greedy");
    leg_equivalence(&cc, 0.8f, 0.95f, 12345, "sampled");
    leg_forced(&cc);
    leg_tiered(&cc);

    apus_gmodel_close(cc.m);
    for (int l = 0; l < cc.L; l++) {
        free((void *)cc.ridx_pre[l]);
        free((void *)cc.ridx_dec[l]);
        free((void *)cc.itopk_pre[l]);
        free((void *)cc.itopk_dec[l]);
    }
    free(cc.ridx_pre);
    free(cc.ridx_dec);
    free(cc.itopk_pre);
    free(cc.itopk_dec);
    free(cc.pids);
    free(cc.dids);
    free(cc.g_pre);
    free(cc.g_dec);
    free(cc.pair_ids);
    free(cc.pair_h);
    free(cc.g_replay_l);
    free(cc.g_replay_h);
    free(cc.g_chain_d);
    free(cc.g_chain_l);
    free(cc.g_chain_h);
    free(cc.fidx_replay);
    free(cc.ftopk_replay);
    free(cc.fidx_chain);
    free(cc.ftopk_chain);
    free((void *)cc.dir);
    free((void *)cc.man);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* progress survives a crash */
    printf("test_m8g: M8b GLM MTP + speculative decoding verification\n");
    static const char *cases[] = { "kda_top", "dsa_top" };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        run_case("tests/m8g/golden", cases[i]);
    printf("tol_worst: %.3g\n", tol_worst);
    printf("digest: %016llx\n", (unsigned long long)dg_state);
    printf("%s: %d checks, %d failures\n",
           failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
