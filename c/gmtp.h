/*
 * c/gmtp.h — GLM-5.3-Flash MTP (classic NextN) draft head + speculative
 * decoding (M8b). Port of the base engine's M8 (../Apus c/mtp.h — ApusSpec
 * step shape, snapshot/rollback, accept rule) adapted to the GLM stack
 * (c/gmodel.h M8b surfaces). C11, libc only (+ the M3/M4/M5 GLM building
 * blocks and c/sample.h).
 *
 * Normative reference: tools/oracle.py mtp_forward / mtp_chain (M8a; the
 * oracle ports SGLang deepseek_nextn.py — HF drops layer 45 at load).
 * Gated BITWISE against tests/m8g goldens.
 *
 * MTP block (layers.<L>.* — a full DSA + sparse-MoE decoder layer WITH a
 * shared expert, PLAIN bf16-rounded residuals, NO mHC):
 *     eh = cat([rms_norm(embed(tok), enorm), rms_norm(prev_h, hnorm)], -1)
 *     x  = bf16_linear(eh, eh_proj)            (eh_proj is BF16 [dim,2dim])
 *     x  = bf16(x + DSA(rms_norm(x, input_norm)))      (one rounding)
 *     x  = bf16(x + MoE(rms_norm(x, post_norm)))       (one rounding)
 *     out = rms_norm(x, shared_head.norm)              (fused add+norm)
 *     logits = bf16_linear(out, SHARED lm_head)
 * Drafts are always argmax of the MTP logits. The chain feeds the draft's
 * own post-shared_head.norm hidden back as prev_h with the drafted token's
 * embedding (vLLM PR #47448 semantics = the parent's apus_spec_chain).
 * rms_norm is the GLM weighted two-rounding RMSNorm (c/gdsa.h
 * apus_gdsa_rmsnorm); every matmul is the m3g sequential-k kernel (bitwise
 * == the oracle _mm); the plain residual adds are one f32 add of
 * bf16-valued operands + one RNE bf16 rounding (the oracle's _B(res + x)).
 *
 * (h, id) pairing per APUS_GMTP_PAIR_LAG: lag 0 (the default) — the true
 * pair at position p is (h_p, tok_p) with h_p the POST-token main hidden
 * at p, predicting tok_{p+1}; lag 1 — the DeepSeek-V3/SGLang EAGLE
 * bookkeeping ((h_{p-1}, tok_p) predicts tok_{p+1}). The hnorm input is ONE
 * 4096-wide hidden collapsed
 * from the trunk's mHC stream: APUS_GMTP_HNORM_SRC selects the source
 * (PINNED: prenorm_mean = the HyperHead unweighted mean, bf16-rounded —
 * oracle.MTP_HNORM_INPUT; the tools/mtp_pin.py real-container verdict
 * flips the one-line constant if it lands elsewhere; the m8g gate checks
 * the compiled-in source against the fixture manifest and fails loudly on
 * a mismatch).
 *
 * Speculative decoding (ApusGspec): draft/verify with EXACT output
 * equivalence to non-speculative decoding. The accept rule: a draft token
 * is accepted iff it equals the MAIN MODEL'S OWN pick at that position —
 * argmax for greedy, the model's own apus_sample() draw for sampled. Every
 * emitted token is sampled from the main model's own logits row (produced
 * by a context that is bitwise equal to one-by-one decoding — the M8b
 * per-token-interleaved decode batch, c/gmodel.h apus_gmodel_decode_batch),
 * consuming exactly one RNG uniform per emitted token in position order;
 * drafts consume no RNG. The emitted stream is therefore bitwise identical
 * to non-speculative decoding for the same seed — greedy included, by
 * construction, not by tolerance. (Chosen over classical rejection
 * sampling: it needs no draft-probability evaluation and reproduces the
 * deterministic per-seed stream exactly.)
 *
 * Step shape (depth D = drafts chained per step; batch = D tokens):
 *   invariant: main state fed <= q-1; x_q held (already sampled from a
 *   valid main logits row); drafts d1..dD chained from the MTP state
 *   consistent through q-1 (true pairs only).
 *   1. snapshot main + MTP state (rollback below).
 *   2. verify batch: main decode-batch of [x_q, d2, ..., d_D] at
 *      q..q+D-1 (ONE batched call, the M8b interleave) -> rows R[j] =
 *      logits for q+j+1, hiddens H[j].
 *   3. walk: emit x_q; for j=1..D-1 sample x_{q+j} from R[j-1]; accept
 *      d_{j+1} iff d_{j+1} == x_{q+j}; stop at the first mismatch. Full
 *      match -> bonus x_{q+D} from R[D-1].
 *   4. state fixup: fed-true run is batch[0..matched]. Full match: keep
 *      the batch state. Partial: restore the snapshot and re-feed the true
 *      prefix in ONE decode-batch call (bitwise == one-by-one, so the
 *      verify-batch rows remain valid and are reused). MTP: restore its
 *      snapshot, replay the true pairs (H[j], batch[j]) j<=matched in one
 *      batched MTP forward (PREFILL ordering — the oracle's batched
 *      mtp_forward semantics), snapshot the clean state, then chain the
 *      next D drafts from the replay's last hidden.
 *
 * Rollback design: ApusGsnap copies pos + each KDA layer's conv + recurrent
 * state (~134 MiB at real scale); DSA layers roll back by REWINDING n only
 * — cache rows past n are dead capacity: the deterministic re-feed
 * rewrites them, and the indexer/attention never read beyond n. The MTP
 * state (DSA-kind) rolls back the same way (pos + n); the draft chain's
 * pollution of its caches is undone by the next snapshot restore. The M6
 * expert cache is never rolled back — cache content is numerics-neutral
 * (a resolve returns the same dequantized bytes regardless).
 *
 * Usage: #define APUS_GMTP_IMPLEMENTATION in exactly one TU (needs the
 * gmodel/gdsa/gmoe/bf16/sample implementations linked).
 */
#ifndef APUS_GMTP_H
#define APUS_GMTP_H

#include <stddef.h>
#include <stdint.h>

#include "gmodel.h"
#include "sample.h"

#ifdef __cplusplus
extern "C" {
#endif

/* hnorm input source selector (the M8a pin; see the file-top comment). */
#define APUS_GMTP_SRC_PRENORM_MEAN 0    /* HyperHead mean, bf16-rounded */
#define APUS_GMTP_SRC_POSTNORM     1    /* post-final-norm yn row */
#define APUS_GMTP_SRC_HC(j)        (2 + (j))    /* mHC stream slot j */
#ifndef APUS_GMTP_HNORM_SRC
/* PINNED post-gate7 (the M8a re-pin, 2026-09-05): postnorm + lag 1 — the
 * SGLang/vLLM reference semantics AND the engine-sweep winner (85.3%
 * accept, 2.53 tok/batch at spec-k 3, 1.47x faster than non-spec). */
#define APUS_GMTP_HNORM_SRC APUS_GMTP_SRC_POSTNORM
#endif
const char *apus_gmtp_hnorm_src_name(void);
/* (h, id) pairing lag (the M8a pin): 0 = (h_p, tok_p) predicts p+1 (the
 * parent-engine convention); 1 = (h_{p-1}, tok_p) predicts p+1 (the
 * DeepSeek-V3/SGLang EAGLE bookkeeping). -D-overridable for the pin
 * sweep; the m8g gate checks the compiled-in value against the fixture
 * manifest. */
#ifndef APUS_GMTP_PAIR_LAG
#define APUS_GMTP_PAIR_LAG 1            /* (h_{p-1}, tok_p) predicts p+1
                                           (PINNED post-gate7, M8a re-pin:
                                           the EAGLE bookkeeping) */
#endif

/* ---- MTP block (weights viewed from the model; the state is owned) ------*/

typedef struct {
    const ApusGmodel *m;        /* embed/lm_head shared */
    int layer;                  /* expert-store layer id (n_main + 0) */
    ApusGmodelMtpW w;
    size_t dim;
    int V;
} ApusGmtp;

/* Bind to the model's loaded MTP block. Returns 0 on success, -1 with err
 * when the model has no MTP block loaded (apus_gmodel_has_mtp == 0). */
int  apus_gmtp_bind(ApusGmtp *mt, const ApusGmodel *m, char *err,
                    size_t errcap);

typedef struct {
    size_t cap;
    size_t pos;                 /* tracks the main model's position */
    ApusGdsaState dsa;          /* the block's attention/indexer caches */
    ApusGdsaScratch dsa_sc;     /* P5: reusable FP8-dequant wbuf */
} ApusGmtpState;

/* kv_cap must cover the main model's cap PLUS the draft-chain overshoot
 * (depth - 1 polluted rows past the main position). */
ApusGmtpState *apus_gmtp_state_new(const ApusGmtp *mt, size_t kv_cap);
void apus_gmtp_state_free(ApusGmtpState *st);

/* The hnorm input for one position, from the model's M8b h-surface:
 * h_row [hc*dim] (last-layer post-block stream) + yn_row [dim]
 * (post-final-norm) -> out_row [dim], per APUS_GMTP_HNORM_SRC. */
void apus_gmtp_hnorm_input(const ApusGmtp *mt, const uint16_t *h_row,
                           const uint16_t *yn_row, uint16_t *out_row);

/* TEST-ONLY teacher-forcing (tests/m8g tolerance tier; NULL in
 * production): router_idx [s, topk], indexer_topk [s, width]. */
typedef struct {
    const int32_t *router_idx;
    const int32_t *indexer_topk;
} ApusGmtpForces;

/* MTP forward (oracle mtp_forward). ids [s], prev_h [s, dim] BF16 codes —
 * row t is the pair (prev_h[t], ids[t]) at consecutive positions (lag 0:
 * BOTH at the same position, predicting the next). Mutates mst (DSA caches
 * appended, pos bumped by s). The s > 1 form takes the PREFILL ordering
 * for the DSA sublayer (one-shot batched, the oracle's batched
 * mtp_forward); s == 1 is the chain step. Outputs: logits [s, V] BF16
 * codes, out_h [s, dim] codes (the post-shared_head.norm hidden — the
 * chaining input). Returns 0 on success, -1 past the state capacity. */
int apus_gmtp_forward(const ApusGmtp *mt, ApusGmtpState *mst,
                      const int32_t *ids, const uint16_t *prev_h, size_t s,
                      uint16_t *logits, uint16_t *out_h,
                      const ApusGmtpForces *forces);

/* ---- snapshots (rollback) ------------------------------------------------*/

/* Main-model snapshot: pos + per-KDA-layer conv/rec copies; DSA layers
 * record n only (the n-rewind — dead-capacity semantics, see the file-top
 * comment). */
typedef struct {
    size_t pos;
    int L;
    size_t conv_bytes, rec_bytes;       /* per KDA layer */
    uint16_t **kda_conv;                /* [L] NULL for DSA layers */
    float **kda_rec;
    size_t *dsa_n;                      /* [L] valid for DSA layers */
} ApusGsnap;

void apus_gsnap_alloc(ApusGsnap *sn, const ApusGmodel *m);
void apus_gsnap_free(ApusGsnap *sn);
void apus_gsnap_take(ApusGsnap *sn, ApusGmodelState *st);
void apus_gsnap_restore(const ApusGsnap *sn, ApusGmodelState *st);

/* MTP state snapshot: pos + the DSA cache count (n-rewind). */
typedef struct {
    size_t pos, n;
} ApusGmtpSnap;
void apus_gmtp_snap_take(ApusGmtpSnap *sn, const ApusGmtpState *st);
void apus_gmtp_snap_restore(const ApusGmtpSnap *sn, ApusGmtpState *st);

/* ---- speculative decode engine --------------------------------------------*/

typedef struct ApusGspec {
    ApusGmodel *m;
    ApusGmodelState *st;
    const ApusGmtp *mt;         /* NULL iff draft_override drives */
    ApusGmtpState *mst;
    int depth;                  /* D: drafts chained per step (>= 1) */
    float temp, top_p;
    ApusRng *rng;
    void *sample_scratch;
    /* Optional test hook replacing the MTP proposer: fill drafts[0..n-1]
     * (d1..dD candidates for positions q..q+n-1). When set, the MTP block
     * is never run. */
    void (*draft_override)(void *ctx, int64_t q, int32_t *drafts, int n);
    void *draft_ctx;
    /* stats */
    uint64_t emitted;           /* tokens emitted across all steps */
    uint64_t batches;           /* verify batches run */
    uint64_t batch_tokens;      /* tokens processed by verify batches */
    uint64_t refeed_tokens;     /* tokens re-fed after partial rejects */
    uint64_t offered;           /* speculative draft tokens checked (d2..dD) */
    uint64_t accepted;          /* ... accepted */
    uint64_t d1_offered, d1_hits;   /* d1 vs the held token (informational) */
    /* internals (opaque to callers) */
    size_t q;                   /* position of the held token */
    int32_t held;               /* x_q: sampled, not yet emitted */
    int32_t *drafts;            /* [depth] d1..dD */
    int32_t *batch;             /* [depth] */
    uint16_t *logits16;         /* [depth * V] verify-batch codes (also the
                                   MTP replay/chain logits scratch) */
    float *R;                   /* [depth * V] widened verify logits */
    uint16_t *H;                /* [depth * hc*dim] verify-batch streams */
    uint16_t *Yn;               /* [depth * dim] verify-batch post-norm rows */
    uint16_t *ph;               /* [depth * dim] replay pair hiddens */
    uint16_t *h_prev;           /* [dim] hnorm input of the last true
                                   position (lag-1 pairing only) */
    uint16_t *oh;               /* [depth * dim] MTP out_h scratch */
    uint16_t *hmtp_a, *hmtp_b;  /* [dim] chaining hidden (double-buffered) */
    float *mtp_logits;          /* [V] widened */
    ApusGsnap snap;             /* main state snapshot */
    ApusGmtpSnap mtp_snap;      /* MTP state snapshot */
    int V, dim, hc;
} ApusGspec;

void apus_gspec_init(ApusGspec *sp, ApusGmodel *m, ApusGmodelState *st,
                     const ApusGmtp *mt, ApusGmtpState *mst, int depth,
                     float temp, float top_p, ApusRng *rng,
                     void *sample_scratch);
void apus_gspec_free(ApusGspec *sp);

/* Prefill: main forward over ids[n] + MTP true-pair replay + first draft
 * chain; samples the first held token. Returns 0 on success. */
int  apus_gspec_prefill(ApusGspec *sp, const int32_t *ids, size_t n);

/* One speculative step. Appends emitted tokens to out (cap must be >=
 * depth + 1); returns the count emitted (>= 1, 0 at the state capacity
 * limit). The caller stops on EOS / max_tokens — the emitted stream is a
 * prefix of the non-speculative stream, so truncation is safe. */
int  apus_gspec_step(ApusGspec *sp, int *out, int cap);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GMTP_IMPLEMENTATION) && !defined(APUS_GMTP_IMPL_INCLUDED)
#define APUS_GMTP_IMPL_INCLUDED

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gcache.h"

/* ---- small helpers ---------------------------------------------------------*/

/* Plain residual add (oracle _B(res + x)): one f32 add of bf16-valued
 * operands (exactly numpy's f32 add), one RNE bf16 rounding. */
static void apus_gmtp_add(uint16_t *x, const uint16_t *sub, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = apus_bf16_bits(apus_bf16_f32(x[i]) + apus_bf16_f32(sub[i]));
}

static void apus_gmtp_widen(const uint16_t *c, float *f, size_t n) {
    for (size_t i = 0; i < n; i++) f[i] = apus_bf16_f32(c[i]);
}

const char *apus_gmtp_hnorm_src_name(void) {
#if APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_PRENORM_MEAN
    return "prenorm_mean";
#elif APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_POSTNORM
    return "postnorm";
#elif APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_HC(0)
    return "hc0";
#elif APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_HC(1)
    return "hc1";
#elif APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_HC(2)
    return "hc2";
#elif APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_HC(3)
    return "hc3";
#else
    return "unknown";
#endif
}

/* ---- MTP block ------------------------------------------------------------*/

int apus_gmtp_bind(ApusGmtp *mt, const ApusGmodel *m, char *err,
                   size_t errcap) {
    memset(mt, 0, sizeof *mt);
    mt->m = m;
    mt->layer = apus_gmodel_mtp_layer(m);
    if (mt->layer < 0 || apus_gmodel_mtp(m, &mt->w)) {
        if (err && errcap)
            snprintf(err, errcap, "gmtp: model has no MTP block loaded");
        return -1;
    }
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    mt->dim = (size_t)c->hidden_size;
    mt->V = c->vocab_size;
    return 0;
}

ApusGmtpState *apus_gmtp_state_new(const ApusGmtp *mt, size_t kv_cap) {
    ApusGmtpState *st = (ApusGmtpState *)calloc(1, sizeof *st);
    if (!st) return NULL;
    st->cap = kv_cap;
    const ApusGdsaW *w = mt->w.dsa;
    st->dsa.cap = kv_cap;
    st->dsa.n = 0;
    const size_t kn = w->H * kv_cap * w->qd;
    const size_t vn = w->H * kv_cap * w->vd;
    const size_t in = kv_cap * w->ID;
    st->dsa.k_cache = (uint16_t *)calloc(kn ? kn : 1, sizeof(uint16_t));
    st->dsa.v_cache = (uint16_t *)calloc(vn ? vn : 1, sizeof(uint16_t));
    st->dsa.idx_k = (uint16_t *)calloc(in ? in : 1, sizeof(uint16_t));
    st->dsa.idx_gate = (uint16_t *)calloc(in ? in : 1, sizeof(uint16_t));
    if (!st->dsa.k_cache || !st->dsa.v_cache || !st->dsa.idx_k
        || !st->dsa.idx_gate) {
        apus_gmtp_state_free(st);
        return NULL;
    }
    return st;
}

void apus_gmtp_state_free(ApusGmtpState *st) {
    if (!st) return;
    free(st->dsa.k_cache);
    free(st->dsa.v_cache);
    free(st->dsa.idx_k);
    free(st->dsa.idx_gate);
    apus_gdsa_scratch_free(&st->dsa_sc);
    free(st);
}

void apus_gmtp_hnorm_input(const ApusGmtp *mt, const uint16_t *h_row,
                           const uint16_t *yn_row, uint16_t *out_row) {
    const size_t dim = mt->dim;
#if APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_PRENORM_MEAN
    /* the HyperHead unweighted mean (bf16-rounded) — c/gmhc.h */
    apus_gmhc_head(h_row, out_row, dim,
                   (size_t)apus_gmodel_config(mt->m)->hc_mult);
    (void)yn_row;
#elif APUS_GMTP_HNORM_SRC == APUS_GMTP_SRC_POSTNORM
    memcpy(out_row, yn_row, dim * sizeof(uint16_t));
    (void)h_row;
#else
    memcpy(out_row, h_row + (APUS_GMTP_HNORM_SRC - 2) * dim,
           dim * sizeof(uint16_t));
    (void)yn_row;
#endif
}

/* The block's MoE sublayer (sparse, WITH shared expert): eager views or
 * the M6 cache wiring (pre-pass union -> demand hints -> just-in-time
 * resolves -> forward -> layer_end) — the c/gmodel.h pattern behind
 * apus_gmodel_expert(m, mtp_layer, e). Bitwise identical either way (the
 * m6g neutrality contract). */
static int apus_gmtp_moe(const ApusGmtp *mt, const uint16_t *x, size_t s,
                         uint16_t *out, const int32_t *forced_idx) {
    const ApusGmodelConfig *c = apus_gmodel_config(mt->m);
    const int E = c->n_routed_experts;
    const int topk = c->num_experts_per_tok;
    const size_t dim = mt->dim;
    ApusGmoeW gw;
    gw.router = *mt->w.router;
    gw.inter = (size_t)c->moe_intermediate_size;
    gw.sg = mt->w.shr_g;
    gw.su = mt->w.shr_u;
    gw.sd = mt->w.shr_d;
    gw.sinter = gw.inter * (size_t)c->n_shared_experts;
    gw.limit = c->swiglu_limit;
    const uint16_t **eg = (const uint16_t **)malloc((size_t)E
                                                    * sizeof *eg);
    const uint16_t **eu = (const uint16_t **)malloc((size_t)E
                                                    * sizeof *eu);
    const uint16_t **ed = (const uint16_t **)malloc((size_t)E
                                                    * sizeof *ed);
    gw.eg = eg;
    gw.eu = eu;
    gw.ed = ed;
    int rc = -1;
    ApusGcache *cache = apus_gmodel_cache(mt->m);
    ApusGmoeScratch sc;
    apus_gmoe_scratch_init(&sc, &gw, s);
    if (!cache) {
        for (int e = 0; e < E; e++) {
            const ApusGmodelExpertW *xw =
                apus_gmodel_expert(mt->m, mt->layer, e);
            if (!xw) goto out;
            eg[e] = xw->gate;
            eu[e] = xw->up;
            ed[e] = xw->down;
        }
        apus_gmoe_forward2(&gw, x, s, out, NULL, &sc, forced_idx);
        rc = 0;
    } else {
        /* tiered: resolve ONLY the routed union (the gmodel wiring) */
        float *scores = (float *)malloc((size_t)E * sizeof(float));
        float *biased = (float *)malloc((size_t)E * sizeof(float));
        float *wgt = (float *)malloc((size_t)topk * sizeof(float));
        int32_t *sel = (int32_t *)malloc(s * (size_t)topk * sizeof(int32_t));
        uint8_t *mark = (uint8_t *)calloc((size_t)E, 1);
        if (!scores || !biased || !wgt || !sel || !mark) {
            free(scores); free(biased); free(wgt); free(sel); free(mark);
            goto out;
        }
        for (size_t t = 0; t < s; t++) {
            int32_t *it = sel + t * (size_t)topk;
            if (forced_idx)
                memcpy(it, forced_idx + t * (size_t)topk,
                       (size_t)topk * sizeof(int32_t));
            else
                apus_gmoe_router(&gw.router, x + t * dim, scores, it, wgt,
                                 biased);
            for (int j = 0; j < topk; j++)
                mark[it[j]] = 1;
        }
        for (int e = 0; e < E; e++) {
            eg[e] = eu[e] = ed[e] = NULL;
            if (mark[e])
                apus_gcache_hint_demand(cache, mt->layer, e);
        }
        for (int e = 0; e < E; e++) {
            if (!mark[e]) continue;
            const ApusGmodelExpertW *xw =
                apus_gmodel_expert(mt->m, mt->layer, e);
            if (!xw) {
                free(scores); free(biased); free(wgt); free(sel); free(mark);
                goto out;
            }
            eg[e] = xw->gate;
            eu[e] = xw->up;
            ed[e] = xw->down;
        }
        apus_gmoe_forward2(&gw, x, s, out, NULL, &sc, forced_idx);
        apus_gcache_layer_end(cache, mt->layer);
        rc = 0;
        free(scores); free(biased); free(wgt); free(sel); free(mark);
    }
out:
    apus_gmoe_scratch_free(&sc);
    free(eg);
    free(eu);
    free(ed);
    return rc;
}

int apus_gmtp_forward(const ApusGmtp *mt, ApusGmtpState *mst,
                      const int32_t *ids, const uint16_t *prev_h, size_t s,
                      uint16_t *logits, uint16_t *out_h,
                      const ApusGmtpForces *forces) {
    const ApusGmodelConfig *c = apus_gmodel_config(mt->m);
    const size_t dim = mt->dim;
    const size_t V = (size_t)mt->V;
    const float eps = c->rms_norm_eps;
    if (mst->pos + s > mst->cap)
        return -1;
    const uint16_t *embed = apus_gmodel_embed(mt->m);
    const uint16_t *head = apus_gmodel_head(mt->m);

    uint16_t *eh;                           /* calloc: the write loop below
        covers the buffer fully, but gcc 13's -Wmaybe-uninitialized cannot
        prove it through the extern rmsnorm calls (false positive on
        Linux; calloc makes the contents provably defined) */
    {
        const size_t ehn = s * 2 * dim;
        eh = (uint16_t *)calloc(ehn ? ehn : 1, sizeof(uint16_t));
    }
    uint16_t *x = (uint16_t *)malloc(s * dim * sizeof(uint16_t));
    uint16_t *xn = (uint16_t *)malloc(s * dim * sizeof(uint16_t));
    uint16_t *sub = (uint16_t *)malloc(s * dim * sizeof(uint16_t));
    float *xf = (float *)malloc(s * 2 * dim * sizeof(float));
    if (!eh || !x || !xn || !sub || !xf) {
        free(eh); free(x); free(xn); free(sub); free(xf);
        return -1;
    }

    /* eh = cat([enorm(embed(tok)), hnorm(prev_h)]) — enorm FIRST */
    for (size_t t = 0; t < s; t++) {
        apus_gdsa_rmsnorm(embed + (size_t)ids[t] * dim, mt->w.enorm,
                          eh + t * 2 * dim, dim, eps);
        apus_gdsa_rmsnorm(prev_h + t * dim, mt->w.hnorm,
                          eh + t * 2 * dim + dim, dim, eps);
    }
    /* x = bf16_linear(eh, eh_proj) (BF16 [dim, 2*dim]) */
    apus_bf16_gemm_mt(mt->w.eh_proj, eh, xf, x, s, dim, 2 * dim);

    /* plain-residual DSA + MoE block (no mHC) */
    for (size_t t = 0; t < s; t++)
        apus_gdsa_rmsnorm(x + t * dim, mt->w.input_norm, xn + t * dim, dim,
                          eps);
    apus_gdsa_forward3(mt->w.dsa, xn, s, &mst->dsa, sub, NULL,
                       forces ? forces->indexer_topk : NULL, &mst->dsa_sc);
    apus_gmtp_add(x, sub, s * dim);                 /* residual 1 */
    for (size_t t = 0; t < s; t++)
        apus_gdsa_rmsnorm(x + t * dim, mt->w.post_norm, xn + t * dim, dim,
                          eps);
    if (apus_gmtp_moe(mt, xn, s, sub,
                      forces ? forces->router_idx : NULL)) {
        free(eh); free(x); free(xn); free(sub); free(xf);
        return -1;
    }
    apus_gmtp_add(x, sub, s * dim);                 /* residual 2 */
    /* fused add+norm: out = rms_norm(x, shared_head.norm) */
    for (size_t t = 0; t < s; t++)
        apus_gdsa_rmsnorm(x + t * dim, mt->w.shared_norm, out_h + t * dim,
                          dim, eps);
    /* shared lm_head */
    apus_bf16_gemm_mt(head, out_h, xf, logits, s, V, dim);
    mst->pos += s;

    free(eh);
    free(x);
    free(xn);
    free(sub);
    free(xf);
    return 0;
}

/* ---- snapshots --------------------------------------------------------------*/

void apus_gsnap_alloc(ApusGsnap *sn, const ApusGmodel *m) {
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    const int L = c->num_hidden_layers;
    const size_t qkv = (size_t)c->linear_num_heads
                     * (size_t)c->linear_head_dim;
    const size_t ck = (size_t)c->linear_conv_kernel_dim;
    memset(sn, 0, sizeof *sn);
    sn->L = L;
    sn->conv_bytes = 3 * qkv * (ck - 1) * sizeof(uint16_t);
    sn->rec_bytes = qkv * (size_t)c->linear_head_dim * sizeof(float);
    sn->kda_conv = (uint16_t **)calloc((size_t)L, sizeof *sn->kda_conv);
    sn->kda_rec = (float **)calloc((size_t)L, sizeof *sn->kda_rec);
    sn->dsa_n = (size_t *)calloc((size_t)L, sizeof *sn->dsa_n);
    for (int l = 0; l < L; l++) {
        if (c->layer_is_dsa[l]) continue;
        sn->kda_conv[l] = (uint16_t *)malloc(sn->conv_bytes);
        sn->kda_rec[l] = (float *)malloc(sn->rec_bytes);
    }
}

void apus_gsnap_free(ApusGsnap *sn) {
    if (!sn->kda_conv) return;
    for (int l = 0; l < sn->L; l++) {
        free(sn->kda_conv[l]);
        free(sn->kda_rec[l]);
    }
    free(sn->kda_conv);
    free(sn->kda_rec);
    free(sn->dsa_n);
    memset(sn, 0, sizeof *sn);
}

void apus_gsnap_take(ApusGsnap *sn, ApusGmodelState *st) {
    sn->pos = apus_gmodel_pos(st);
    for (int l = 0; l < sn->L; l++) {
        ApusGkdaState *ks = apus_gmodel_kda_state_mut(st, l);
        if (ks) {
            memcpy(sn->kda_conv[l], ks->conv_state, sn->conv_bytes);
            memcpy(sn->kda_rec[l], ks->rec_state, sn->rec_bytes);
        } else {
            ApusGdsaState *ds = apus_gmodel_dsa_state_mut(st, l);
            sn->dsa_n[l] = ds->n;
        }
    }
}

void apus_gsnap_restore(const ApusGsnap *sn, ApusGmodelState *st) {
    apus_gmodel_set_pos(st, sn->pos);
    for (int l = 0; l < sn->L; l++) {
        ApusGkdaState *ks = apus_gmodel_kda_state_mut(st, l);
        if (ks) {
            memcpy(ks->conv_state, sn->kda_conv[l], sn->conv_bytes);
            memcpy(ks->rec_state, sn->kda_rec[l], sn->rec_bytes);
        } else {
            ApusGdsaState *ds = apus_gmodel_dsa_state_mut(st, l);
            ds->n = sn->dsa_n[l];   /* rejected-tail slots: dead capacity */
        }
    }
}

void apus_gmtp_snap_take(ApusGmtpSnap *sn, const ApusGmtpState *st) {
    sn->pos = st->pos;
    sn->n = st->dsa.n;
}

void apus_gmtp_snap_restore(const ApusGmtpSnap *sn, ApusGmtpState *st) {
    st->pos = sn->pos;
    st->dsa.n = sn->n;
}

/* ---- speculative decode engine -------------------------------------------*/

void apus_gspec_init(ApusGspec *sp, ApusGmodel *m, ApusGmodelState *st,
                     const ApusGmtp *mt, ApusGmtpState *mst, int depth,
                     float temp, float top_p, ApusRng *rng,
                     void *sample_scratch) {
    memset(sp, 0, sizeof *sp);
    sp->m = m;
    sp->st = st;
    sp->mt = mt;
    sp->mst = mst;
    sp->depth = depth > 0 ? depth : 1;
    sp->temp = temp;
    sp->top_p = top_p;
    sp->rng = rng;
    sp->sample_scratch = sample_scratch;
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    sp->V = c->vocab_size;
    sp->dim = c->hidden_size;
    sp->hc = c->hc_mult;
    const int D = sp->depth;
    const size_t V = (size_t)sp->V, dim = (size_t)sp->dim;
    const size_t hcd = (size_t)sp->hc * dim;
    sp->drafts = (int32_t *)malloc((size_t)D * sizeof(int32_t));
    sp->batch = (int32_t *)malloc((size_t)D * sizeof(int32_t));
    sp->logits16 = (uint16_t *)malloc((size_t)D * V * sizeof(uint16_t));
    sp->R = (float *)malloc((size_t)D * V * sizeof(float));
    sp->H = (uint16_t *)malloc((size_t)D * hcd * sizeof(uint16_t));
    sp->Yn = (uint16_t *)malloc((size_t)D * dim * sizeof(uint16_t));
    sp->ph = (uint16_t *)malloc((size_t)D * dim * sizeof(uint16_t));
    sp->h_prev = (uint16_t *)malloc(dim * sizeof(uint16_t));
    sp->oh = (uint16_t *)malloc((size_t)D * dim * sizeof(uint16_t));
    sp->hmtp_a = (uint16_t *)malloc(dim * sizeof(uint16_t));
    sp->hmtp_b = (uint16_t *)malloc(dim * sizeof(uint16_t));
    sp->mtp_logits = (float *)malloc(V * sizeof(float));
    apus_gsnap_alloc(&sp->snap, m);
}

void apus_gspec_free(ApusGspec *sp) {
    free(sp->drafts);
    free(sp->batch);
    free(sp->logits16);
    free(sp->R);
    free(sp->H);
    free(sp->Yn);
    free(sp->ph);
    free(sp->h_prev);
    free(sp->oh);
    free(sp->hmtp_a);
    free(sp->hmtp_b);
    free(sp->mtp_logits);
    apus_gsnap_free(&sp->snap);
    memset(sp, 0, sizeof *sp);
}

/* Draft chain (parent apus_spec_chain shape): drafts[0] = argmax of the
 * current MTP logits; d_{i+1} = MTP(out_h_i, d_i) — the MTP block's own
 * post-shared_head.norm hidden takes the role of the main hidden when
 * chaining. Pollutes the MTP attention state — cleaned by the next
 * snapshot restore. A chain step past the MTP state capacity (-1) fills
 * the remaining drafts with token 0: dead drafts, the accept rule simply
 * rejects them (never emitted — the invariant is untouched). */
static void apus_gspec_chain(ApusGspec *sp) {
    const size_t V = (size_t)sp->V, dim = (size_t)sp->dim;
    sp->drafts[0] = apus_sample_argmax(sp->mtp_logits, V);
    uint16_t *cur = sp->hmtp_a, *nxt = sp->hmtp_b;
    for (int i = 1; i < sp->depth; i++) {
        if (apus_gmtp_forward(sp->mt, sp->mst, &sp->drafts[i - 1], cur, 1,
                              sp->logits16, nxt, NULL)) {
            for (int j = i; j < sp->depth; j++)
                sp->drafts[j] = 0;
            break;
        }
        apus_gmtp_widen(sp->logits16, sp->mtp_logits, V);
        sp->drafts[i] = apus_sample_argmax(sp->mtp_logits, V);
        uint16_t *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }
    if (cur != sp->hmtp_a)
        memcpy(sp->hmtp_a, cur, dim * sizeof(uint16_t));
}

int apus_gspec_prefill(ApusGspec *sp, const int32_t *ids, size_t n) {
    const size_t V = (size_t)sp->V, dim = (size_t)sp->dim;
    const size_t hcd = (size_t)sp->hc * dim;
    uint16_t *h_all = (uint16_t *)malloc(n * hcd * sizeof(uint16_t));
    uint16_t *yn_all = (uint16_t *)malloc(n * dim * sizeof(uint16_t));
    uint16_t *l16 = (uint16_t *)malloc(n * V * sizeof(uint16_t));
    if (!h_all || !yn_all || !l16) {
        free(h_all); free(yn_all); free(l16);
        return -1;
    }
    if (apus_gmodel_prefill_h(sp->m, sp->st, ids, n, l16, NULL,
                              h_all, yn_all)) {
        free(h_all); free(yn_all); free(l16);
        return -1;
    }
    /* first held token: exactly what non-spec sampling draws first */
    apus_gmtp_widen(l16 + (n - 1) * V, sp->mtp_logits, V);
    sp->held = apus_sample(sp->mtp_logits, V, sp->temp, sp->top_p,
                           sp->rng, sp->sample_scratch);
    sp->q = n;
    if (sp->draft_override) {
        sp->draft_override(sp->draft_ctx, (int64_t)sp->q, sp->drafts,
                           sp->depth);
    } else {
        /* MTP true-pair replay over the prompt (batched): builds the
         * draft head's caches from the true (h, id) pairs and yields the
         * first draft; then chain the remaining depth-1. Pairing per
         * APUS_GMTP_PAIR_LAG: lag 0 = (h_t, tok_t) at every position;
         * lag 1 (EAGLE bookkeeping) = (h_{t-1}, tok_t) — ids[1..n-1]
         * against h[0..n-2]; position 0 has no pair. */
        uint16_t *ph = (uint16_t *)malloc(n * dim * sizeof(uint16_t));
        uint16_t *oh = (uint16_t *)malloc(n * dim * sizeof(uint16_t));
        if (!ph || !oh) {
            free(ph); free(oh);
            free(h_all); free(yn_all); free(l16);
            return -1;
        }
        for (size_t t = 0; t < n; t++)
            apus_gmtp_hnorm_input(sp->mt, h_all + t * hcd,
                                  yn_all + t * dim, ph + t * dim);
        /* l16 [n, V] doubles as the replay logits output */
        int rc;
        size_t last;
#if APUS_GMTP_PAIR_LAG == 0
        rc = apus_gmtp_forward(sp->mt, sp->mst, ids, ph, n, l16, oh,
                               NULL);
        last = n - 1;
#else
        if (n < 2) {
            rc = -1;    /* no pair exists below 2 prompt tokens */
            last = 0;
        } else {
            rc = apus_gmtp_forward(sp->mt, sp->mst, ids + 1, ph, n - 1,
                                   l16, oh, NULL);
            last = n - 2;
        }
        /* the hidden at position n-1 pairs with the next true token */
        apus_gmtp_hnorm_input(sp->mt, h_all + (n - 1) * hcd,
                              yn_all + (n - 1) * dim, sp->h_prev);
#endif
        if (rc == 0) {
            apus_gmtp_widen(l16 + last * V, sp->mtp_logits, V);
            memcpy(sp->hmtp_a, oh + last * dim, dim * sizeof(uint16_t));
            apus_gmtp_snap_take(&sp->mtp_snap, sp->mst);
            apus_gspec_chain(sp);
        } else {
            /* capacity stop: dead drafts (never accepted) */
            for (int i = 0; i < sp->depth; i++) sp->drafts[i] = 0;
        }
        free(ph);
        free(oh);
    }
    free(h_all);
    free(yn_all);
    free(l16);
    return 0;
}

int apus_gspec_step(ApusGspec *sp, int *out, int cap) {
    const int D = sp->depth;
    const size_t V = (size_t)sp->V, dim = (size_t)sp->dim;
    const size_t hcd = (size_t)sp->hc * dim;
    if (cap < D + 1) return 0;
    const size_t maxp = apus_gmodel_state_cap(sp->st);
    int D_eff = D;
    if (sp->q + (size_t)D_eff > maxp) D_eff = (int)(maxp - sp->q);
    if (D_eff <= 0) return 0;

    sp->d1_offered++;
    if (sp->drafts[0] == sp->held) sp->d1_hits++;

    /* verify batch: [x_q, d2, ..., d_{D_eff}] at positions q..q+D_eff-1,
     * ONE batched decode (the M8b per-token interleave: row j is bitwise
     * the one-by-one decode of the same context) */
    apus_gsnap_take(&sp->snap, sp->st);
    sp->batch[0] = sp->held;
    for (int j = 1; j < D_eff; j++) sp->batch[j] = sp->drafts[j];
    if (apus_gmodel_decode_batch(sp->m, sp->st, sp->batch, (size_t)D_eff,
                                 sp->logits16, NULL, sp->H, sp->Yn))
        return 0;
    sp->batches++;
    sp->batch_tokens += (uint64_t)D_eff;

    /* accept walk: every sampled token is the main model's own draw from
     * its own logits row — one RNG uniform per token, in position order.
     * Accepted drafts are emitted at once; the token that ends the walk
     * (the replacement after a mismatch, or the bonus after a full match)
     * becomes the next HELD token: sampled now, emitted at the start of
     * the next step (it is a true token either way). */
    int ne = 0, matched = 0, last = -1;
    out[ne++] = (int)sp->held;
    for (int j = 1; j < D_eff; j++) {
        apus_gmtp_widen(sp->logits16 + (size_t)(j - 1) * V, sp->R
                        + (size_t)(j - 1) * V, V);
        int x = apus_sample(sp->R + (size_t)(j - 1) * V, V, sp->temp,
                            sp->top_p, sp->rng, sp->sample_scratch);
        sp->offered++;
        if ((int)sp->drafts[j] == x) {
            matched++;
            sp->accepted++;
            out[ne++] = x;
        } else {
            last = x;
            break;
        }
    }
    if (last < 0) { /* full match: bonus from R[D_eff-1] (valid — the
                       whole batch proved true) */
        apus_gmtp_widen(sp->logits16 + (size_t)(D_eff - 1) * V,
                        sp->R + (size_t)(D_eff - 1) * V, V);
        last = apus_sample(sp->R + (size_t)(D_eff - 1) * V, V, sp->temp,
                           sp->top_p, sp->rng, sp->sample_scratch);
    }
    sp->emitted += (uint64_t)ne;

    /* state fixup: fed-true run = batch[0..matched] */
    if (matched < D_eff - 1) {
        /* partial reject: rollback = snapshot restore (KDA memcpy + DSA
         * n-rewind), then re-feed the true prefix in one batched decode
         * (bitwise the one-by-one state by the interleave) */
        apus_gsnap_restore(&sp->snap, sp->st);
        if (apus_gmodel_decode_batch(sp->m, sp->st, sp->batch,
                                     (size_t)(matched + 1), sp->logits16,
                                     NULL, NULL, NULL))
            return 0;
        sp->refeed_tokens += (uint64_t)(matched + 1);
    }

    /* next drafts */
    sp->held = last;
    sp->q += (size_t)(matched + 1);
    if (sp->mt && !sp->draft_override) {
        /* MTP: restore the clean (true-pairs-only) state, replay the newly
         * accepted true pairs — batched — then snapshot the clean state
         * and chain the next depth drafts. Pairing per APUS_GMTP_PAIR_LAG:
         * lag 0 = (H[j], batch[j]) j<=matched; lag 1 (EAGLE) = the held
         * token pairs with the carried h_prev, batch[j] with H[j-1]. */
#if APUS_GMTP_PAIR_LAG == 0
        for (int j = 0; j <= matched; j++)
            apus_gmtp_hnorm_input(sp->mt, sp->H + (size_t)j * hcd,
                                  sp->Yn + (size_t)j * dim,
                                  sp->ph + (size_t)j * dim);
#else
        memcpy(sp->ph, sp->h_prev, dim * sizeof(uint16_t));
        for (int j = 1; j <= matched; j++)
            apus_gmtp_hnorm_input(sp->mt, sp->H + (size_t)(j - 1) * hcd,
                                  sp->Yn + (size_t)(j - 1) * dim,
                                  sp->ph + (size_t)j * dim);
#endif
        apus_gmtp_snap_restore(&sp->mtp_snap, sp->mst);
        if (apus_gmtp_forward(sp->mt, sp->mst, sp->batch, sp->ph,
                              (size_t)(matched + 1), sp->logits16, sp->oh,
                              NULL)) {
            /* capacity stop: dead drafts (never accepted) */
            for (int i = 0; i < sp->depth; i++) sp->drafts[i] = 0;
            return ne;
        }
        apus_gmtp_widen(sp->logits16 + (size_t)matched * V, sp->mtp_logits,
                        V);
        memcpy(sp->hmtp_a, sp->oh + (size_t)matched * dim,
               dim * sizeof(uint16_t));
#if APUS_GMTP_PAIR_LAG != 0
        /* carry the hnorm input of the new last true position */
        apus_gmtp_hnorm_input(sp->mt, sp->H + (size_t)matched * hcd,
                              sp->Yn + (size_t)matched * dim, sp->h_prev);
#endif
        apus_gmtp_snap_take(&sp->mtp_snap, sp->mst);
        apus_gspec_chain(sp);
    } else if (sp->draft_override) {
        sp->draft_override(sp->draft_ctx, (int64_t)sp->q, sp->drafts,
                           sp->depth);
    }
    return ne;
}

#endif /* APUS_GMTP_IMPLEMENTATION */
#endif /* APUS_GMTP_H */
