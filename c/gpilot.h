/*
 * c/gpilot.h — GLM router-lookahead prefetch ("pilot", M6): predict the
 * next sparse layer's routed experts from the current layer's
 * post-attention hidden state and warm the M6 expert cache
 * (c/gcache.h) before the demand resolve arrives. The V4 c/pilot.h design
 * (dL=1 router lookahead, speculative hint delivery, recall accounting)
 * ported to the GLM router/mHC semantics — NEW CODE; c/pilot.h is
 * untouched (the retained V4 battery owns it). C11, libc only.
 *
 * Prediction math (reuse of the m4a/m4b gated pieces, never duplicated):
 *   - router input  = the target layer's ffn mHC hc_pre collapse
 *                     (apus_gmhc_maps + apus_gmhc_collapse, c/gmhc.h) of the
 *                     SOURCE layer's post-attention stream h (BF16 codes),
 *                     then the ffn weighted RMSNorm (apus_gdsa_rmsnorm,
 *                     c/gdsa.h — the round-then-x-weight GLM order) — the
 *                     exact code path c/gmodel.h feeds the real router,
 *                     applied to the TARGET layer's own weights on the
 *                     source layer's state (a dL=1 heuristic; recall is
 *                     measured, never assumed);
 *   - scores        = apus_gmoe_router (c/gmoe.h) — the same sigmoid +
 *                     selection-only-bias code path as the real gate;
 *   - top-N         = apus_gmoe_topk_stable (the M0 stable-ties pin).
 * Depth dL = 1 only: at layer L's post-attention hook the pilot predicts
 * layer L+1 (skipped when L+1 has no router — the dense layers 0..2).
 *
 * Delivery: hooks fire on the COMPUTE thread (c/gmodel.h ApusGmodelHooks —
 * no pilot thread, no ring; the V4 SPSC ring exists to decouple hint
 * submission cost, which is negligible here: apus_gcache_hint is a locked
 * dedup + queue push). Overlap comes from the gcache I/O pool: hints
 * issued at layer L's post-attention stream layer L+1's slabs while layer
 * L's MoE computes. Numerics are never touched: the pilot only reads
 * hidden states and issues cache hints (tests/m6g gates digest equality
 * pilot-ON vs pilot-OFF).
 *
 * Decode (s == 1): predict the target's top-pilot_k, record the pending
 * set, hint each predicted expert (SPECULATIVE class — demand loads from
 * the MoE wiring always overtake them). Recall accounting: the wiring's
 * routed hook reports the actual top-k; actual_hits / actual_experts is
 * the pilot recall, readable from apus_gpilot_stats alone.
 *
 * Prefill (s > 1, prefill_k > 0): the SAME prediction per token over all
 * s tokens (the maps are per-token independent, so per-token predictions
 * need no batched-matmul twin), unioned through a seen bitset; one
 * speculative hint per unique expert. Lead time: the target's slabs stream
 * while the source layer's attention + MoE run.
 *
 * Usage: #define APUS_GPILOT_IMPLEMENTATION in exactly one TU (needs the
 * gmodel/gcache/gmoe/gdsa/gmhc/bf16 implementations linked).
 */
#ifndef APUS_GPILOT_H
#define APUS_GPILOT_H

#include <stddef.h>
#include <stdint.h>

#include "gmodel.h"     /* ApusGmodel, ApusGmodelHooks, ApusGmodelPilotView */
#include "gcache.h"     /* ApusGcache, apus_gcache_hint */

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only view of one sparse layer's router + ffn-norm weights (owned by
 * the ApusGmodel, not by the pilot). */
typedef struct {
    const float    *hc_fn;      /* [(2+hc)*hc, hc*dim] f32 (widened) */
    const float    *hc_base;    /* [(2+hc)*hc] f32 */
    const float    *hc_scale;   /* [3] f32 */
    const uint16_t *post_norm;  /* [dim] BF16 codes */
    const uint16_t *gate_w;     /* [E, dim] BF16 codes */
    const float    *gate_bias;  /* [E] f32 */
} ApusGpilotRouter;

typedef struct {
    ApusGcache *cache;          /* hint target; NULL = predict-only */
    int         n_layers;       /* required */
    int         n_experts;      /* required (E) */
    int         topk;           /* required: router top-k */
    size_t      dim;
    int         hc_mult, sinkhorn_iters;
    float       norm_eps, hc_eps, route_scale;
    int         enabled;        /* 0 = attach but never predict/prefetch */
    int         pilot_k;        /* decode top-N cap; 0 = APUS_GPILOT_K env
                                   (default 12); <0 = no decode lookahead */
    int         prefill_k;      /* prefill union top-N; 0 = env
                                   APUS_GPILOT_PREFILL_K (default 12);
                                   <0 = off */
} ApusGpilotCfg;

typedef struct {
    uint64_t predictions;         /* decode top-N sets computed */
    uint64_t hints_issued;        /* decode: apus_gcache_hint calls */
    uint64_t prefill_predictions; /* prefill: tokens scored */
    uint64_t prefill_hints;       /* prefill: unique union experts hinted */
    uint64_t actual_experts;      /* routed experts observed (s == 1 layers
                                     with a pending prediction) */
    uint64_t actual_hits;         /* of those, present in the predicted set */
} ApusGpilotStats;

typedef struct ApusGpilot ApusGpilot;

ApusGpilot *apus_gpilot_create(const ApusGpilotCfg *cfg);
void        apus_gpilot_destroy(ApusGpilot *p);

/* Register layer `layer`'s router view (sparse layers only). */
void apus_gpilot_attach_router(ApusGpilot *p, int layer,
                               const ApusGpilotRouter *r);

/* Wire the pilot into a loaded model: registers every sparse layer's
 * router view (apus_gmodel_pilot_view) and installs the model hooks. */
void apus_gpilot_attach(ApusGpilot *p, ApusGmodel *m);

/* Fill *hooks with this pilot's callbacks (for manual wiring). */
void apus_gpilot_hooks(ApusGpilot *p, ApusGmodelHooks *hooks);

/* Pure prediction (no hints, no stats): predicted top-n expert ids
 * (stable biased-score order) for layer `target` from one token's
 * post-attention stream h [hc*dim] BF16 codes. Returns -1 if the target
 * has no router view. */
int  apus_gpilot_predict(const ApusGpilot *p, int target,
                         const uint16_t *h, int32_t *idx, int n);

void apus_gpilot_stats(ApusGpilot *p, ApusGpilotStats *out);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GPILOT_IMPLEMENTATION) && !defined(APUS_GPILOT_IMPL_INCLUDED)
#define APUS_GPILOT_IMPL_INCLUDED

#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "bf16.h"
#include "gmhc.h"
#include "gmoe.h"
#include "gdsa.h"

#define APUS_GPILOT_MAX_K 64    /* cap for a prediction set */

struct ApusGpilot {
    ApusGpilotCfg     cfg;
    ApusGpilotRouter *routers;      /* [n_layers] */
    /* prediction scratch (compute thread only) */
    float    *x4f;                  /* [hc*dim] */
    float    *pre, *post, *comb, *mixes;
    uint16_t *x;                    /* [dim] router input */
    float    *scores, *biased;      /* [E] */
    float    *wgt;                  /* [topk] */
    int32_t  *tidx;                 /* [topk] */
    int32_t  *tset;                 /* [APUS_GPILOT_MAX_K] */
    uint64_t *seen;                 /* [(E+63)/64] prefill union bitset */
    /* pending predictions for recall accounting (compute thread only) */
    int32_t  *pending;              /* [n_layers][pilot_k] */
    uint8_t  *pending_ok;           /* [n_layers] */
    int64_t  *pending_pos;          /* [n_layers] */
    ApusGpilotStats stats;
};

/* --- prediction -----------------------------------------------------------*/

int apus_gpilot_predict(const ApusGpilot *p0, int target,
                        const uint16_t *h, int32_t *idx, int n) {
    ApusGpilot *p = (ApusGpilot *)p0;   /* scratch mutation */
    if (!p || target < 0 || target >= p->cfg.n_layers || n <= 0) return -1;
    const ApusGpilotRouter *r = &p->routers[target];
    if (!r->gate_w) return -1;
    size_t hc = (size_t)p->cfg.hc_mult, dim = p->cfg.dim;
    int E = p->cfg.n_experts;
    if (n > E) n = E;
    /* identical to the c/gmodel.h ffn-site input path (apus_gm_hc_pre +
     * the post_norm rmsnorm): widen, maps, collapse (BF16 codes), weighted
     * RMSNorm (round-then-x-weight) */
    for (size_t i = 0; i < hc * dim; i++)
        p->x4f[i] = apus_bf16_f32(h[i]);
    apus_gmhc_maps(p->x4f, dim, hc, r->hc_fn, r->hc_scale, r->hc_base,
                   p->cfg.norm_eps, p->cfg.hc_eps, p->cfg.sinkhorn_iters,
                   p->pre, p->post, p->comb, p->mixes);
    apus_gmhc_collapse(p->x4f, p->pre, p->x, dim, hc);
    apus_gdsa_rmsnorm(p->x, r->post_norm, p->x, dim, p->cfg.norm_eps);
    /* the real router's own scoring path (c/gmoe.h) + the same stable
     * top-k the gate uses */
    ApusGmoeRouterW rw;
    rw.E = E;
    rw.topk = p->cfg.topk;
    rw.dim = dim;
    rw.route_scale = p->cfg.route_scale;
    rw.gate_w = r->gate_w;
    rw.gate_bias = r->gate_bias;
    apus_gmoe_router(&rw, p->x, p->scores, p->tidx, p->wgt, p->biased);
    apus_gmoe_topk_stable(p->biased, E, n, idx);
    return 0;
}

/* --- compute-thread hooks ----------------------------------------------------*/

static void apus_gpilot_on_post_attn(void *ctx, int layer,
                                     const uint16_t *h, size_t s,
                                     size_t pos0) {
    ApusGpilot *p = ctx;
    if (!p->cfg.enabled) return;
    int target = layer + 1;
    if (target >= p->cfg.n_layers || !p->routers[target].gate_w) return;
    size_t hcd = (size_t)p->cfg.hc_mult * p->cfg.dim;
    if (s == 1) {
        int k = p->cfg.pilot_k;
        if (k <= 0) return;
        if (apus_gpilot_predict(p, target, h, p->tset, k)) return;
        p->stats.predictions++;
        memcpy(p->pending + (size_t)target * k, p->tset,
               (size_t)k * sizeof(int32_t));
        p->pending_ok[target] = 1;
        p->pending_pos[target] = (int64_t)pos0;
        if (p->cfg.cache)
            for (int j = 0; j < k; j++) {
                apus_gcache_hint(p->cfg.cache, target, p->tset[j]);
                p->stats.hints_issued++;
            }
        return;
    }
    /* prefill union lookahead: per-token predictions (the maps are
     * per-token independent), unioned through the seen bitset */
    int k = p->cfg.prefill_k;
    if (k <= 0) return;
    int E = p->cfg.n_experts;
    size_t nw = ((size_t)E + 63) / 64;
    memset(p->seen, 0, nw * sizeof(uint64_t));
    for (size_t t = 0; t < s; t++) {
        if (apus_gpilot_predict(p, target, h + t * hcd, p->tset, k))
            continue;
        p->stats.prefill_predictions++;
        for (int j = 0; j < k; j++)
            p->seen[p->tset[j] >> 6] |= 1ull << (p->tset[j] & 63);
    }
    if (p->cfg.cache)
        for (int e = 0; e < E; e++)
            if (p->seen[e >> 6] & (1ull << (e & 63))) {
                apus_gcache_hint(p->cfg.cache, target, e);
                p->stats.prefill_hints++;
            }
}

static void apus_gpilot_on_routed(void *ctx, int layer, const int32_t *idx,
                                  size_t s, size_t pos0) {
    ApusGpilot *p = ctx;
    if (s != 1 || layer < 0 || layer >= p->cfg.n_layers) return;
    if (!p->pending_ok[layer]
        || p->pending_pos[layer] != (int64_t)pos0) return;
    int k = p->cfg.pilot_k, topk = p->cfg.topk;
    const int32_t *pd = p->pending + (size_t)layer * k;
    for (int j = 0; j < topk; j++) {
        p->stats.actual_experts++;
        for (int q = 0; q < k; q++)
            if (pd[q] == idx[j]) {
                p->stats.actual_hits++;
                break;
            }
    }
}

void apus_gpilot_hooks(ApusGpilot *p, ApusGmodelHooks *hooks) {
    hooks->ctx = p;
    hooks->post_attn = apus_gpilot_on_post_attn;
    hooks->routed = apus_gpilot_on_routed;
}

/* --- lifecycle ------------------------------------------------------------------*/

ApusGpilot *apus_gpilot_create(const ApusGpilotCfg *cfg) {
    if (!cfg || cfg->n_layers <= 0 || cfg->n_experts <= 0 || cfg->topk <= 0)
        return NULL;
    ApusGpilot *p = (ApusGpilot *)calloc(1, sizeof *p);
    p->cfg = *cfg;
    if (p->cfg.pilot_k == 0)
        p->cfg.pilot_k = apus_env_int("APUS_GPILOT_K", 12);
    if (p->cfg.pilot_k > APUS_GPILOT_MAX_K)
        p->cfg.pilot_k = APUS_GPILOT_MAX_K;
    if (p->cfg.prefill_k == 0)
        p->cfg.prefill_k = apus_env_int("APUS_GPILOT_PREFILL_K", 12);
    if (p->cfg.prefill_k > APUS_GPILOT_MAX_K)
        p->cfg.prefill_k = APUS_GPILOT_MAX_K;
    /* M7a: clamp both depths to n_experts. apus_gpilot_predict clamps
     * n > E internally and writes only E entries; the hooks used the
     * UNCLAMPED k for the pending memcpy / hint loop / seen bitset, so
     * k > E read uninitialized tset entries and hinted garbage expert
     * ids (OOB cache records; OOB seen-bitset writes) — segfault on
     * Linux/gcc with the default k=12 on an E=8 fixture (silent UB on
     * macOS). m6g never hit it (k=5 < E=8; the real model has E=288).
     * No-op for every gated config. */
    if (p->cfg.pilot_k > p->cfg.n_experts)
        p->cfg.pilot_k = p->cfg.n_experts;
    if (p->cfg.prefill_k > p->cfg.n_experts)
        p->cfg.prefill_k = p->cfg.n_experts;
    size_t hc = (size_t)p->cfg.hc_mult, dim = p->cfg.dim;
    size_t mix = (2 + hc) * hc;
    int E = p->cfg.n_experts, nl = p->cfg.n_layers;
    int k = p->cfg.pilot_k > 0 ? p->cfg.pilot_k : 1;
    p->routers = (ApusGpilotRouter *)calloc((size_t)nl, sizeof *p->routers);
    p->x4f = (float *)malloc(hc * dim * sizeof(float));
    p->pre = (float *)malloc(hc * sizeof(float));
    p->post = (float *)malloc(hc * sizeof(float));
    p->comb = (float *)malloc(hc * hc * sizeof(float));
    p->mixes = (float *)malloc(mix * sizeof(float));
    p->x = (uint16_t *)malloc(dim * sizeof(uint16_t));
    p->scores = (float *)malloc((size_t)E * sizeof(float));
    p->biased = (float *)malloc((size_t)E * sizeof(float));
    p->wgt = (float *)malloc((size_t)p->cfg.topk * sizeof(float));
    p->tidx = (int32_t *)malloc((size_t)p->cfg.topk * sizeof(int32_t));
    p->tset = (int32_t *)malloc(APUS_GPILOT_MAX_K * sizeof(int32_t));
    p->seen = (uint64_t *)malloc(((size_t)E + 63) / 64 * sizeof(uint64_t));
    p->pending = (int32_t *)calloc((size_t)nl * (size_t)k,
                                   sizeof(int32_t));
    p->pending_ok = (uint8_t *)calloc((size_t)nl, 1);
    p->pending_pos = (int64_t *)calloc((size_t)nl, sizeof(int64_t));
    if (!p->routers || !p->x4f || !p->pre || !p->post || !p->comb
        || !p->mixes || !p->x || !p->scores || !p->biased || !p->wgt
        || !p->tidx || !p->tset || !p->seen || !p->pending
        || !p->pending_ok || !p->pending_pos) {
        apus_gpilot_destroy(p);
        return NULL;
    }
    return p;
}

void apus_gpilot_destroy(ApusGpilot *p) {
    if (!p) return;
    free(p->routers);
    free(p->x4f);
    free(p->pre);
    free(p->post);
    free(p->comb);
    free(p->mixes);
    free(p->x);
    free(p->scores);
    free(p->biased);
    free(p->wgt);
    free(p->tidx);
    free(p->tset);
    free(p->seen);
    free(p->pending);
    free(p->pending_ok);
    free(p->pending_pos);
    free(p);
}

void apus_gpilot_attach_router(ApusGpilot *p, int layer,
                               const ApusGpilotRouter *r) {
    if (!p || layer < 0 || layer >= p->cfg.n_layers) return;
    p->routers[layer] = *r;
}

void apus_gpilot_attach(ApusGpilot *p, ApusGmodel *m) {
    if (!p || !m) return;
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    int nl = c->num_hidden_layers < p->cfg.n_layers
             ? c->num_hidden_layers : p->cfg.n_layers;
    for (int l = 0; l < nl; l++) {
        ApusGmodelPilotView v;
        if (apus_gmodel_pilot_view(m, l, &v)) continue;
        ApusGpilotRouter r;
        r.hc_fn = v.hc_fn;
        r.hc_base = v.hc_base;
        r.hc_scale = v.hc_scale;
        r.post_norm = v.post_norm;
        r.gate_w = v.gate_w;
        r.gate_bias = v.gate_bias;
        apus_gpilot_attach_router(p, l, &r);
    }
    ApusGmodelHooks hooks;
    apus_gpilot_hooks(p, &hooks);
    apus_gmodel_set_hooks(m, &hooks);
}

void apus_gpilot_stats(ApusGpilot *p, ApusGpilotStats *out) {
    *out = p->stats;
}

#endif /* APUS_GPILOT_IMPLEMENTATION */
#endif /* APUS_GPILOT_H */
