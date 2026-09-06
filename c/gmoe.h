/*
 * c/gmoe.h — GLM-5.3-Flash MoE sublayer (M4a): sigmoid router with
 * selection-only noaux_tc bias, FP8-E4M3 routed experts with the
 * swiglu_limit=10 clamp semantics, shared expert, bf16-stepped
 * accumulation. C11, libc only (+ c/bf16.h kernels). Scalar + the M3
 * bitwise BF16 kernels; batching/mt beyond the per-GEMV row pool is M7
 * perf work.
 *
 * Normative reference: tools/oracle.py router_forward / moe_forward /
 * swiglu_mlp (f32 mode), porting reference/inference/modeling_glm5_next.py
 * (glm5:86-207). Semantics pins (tests/m0/README.md):
 *
 *   Router (glm5:158-183), per token, ALL fp32:
 *       logits[e] = x @ gate_w[e]         (fp32 sequential-k, NO rounding)
 *       scores[e] = sigmoid(logits[e])    (numpy-stable form, c/gmhc.h)
 *       biased[e] = scores[e] + gate_bias[e]     (noaux_tc bias)
 *       idx  = top-k over BIASED scores, stable descending, ties to the
 *              LOWER index (the M0 pin; torch.topk tie order unspecified)
 *       w[j] = scores[idx[j]]             (from the UNBIASED scores)
 *       w[j] = w[j] / (sum_j w[j] + 1e-20)       (norm_topk_prob)
 *       w[j] = w[j] * 2.5                        (routed_scaling_factor)
 *     group-limiting is a literal no-op at n_group=1/topk_group=1 (the
 *     only GLM-5.3-Flash configuration; NOT implemented otherwise).
 *   Experts (swiglu_mlp, glm5:98-104 dense MLP / 120-142 expert), FP8
 *   weights via the oracle's fp8_linear (dequant -> bf16 -> bf16 matmul
 *   fp32-accumulate -> bf16; the caller dequantizes with c/fp8blk.h, the
 *   composition is exactly m3g's bitwise-gated path):
 *       g = bf16(x @ gate^T); u = bf16(x @ up^T)
 *       g = min(g, limit)                  gate clamped ABOVE only
 *       u = clip(u, -limit, limit)         up clamped BOTH sides
 *       h = bf16( g * sigmoid(g) )         (silu; clamps BEFORE silu)
 *       h = bf16( h * u )                  (u is bf16-valued; the oracle's
 *                                           second _B(u) is a no-op)
 *       out = bf16( h @ down^T )
 *   MoE forward (glm5:200-207 + the eager experts loop 120-135):
 *       y = +0.0 (bf16-valued)
 *       for e in ASCENDING EXPERT INDEX order:
 *           contrib = bf16( expert_e(x_t) * w[t,e] )     per routed token
 *           y[t]    = bf16( y[t] + contrib )             one rounding PER ADD
 *       shared  = expert_shared(x_t)                     (same swiglu)
 *       out[t]  = bf16( y[t] + shared[t] )               shared LAST
 *
 * The batched (s > 1) forward runs per-token GEMVs: every oracle matmul
 * row is M-independent bitwise (m3g), so per-token == batched bitwise.
 * Per-element accumulations keep the oracle's ascending-expert order.
 * GEMVs dispatch to apus_bf16_gemv_mt (bitwise at every APUS_THREADS).
 *
 * Bitwise contract vs the numpy oracle f32 mode: same as c/gmhc.h —
 * numpy-stable sigmoid, numpy pairwise-sum replica for the weight
 * normalization sum (topk < 8: sequential from w[0]; topk = 8: the
 * 8-accumulator tree), _mm-order dots, -ffp-contract=off. The
 * host-transcendental (expf) caveat and the m4g probe/fallback are
 * documented in tests/m4g/README.md.
 *
 * Usage: #define APUS_GMOE_IMPLEMENTATION in exactly one TU. Needs
 * c/bf16.h (APUS_BF16_IMPLEMENTATION) and c/gmhc.h
 * (APUS_GMHC_IMPLEMENTATION — the shared numpy-stable sigmoid and the
 * pairwise-sum replica) linked in the same binary.
 */
#ifndef APUS_GMOE_H
#define APUS_GMOE_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"
#include "gmhc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* fp32 sequential dot of two BF16-code rows (widened exactly), +0.0f
 * init, ascending k, mul+add two roundings (oracle _mm order), NO output
 * rounding — the router logits. */
float apus_gmoe_dot_bf16(const uint16_t *wrow, const uint16_t *x, size_t k);

/* Stable descending top-k: ties break to the LOWER index (the M0 pin,
 * np.argsort(-row, kind="stable")[:k]). row [n], idx [k] out. */
void apus_gmoe_topk_stable(const float *row, int n, int k, int32_t *idx);

typedef struct {
    int E, topk;
    size_t dim;
    float route_scale;          /* routed_scaling_factor (2.5) */
    const uint16_t *gate_w;     /* [E, dim] BF16 codes */
    const float *gate_bias;     /* [E] f32 (e_score_correction_bias) */
} ApusGmoeRouterW;

/* Per-token router. x [dim] BF16 codes. Outputs: scores [E] (UNBIASED
 * sigmoid scores), idx [topk], w [topk] (normalized, scaled).
 * biased: scratch [E]; when non-NULL it returns the biased scores. */
void apus_gmoe_router(const ApusGmoeRouterW *r, const uint16_t *x,
                      float *scores, int32_t *idx, float *w, float *biased);

/* One expert / dense MLP / shared expert on dequantized BF16 weights.
 * wg/wu [inter, dim], wd [dim, inter] BF16 codes; x [dim], out [dim]
 * BF16 codes. Scratch: g/u/h [inter] codes, xf [max(dim, inter)] floats
 * (gemv widen scratch; may alias nothing). limit > 0 (swiglu_limit=10). */
void apus_gmoe_expert(const uint16_t *wg, const uint16_t *wu,
                      const uint16_t *wd, const uint16_t *x,
                      size_t dim, size_t inter, float limit,
                      uint16_t *out, uint16_t *g, uint16_t *u,
                      uint16_t *h, float *xf);

typedef struct {
    ApusGmoeRouterW router;
    size_t inter;                       /* routed-expert intermediate */
    const uint16_t *const *eg;          /* [E] gate [inter, dim] codes */
    const uint16_t *const *eu;          /* [E] up   [inter, dim] codes */
    const uint16_t *const *ed;          /* [E] down [dim, inter] codes */
    const uint16_t *sg, *su, *sd;       /* shared expert (BF16 codes) */
    size_t sinter;                      /* shared intermediate */
    float limit;                        /* swiglu_limit (10) */
} ApusGmoeW;

/* Named intermediates (NULL to skip). Buffers: router_scores/router_biased
 * [s,E] f32, router_idx [s,topk] i32, router_w [s,topk] f32,
 * moe_routed/moe_shared [s,dim] f32 (bf16-valued). */
typedef struct {
    float *router_scores, *router_biased, *router_w;
    int32_t *router_idx;
    float *moe_routed, *moe_shared;
} ApusGmoeInterm;

/* Engine-owned scratch (M5 re-home: apus_gmoe_forward used to malloc all
 * of this per call). Sizes derive from the weight struct + the max token
 * count s; one scratch may be reused across layers with identical
 * dims and across calls with s <= the init s. Pure allocation change —
 * numerics identical to the per-call malloc version. */
typedef struct {
    float *scores, *biased, *wgt, *xf, *y;
    int32_t *idx;
    uint16_t *g, *u, *h, *eo, *sh;
    size_t y_elems;                 /* allocated s*dim */
} ApusGmoeScratch;

void apus_gmoe_scratch_init(ApusGmoeScratch *sc, const ApusGmoeW *w,
                            size_t s);
void apus_gmoe_scratch_free(ApusGmoeScratch *sc);

/* MoE sublayer on caller-owned scratch. x [s, dim] BF16 codes, out
 * [s, dim] BF16 codes. forced_idx (TEST-ONLY teacher-forcing, tests/m5g
 * tolerance tier): [s, topk] selections to use INSTEAD of the computed
 * top-k (scores/weights are still computed from the UNBIASED scores at
 * those indices); NULL computes normally. */
void apus_gmoe_forward2(const ApusGmoeW *w, const uint16_t *x, size_t s,
                        uint16_t *out, ApusGmoeInterm *interm,
                        ApusGmoeScratch *sc, const int32_t *forced_idx);

/* Convenience wrapper (the pre-M5 API): owns a temporary scratch.
 * Identical numerics to apus_gmoe_forward2(..., NULL). */
void apus_gmoe_forward(const ApusGmoeW *w, const uint16_t *x, size_t s,
                       uint16_t *out, ApusGmoeInterm *interm);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GMOE_IMPLEMENTATION) && !defined(APUS_GMOE_IMPL_INCLUDED)
#define APUS_GMOE_IMPL_INCLUDED

#include <stdlib.h>
#include <string.h>

float apus_gmoe_dot_bf16(const uint16_t *wrow, const uint16_t *x, size_t k) {
    float acc = 0.0f;
    for (size_t i = 0; i < k; i++)
        acc += apus_bf16_f32(wrow[i]) * apus_bf16_f32(x[i]);
    return acc;
}

void apus_gmoe_topk_stable(const float *row, int n, int k, int32_t *idx) {
    /* k passes of strict-> argmax over the remaining entries: descending
     * order, ties keep the lower index first (stable). */
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            int used = 0;
            for (int q = 0; q < j; q++)
                if (idx[q] == i) { used = 1; break; }
            if (used) continue;
            if (best < 0 || row[i] > row[best]) best = i;
        }
        idx[j] = best;
    }
}

void apus_gmoe_router(const ApusGmoeRouterW *r, const uint16_t *x,
                      float *scores, int32_t *idx, float *w, float *biased) {
    int E = r->E, topk = r->topk;
    float bstack[512];
    float *b = biased ? biased : bstack;   /* E <= 512 without a buffer */
    for (int e = 0; e < E; e++) {
        float logit = apus_gmoe_dot_bf16(r->gate_w + (size_t)e * r->dim,
                                         x, r->dim);
        scores[e] = apus_gmhc_sigmoid(logit);
        b[e] = scores[e] + r->gate_bias[e];
    }
    apus_gmoe_topk_stable(b, E, topk, idx);
    for (int j = 0; j < topk; j++) w[j] = scores[idx[j]];
    /* norm_topk_prob: numpy pairwise order over the gathered weights */
    float sum = apus_gmhc_pw_sum(w, (size_t)topk);
    for (int j = 0; j < topk; j++)
        w[j] = w[j] / (sum + 1e-20f) * r->route_scale;
}

void apus_gmoe_expert(const uint16_t *wg, const uint16_t *wu,
                      const uint16_t *wd, const uint16_t *x,
                      size_t dim, size_t inter, float limit,
                      uint16_t *out, uint16_t *g, uint16_t *u,
                      uint16_t *h, float *xf) {
    apus_bf16_gemv_mt(wg, x, xf, g, inter, dim);
    apus_bf16_gemv_mt(wu, x, xf, u, inter, dim);
    for (size_t i = 0; i < inter; i++) {
        float gv = apus_bf16_f32(g[i]), uv = apus_bf16_f32(u[i]);
        /* clamps BEFORE the silu: gate above only, up both sides */
        if (gv > limit) gv = limit;
        if (uv > limit) uv = limit;
        else if (uv < -limit) uv = -limit;
        float hv = apus_bf16_round(gv * apus_gmhc_sigmoid(gv));
        h[i] = apus_bf16_bits(apus_bf16_round(hv * uv));
    }
    apus_bf16_gemv_mt(wd, h, xf, out, dim, inter);
}

void apus_gmoe_scratch_init(ApusGmoeScratch *sc, const ApusGmoeW *w,
                            size_t s) {
    int E = w->router.E, topk = w->router.topk;
    size_t dim = w->router.dim;
    size_t maxinter = w->inter > w->sinter ? w->inter : w->sinter;
    size_t maxk = dim > maxinter ? dim : maxinter;
    sc->scores = (float *)malloc((size_t)E * sizeof(float));
    sc->biased = (float *)malloc((size_t)E * sizeof(float));
    sc->idx = (int32_t *)malloc((size_t)topk * sizeof(int32_t));
    sc->wgt = (float *)malloc((size_t)topk * sizeof(float));
    sc->g = (uint16_t *)malloc(maxinter * sizeof(uint16_t));
    sc->u = (uint16_t *)malloc(maxinter * sizeof(uint16_t));
    sc->h = (uint16_t *)malloc(maxinter * sizeof(uint16_t));
    sc->xf = (float *)malloc(maxk * sizeof(float));
    sc->eo = (uint16_t *)malloc(dim * sizeof(uint16_t));
    sc->sh = (uint16_t *)malloc(dim * sizeof(uint16_t));
    const size_t yn = (size_t)s * dim;
    sc->y = (float *)calloc(yn ? yn : 1, sizeof(float));
    sc->y_elems = s * dim;
}

void apus_gmoe_scratch_free(ApusGmoeScratch *sc) {
    free(sc->y);
    free(sc->sh);
    free(sc->eo);
    free(sc->xf);
    free(sc->h);
    free(sc->u);
    free(sc->g);
    free(sc->wgt);
    free(sc->idx);
    free(sc->biased);
    free(sc->scores);
    memset(sc, 0, sizeof *sc);
}

void apus_gmoe_forward2(const ApusGmoeW *w, const uint16_t *x, size_t s,
                        uint16_t *out, ApusGmoeInterm *interm,
                        ApusGmoeScratch *sc, const int32_t *forced_idx) {
    int E = w->router.E, topk = w->router.topk;
    size_t dim = w->router.dim;
    float *scores = sc->scores, *biased = sc->biased, *wgt = sc->wgt;
    float *xf = sc->xf, *y = sc->y;
    int32_t *idx = sc->idx;
    uint16_t *g = sc->g, *u = sc->u, *h = sc->h, *eo = sc->eo, *sh = sc->sh;
    memset(y, 0, s * dim * sizeof(float));      /* +0.0f init */

    /* router (per token; selection on the BIASED scores, weights from the
     * UNBIASED ones) */
    for (size_t t = 0; t < s; t++) {
        apus_gmoe_router(&w->router, x + t * dim, scores, idx, wgt, biased);
        if (forced_idx) {
            /* teacher-forcing (tests/m5g tolerance tier): weights are
             * re-gathered from the UNBIASED scores at the FORCED indices,
             * same normalize+scale recipe as apus_gmoe_router. */
            memcpy(idx, forced_idx + t * (size_t)topk,
                   (size_t)topk * sizeof(int32_t));
            for (int j = 0; j < topk; j++) wgt[j] = scores[idx[j]];
            float sum = apus_gmhc_pw_sum(wgt, (size_t)topk);
            for (int j = 0; j < topk; j++)
                wgt[j] = wgt[j] / (sum + 1e-20f) * w->router.route_scale;
        }
        if (interm) {
            if (interm->router_scores)
                memcpy(interm->router_scores + t * (size_t)E, scores,
                       (size_t)E * sizeof(float));
            if (interm->router_biased)
                memcpy(interm->router_biased + t * (size_t)E, biased,
                       (size_t)E * sizeof(float));
            if (interm->router_idx)
                memcpy(interm->router_idx + t * (size_t)topk, idx,
                       (size_t)topk * sizeof(int32_t));
            if (interm->router_w)
                memcpy(interm->router_w + t * (size_t)topk, wgt,
                       (size_t)topk * sizeof(float));
        }
        /* routed experts, ASCENDING EXPERT INDEX, one bf16 rounding per
         * add (the oracle's eager loop) */
        for (int e = 0; e < E; e++) {
            int slot = -1;
            for (int j = 0; j < topk; j++)
                if (idx[j] == e) { slot = j; break; }
            if (slot < 0) continue;
            apus_gmoe_expert(w->eg[e], w->eu[e], w->ed[e], x + t * dim,
                             dim, w->inter, w->limit, eo, g, u, h, xf);
            float wt = wgt[slot];
            float *yt = y + t * dim;
            for (size_t i = 0; i < dim; i++) {
                float contrib = apus_bf16_round(apus_bf16_f32(eo[i]) * wt);
                yt[i] = apus_bf16_round(yt[i] + contrib);
            }
        }
    }
    if (interm && interm->moe_routed)
        memcpy(interm->moe_routed, y, s * dim * sizeof(float));

    /* shared expert, added LAST (one more bf16 rounding) */
    for (size_t t = 0; t < s; t++) {
        apus_gmoe_expert(w->sg, w->su, w->sd, x + t * dim, dim, w->sinter,
                         w->limit, sh, g, u, h, xf);
        float *yt = y + t * dim;
        uint16_t *orow = out + t * dim;
        for (size_t i = 0; i < dim; i++) {
            if (interm && interm->moe_shared)
                interm->moe_shared[t * dim + i] = apus_bf16_f32(sh[i]);
            orow[i] = apus_bf16_bits(yt[i] + apus_bf16_f32(sh[i]));
        }
    }
}

void apus_gmoe_forward(const ApusGmoeW *w, const uint16_t *x, size_t s,
                       uint16_t *out, ApusGmoeInterm *interm) {
    ApusGmoeScratch sc;
    apus_gmoe_scratch_init(&sc, w, s);
    apus_gmoe_forward2(w, x, s, out, interm, &sc, NULL);
    apus_gmoe_scratch_free(&sc);
}

#endif /* APUS_GMOE_IMPLEMENTATION */
#endif /* APUS_GMOE_H */
