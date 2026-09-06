/*
 * c/gdsa.h — GLM-5.3-Flash DSA sublayer (M4b): MLA pure-NoPE attention
 * (q_a/q_b, kv_a/kv_b, o_proj — FP8 via m3g dequant+bf16 GEMM, kv_b BF16)
 * with the Lightning Indexer (top-k pool selection). C11, libc only
 * (+ c/bf16.h, c/fp8blk.h kernels). Scalar reference paths (SIMD/mt
 * beyond the m3g GEMMs is M7 perf work — it must preserve these orders
 * bitwise).
 *
 * Normative reference: tools/oracle.py dsa_forward / indexer_forward /
 * dsa_attention (f32 mode), porting reference/inference/modeling_glm5_next.py
 * (glm5:737-1257). This is a FRESH port of the GLM indexer variant — the
 * inherited V4 c/attn.h machinery is NOT reused for the numerics: GLM has
 * NO Hadamard rotation, NO FP4/score quantization, NO RoPE anywhere, NO
 * attention sink; top-k via kpool-4 compressed pools + always-selected
 * tail (width index_topk + kpool - 1).
 *
 * Semantics (f32-faithful; tests/m0/README.md is the pin):
 *
 *   MLA (glm5:1156-1217), per token:
 *       q_resid = RMSNorm(fp8(q_a))          (weighted, eps 1e-5: fp32
 *                  internal, bf16 round, THEN * weight, round again)
 *       q       = fp8(q_b) reshaped [H, qd]
 *       k_pass  = RMSNorm(fp8(kv_a))
 *       k|v     = bf16(kv_b) split [H, qd] | [H, vd]
 *       out     = fp8(o_proj) of the attention output
 *     All fp8 linears: m3g dequant -> bf16 -> bf16 GEMM (fp32 sequential
 *     ascending-k accumulate, bf16 out) — bitwise == the oracle _mm.
 *     The expanded kv cache stores BF16 codes (engine layout: head-major
 *     [H][pos*dim]; the oracle's [pos][H][dim] carries the same values).
 *   Indexer (glm5:774-1025), ALL bf16 weights, under no_grad:
 *       idx_q   = bf16(wq_b @ q_resid) [s, IH, ID]
 *       idx_k   = LayerNorm(bf16(wk @ x), w, b, eps 1e-6)  (fp32 internal
 *                 with the numpy pairwise mean/var, one bf16 rounding)
 *       idx_g   = bf16(compress_gate @ x)
 *     The caches are appended with THIS call's tokens and ALL k-pools are
 *     REBUILT from the full cache every forward, decode included
 *     (glm5:811): only FULL pools (all kpool tokens valid) are scored;
 *     pool key = softmax(gate + ape) weighted average of member keys —
 *     the pool probs are CAST TO BF16 before the weighted average
 *     (glm5:965-968) and the average rounds to bf16 again.
 *       sc[t,h,p] = idx_q[t,h] @ pool_key[p]     (fp32 _mm, NO rounding)
 *       sc        = relu(sc * ID^-0.5)           (relu AFTER the scale)
 *       w[t,h]    = bf16(weights_proj @ x) * IH^-0.5
 *       index_scores[t,p] = sum_h w[t,h] * sc[t,h,p]   (fp32, +0.0f)
 *     Candidate validity: the pool's LAST token must be causally visible
 *     (pool_end = p*kpool + kpool-1 <= q_pos); masked scores are
 *     finfo.min. top-(index_topk/kpool) pools, stable descending, ties to
 *     the LOWER pool index (the M0 pin), invalid selections -> -1; each
 *     selected pool expands to its kpool raw token indices. The CURRENT
 *     INCOMPLETE pool is always appended as up to kpool-1 raw tail
 *     indices (tail_count = (q_pos+1) % kpool; tail_count = 0 -> all
 *     -1). Output width = index_topk + kpool - 1, -1-padded.
 *   Attention (glm5:1040-1062, 1219-1257): over the FULL cached kv with
 *     the additive-mask semantics (NOT a gather — duplicated/invalid
 *     indices are excluded via the boolean dedup, everything not
 *     selected gets finfo.min):
 *       sc = bf16(bf16(q) @ bf16(k)^T)   (bf16 matmul, fp32 acc, round)
 *       sc = bf16(sc * scale)            (scale = qd^-0.5, round again)
 *       sc = sc + (visible ? 0 : finfo.min)
 *       p  = bf16(softmax_fp32(sc))      (max-sub, exp, pairwise sum)
 *       o  = bf16(p @ bf16(v))           (ascending kv positions, +0.0f)
 *
 * numpy reduction orders (verified bitwise against numpy 2.5.2,
 * tests/m4h README): CONTIGUOUS-axis sums use numpy pairwise
 * (apus_gmhc_pw_sum); STRIDED-axis sums (the pool softmax/weighted
 * average over the kpool axis) are PLAIN SEQUENTIAL from +0.0f; matmul
 * contractions follow the oracle _mm order.
 *
 * Usage: #define APUS_GDSA_IMPLEMENTATION in exactly one TU. Needs
 * c/bf16.h (APUS_BF16_IMPLEMENTATION), c/fp8blk.h
 * (APUS_FP8BLK_IMPLEMENTATION), c/gmhc.h (APUS_GMHC_IMPLEMENTATION) and
 * c/gmoe.h (APUS_GMOE_IMPLEMENTATION — the stable top-k) linked in the
 * same binary.
 */
#ifndef APUS_GDSA_H
#define APUS_GDSA_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"
#include "fp8blk.h"
#include "gmhc.h"
#include "gmoe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* finfo(np.float32).min — the additive mask value (glm5:1254). */
#define APUS_GDSA_NEG_INF (-3.4028234663852886e+38f)

typedef struct {
    size_t dim, H, qd, vd, ql, kl;
    const uint8_t *q_a_c;  const float *q_a_s;   /* FP8 [ql, dim] */
    const uint16_t *q_a_norm;                     /* [ql] codes */
    const uint8_t *q_b_c;  const float *q_b_s;   /* FP8 [H*qd, ql] */
    const uint8_t *kv_a_c; const float *kv_a_s;  /* FP8 [kl, dim] */
    const uint16_t *kv_a_norm;                    /* [kl] codes */
    const uint16_t *kv_b;                         /* [H*(qd+vd), kl] codes */
    const uint8_t *o_c;    const float *o_s;     /* FP8 [dim, H*vd] */
    float eps;                                    /* rms_norm_eps (1e-5) */
    /* Lightning indexer (all BF16) */
    size_t IH, ID;
    size_t topk, kpool;
    int tail;                 /* index_kpool_always_select_tail */
    const uint16_t *idx_wq_b;      /* [IH*ID, ql] */
    const uint16_t *idx_wk;        /* [ID, dim] */
    const uint16_t *idx_knorm_w;   /* [ID] */
    const uint16_t *idx_knorm_b;   /* [ID] */
    const uint16_t *idx_wproj;     /* [IH, dim] */
    const uint16_t *idx_ape;       /* [kpool, ID] */
    const uint16_t *idx_gate;      /* [ID, dim] */
} ApusGdsaW;

/* Cache state (capacity managed by the caller). k/v: head-major
 * [H][cap*dim] codes (cap = the stride between heads); idx_k/idx_gate:
 * token-major [cap][ID] codes. */
typedef struct {
    uint16_t *k_cache;
    uint16_t *v_cache;
    uint16_t *idx_k;
    uint16_t *idx_gate;
    size_t cap;                   /* allocated positions */
    size_t n;                     /* cached tokens */
} ApusGdsaState;

/* Selection width: index_topk + (tail ? kpool - 1 : 0). */
size_t apus_gdsa_width(const ApusGdsaW *W);

/* Weighted RMSNorm (glm5:65-83): fp32 internal (numpy pairwise mean),
 * bf16 round, THEN * weight, round again. x/w/y [n] codes. */
void apus_gdsa_rmsnorm(const uint16_t *x, const uint16_t *w, uint16_t *y,
                       size_t n, float eps);

/* LayerNorm (glm5:764, indexer k_norm): fp32 internal (numpy pairwise
 * biased mean/var), y = (x-mu)*(1/sqrt(var+eps)) * w + b, ONE bf16
 * rounding. x/w/b/y [n] codes. */
void apus_gdsa_layernorm(const uint16_t *x, const uint16_t *w,
                         const uint16_t *b, uint16_t *y,
                         size_t n, float eps);

/* Lightning indexer (oracle indexer_forward): computes this call's
 * idx k/gate rows, APPENDS them to st->idx_k/idx_gate (st->n is the base
 * count BEFORE this call; the caller bumps st->n after appending the
 * kv cache — the indexer itself only reads st->n), rebuilds all pools,
 * and emits topk [s, apus_gdsa_width(W)] (-1 = invalid).
 * idx_scores (optional): [s, (st->n+s)/kpool] f32 raw pool scores.
 * idx_knew/idx_gnew (optional): [s, ID] codes, this call's rows. */
void apus_gdsa_indexer(const ApusGdsaW *W, const uint16_t *x,
                       const uint16_t *q_resid, size_t s,
                       ApusGdsaState *st, int32_t *topk,
                       float *idx_scores,
                       uint16_t *idx_knew, uint16_t *idx_gnew);

/* Sparse attention over the full cached kv with the additive mask
 * (oracle dsa_attention). q [s, H*qd] codes; st holds n tokens of cache;
 * topk [s, width]; out [s, H*vd] codes; probs (optional) [s, H, n]
 * codes. */
void apus_gdsa_attention(const ApusGdsaW *W, const uint16_t *q,
                         const ApusGdsaState *st, size_t s,
                         const int32_t *topk, uint16_t *out,
                         uint16_t *probs);

/* Named intermediates (NULL to skip): q_resid [s,ql], q [s,H*qd],
 * k_new/v_new [s,H*qd]/[s,H*vd], idx_knew/idx_gnew [s,ID], attn
 * [s,H*vd] — all BF16 codes; idx_scores [s,n_full] f32; idx_topk
 * [s,width] i32; probs [s,H,n] codes. */
typedef struct {
    uint16_t *q_resid;
    uint16_t *q;
    uint16_t *k_new;
    uint16_t *v_new;
    uint16_t *idx_knew;
    uint16_t *idx_gnew;
    float *idx_scores;
    int32_t *idx_topk;
    uint16_t *probs;
    uint16_t *attn;
} ApusGdsaInterm;

/* Reusable dequant workspace for the FP8 projections (P5). The FP8
 * q_a/q_b/kv_a/o projections dequant to BF16 on every forward in CPU mode
 * (the M7b fused Metal hook skips the materialization entirely); sizing the
 * scratch ONCE to the true max projection (not omax*xmax) and reusing it
 * across layers/tokens removes a per-call ~O*K malloc+page-fault churn.
 * Numerics-neutral: wbuf is a pure workspace, fully rewritten by the
 * dequant before the GEMM reads it. */
typedef struct {
    uint16_t *wbuf;         /* owned; grown to the max projection O*K */
    size_t    wbuf_cap;     /* elements */
} ApusGdsaScratch;

void apus_gdsa_scratch_free(ApusGdsaScratch *sc);

/* Full DSA sublayer (oracle dsa_forward): x [s, dim] codes -> out
 * [s, dim] codes. Appends s tokens to the caches (st->n updated). */
void apus_gdsa_forward(const ApusGdsaW *W, const uint16_t *x, size_t s,
                       ApusGdsaState *st, uint16_t *out,
                       ApusGdsaInterm *im);

/* apus_gdsa_forward + TEST-ONLY teacher-forcing (tests/m5g tolerance
 * tier): when forced_topk ([s, width], -1 = invalid) is non-NULL the
 * attention uses it INSTEAD of this call's own indexer selection. The
 * indexer still runs (its caches must append identically); only the
 * selection consumed by the attention is replaced. NULL == apus_gdsa_forward. */
void apus_gdsa_forward2(const ApusGdsaW *W, const uint16_t *x, size_t s,
                        ApusGdsaState *st, uint16_t *out,
                        ApusGdsaInterm *im, const int32_t *forced_topk);

/* apus_gdsa_forward2 + a reusable dequant scratch (P5; NULL = per-call
 * malloc, identical numerics). BITWISE == apus_gdsa_forward2 either way. */
void apus_gdsa_forward3(const ApusGdsaW *W, const uint16_t *x, size_t s,
                        ApusGdsaState *st, uint16_t *out,
                        ApusGdsaInterm *im, const int32_t *forced_topk,
                        ApusGdsaScratch *sc);

/* Decode-batch entry (M8b, the GLM speculative verify batch): s tokens
 * stepped through the caches in EXACT sequential-decode order — per token
 * the kv append, the indexer cache append + full-pool rebuild, the
 * selection and the attention all run at cache count base+t+1 (never
 * base+s), so the result is BITWISE == s one-by-one apus_gdsa_forward2
 * calls by construction (the interleave IS the sequential call loop).
 * The one-shot batched forward above is the PREFILL ordering and is NOT
 * bitwise-equal to sequential decode: its single pool rebuild would
 * include tokens a sequential decode has not written yet (masked, but the
 * top-k row layout clamps differently), and its attention softmax pairwise
 * sums run over base+s entries instead of base+t+1. forced_topk: per-token
 * rows [s, width] as in apus_gdsa_forward2 (TEST-ONLY). */
void apus_gdsa_decode_batch(const ApusGdsaW *W, const uint16_t *x, size_t s,
                            ApusGdsaState *st, uint16_t *out,
                            const int32_t *forced_topk);

/* decode_batch + the P5 reusable dequant scratch (NULL = per-call malloc;
 * identical numerics either way). */
void apus_gdsa_decode_batch2(const ApusGdsaW *W, const uint16_t *x, size_t s,
                             ApusGdsaState *st, uint16_t *out,
                             const int32_t *forced_topk,
                             ApusGdsaScratch *sc);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GDSA_IMPLEMENTATION) && !defined(APUS_GDSA_IMPL_INCLUDED)
#define APUS_GDSA_IMPL_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <string.h>

size_t apus_gdsa_width(const ApusGdsaW *W) {
    return W->topk + (W->tail ? W->kpool - 1 : 0);
}

void apus_gdsa_rmsnorm(const uint16_t *x, const uint16_t *w, uint16_t *y,
                       size_t n, float eps) {
    float *sq = (float *)malloc(n * sizeof(float));
    float *xf = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        xf[i] = apus_bf16_f32(x[i]);
        sq[i] = xf[i] * xf[i];
    }
    float var = apus_gmhc_pw_sum(sq, n) / (float)n;
    float inv = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++) {
        float t = apus_bf16_round(xf[i] * inv);        /* cast first */
        y[i] = apus_bf16_bits(apus_bf16_f32(w[i]) * t); /* then * weight */
    }
    free(xf);
    free(sq);
}

void apus_gdsa_layernorm(const uint16_t *x, const uint16_t *w,
                         const uint16_t *b, uint16_t *y,
                         size_t n, float eps) {
    /* calloc for xf: the fill loop covers [0, n) fully, but gcc 13's
     * -Wmaybe-uninitialized cannot prove it (false positive, Linux);
     * calloc makes the contents provably defined. */
    float *xf = (float *)calloc(n ? n : 1, sizeof(float));
    float *d = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        xf[i] = apus_bf16_f32(x[i]);
    float mu = apus_gmhc_pw_sum(xf, n) / (float)n;
    for (size_t i = 0; i < n; i++)
        d[i] = xf[i] - mu;
    float var = 0.0f;
    {
        float *sq = (float *)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++)
            sq[i] = d[i] * d[i];
        var = apus_gmhc_pw_sum(sq, n) / (float)n;
        free(sq);
    }
    float inv = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++) {
        float t = d[i] * inv;
        t = t * apus_bf16_f32(w[i]) + apus_bf16_f32(b[i]);
        y[i] = apus_bf16_bits(t);
    }
    free(d);
    free(xf);
}

void apus_gdsa_indexer(const ApusGdsaW *W, const uint16_t *x,
                       const uint16_t *q_resid, size_t s,
                       ApusGdsaState *st, int32_t *topk,
                       float *idx_scores,
                       uint16_t *idx_knew, uint16_t *idx_gnew) {
    size_t dim = W->dim, ql = W->ql, IH = W->IH, ID = W->ID;
    size_t kpool = W->kpool, width = apus_gdsa_width(W);
    size_t base = st->n;                  /* cache count BEFORE this call */
    size_t n = base + s;
    size_t n_full = n / kpool;
    float idx_scale = (float)pow((double)ID, -0.5);
    float ih_scale = (float)pow((double)IH, -0.5);

    /* idx_q = bf16(wq_b @ q_resid) [s, IH*ID] */
    uint16_t *iq = (uint16_t *)malloc(s * IH * ID * sizeof(uint16_t));
    float *xf = (float *)malloc(s * (ql > dim ? ql : dim) * sizeof(float));
    apus_bf16_gemm_mt(W->idx_wq_b, q_resid, xf, iq, s, IH * ID, ql);
    /* idx_k = LayerNorm(bf16(wk @ x)); idx_g = bf16(gate @ x) */
    uint16_t *kpre = (uint16_t *)malloc(s * ID * sizeof(uint16_t));
    apus_bf16_gemm_mt(W->idx_wk, x, xf, kpre, s, ID, dim);
    uint16_t *gnew = (uint16_t *)malloc(s * ID * sizeof(uint16_t));
    apus_bf16_gemm_mt(W->idx_gate, x, xf, gnew, s, ID, dim);
    for (size_t t = 0; t < s; t++) {
        apus_gdsa_layernorm(kpre + t * ID, W->idx_knorm_w, W->idx_knorm_b,
                            st->idx_k + (base + t) * ID, ID, 1e-6f);
        memcpy(st->idx_gate + (base + t) * ID, gnew + t * ID,
               ID * sizeof(uint16_t));
    }
    if (idx_knew)
        for (size_t t = 0; t < s; t++)
            memcpy(idx_knew + t * ID, st->idx_k + (base + t) * ID,
                   ID * sizeof(uint16_t));
    if (idx_gnew)
        memcpy(idx_gnew, gnew, s * ID * sizeof(uint16_t));
    free(gnew);
    free(kpre);

    /* Rebuild ALL full pools from the full cache (glm5:900-973): pool
     * probs cast to BF16 before the weighted average; the average rounds
     * to bf16 again. kpool-axis sums are STRIDED -> plain sequential. */
    float *pkey = (float *)malloc((n_full ? n_full : 1) * ID
                                  * sizeof(float));
    float *lg = (float *)malloc(kpool * sizeof(float));
    float *e = (float *)malloc(kpool * sizeof(float));
    for (size_t p = 0; p < n_full; p++) {
        for (size_t d = 0; d < ID; d++) {
            for (size_t c = 0; c < kpool; c++)
                lg[c] = apus_bf16_f32(st->idx_gate[(p * kpool + c) * ID + d])
                      + apus_bf16_f32(W->idx_ape[c * ID + d]);
            float mx = lg[0];
            for (size_t c = 1; c < kpool; c++)
                if (lg[c] > mx) mx = lg[c];
            float sum = 0.0f;
            for (size_t c = 0; c < kpool; c++) {
                e[c] = expf(lg[c] - mx);
                sum += e[c];
            }
            float acc = 0.0f;
            for (size_t c = 0; c < kpool; c++) {
                float pr = apus_bf16_round(e[c] / sum);
                acc += pr
                     * apus_bf16_f32(st->idx_k[(p * kpool + c) * ID + d]);
            }
            pkey[p * ID + d] = apus_bf16_round(acc);
        }
    }
    free(e);
    free(lg);

    /* Scores (glm5:826-831): fp32 matmuls, NO bf16 rounding; relu AFTER
     * the head_dim^-0.5 scale; weights x n_heads^-0.5. */
    uint16_t *wp = (uint16_t *)malloc(s * IH * sizeof(uint16_t));
    apus_bf16_gemm_mt(W->idx_wproj, x, xf, wp, s, IH, dim);
    float *sm = (float *)malloc(s * (n_full ? n_full : 1) * sizeof(float));
    float *sc = (float *)malloc(IH * sizeof(float));
    float *w32 = (float *)malloc(IH * sizeof(float));
    for (size_t t = 0; t < s; t++) {
        size_t q_pos = base + t;          /* absolute query position */
        for (size_t h = 0; h < IH; h++)
            w32[h] = apus_bf16_f32(wp[t * IH + h]) * ih_scale;
        for (size_t p = 0; p < n_full; p++) {
            float acc = 0.0f;
            for (size_t h = 0; h < IH; h++) {
                float dot = 0.0f;
                for (size_t i = 0; i < ID; i++)
                    dot += apus_bf16_f32(iq[t * IH * ID + h * ID + i])
                         * pkey[p * ID + i];
                float v = dot * idx_scale;      /* relu AFTER the scale */
                sc[h] = v > 0.0f ? v : 0.0f;
            }
            for (size_t h = 0; h < IH; h++)
                acc += w32[h] * sc[h];
            sm[t * n_full + p] = acc;
        }
        if (idx_scores)
            memcpy(idx_scores + t * n_full, sm + t * n_full,
                   n_full * sizeof(float));
        /* candidate validity: the pool's LAST token must be visible */
        for (size_t p = 0; p < n_full; p++) {
            size_t pool_end = p * kpool + (kpool - 1);
            if (pool_end > q_pos)
                sm[t * n_full + p] = APUS_GDSA_NEG_INF;
        }
        /* top-(topk/kpool) pools, stable descending, lower index first */
        size_t select_k = W->topk / kpool;
        if (select_k > n_full)
            select_k = n_full;
        int32_t *row = topk + t * width;
        for (size_t i = 0; i < width; i++)
            row[i] = -1;
        if (select_k > 0) {
            int32_t *sel = (int32_t *)malloc(select_k * sizeof(int32_t));
            apus_gmoe_topk_stable(sm + t * n_full, (int)n_full,
                                  (int)select_k, sel);
            for (size_t j = 0; j < select_k; j++) {
                size_t pool_end = (size_t)sel[j] * kpool + (kpool - 1);
                int valid = pool_end <= q_pos;
                for (size_t o = 0; o < kpool; o++)
                    row[j * kpool + o] =
                        valid ? sel[j] * (int32_t)kpool + (int32_t)o : -1;
            }
            free(sel);
        }
        /* always-selected tail: the current incomplete pool — appended
         * IMMEDIATELY after the selected pools (glm5:1025), -1-padded to
         * the full width after that (glm5:873-875) */
        if (W->tail) {
            size_t visible = q_pos + 1;
            size_t tc = visible % kpool;
            size_t ts = visible - tc;
            for (size_t o = 0; o + 1 < kpool; o++)
                row[select_k * kpool + o] =
                    o < tc ? (int32_t)(ts + o) : -1;
        }
    }
    free(w32);
    free(sc);
    free(sm);
    free(wp);
    free(pkey);
    free(xf);
    free(iq);
}

void apus_gdsa_attention(const ApusGdsaW *W, const uint16_t *q,
                         const ApusGdsaState *st, size_t s,
                         const int32_t *topk, uint16_t *out,
                         uint16_t *probs) {
    size_t H = W->H, qd = W->qd, vd = W->vd, n = st->n;
    size_t width = apus_gdsa_width(W);
    float scale = (float)pow((double)qd, -0.5);
    uint16_t *qh = (uint16_t *)malloc(s * qd * sizeof(uint16_t));
    uint16_t *scb = (uint16_t *)malloc(s * n * sizeof(uint16_t));
    float *sc = (float *)malloc(n * sizeof(float));
    float *e = (float *)malloc(n * sizeof(float));
    unsigned char *vis = (unsigned char *)malloc(n);
    float *xf = (float *)malloc(s * qd * sizeof(float));

    for (size_t h = 0; h < H; h++) {
        /* extract the head's q rows (token-major -> contiguous) */
        for (size_t t = 0; t < s; t++)
            memcpy(qh + t * qd, q + t * H * qd + h * qd,
                   qd * sizeof(uint16_t));
        /* sc = bf16(q @ k^T) — the oracle's bf16 _mm (m3g kernel) */
        apus_bf16_gemm_mt(st->k_cache + h * st->cap * qd, qh, xf, scb,
                          s, n, qd);
        for (size_t t = 0; t < s; t++) {
            const int32_t *ids = topk + t * width;
            memset(vis, 0, n);
            for (size_t i = 0; i < width; i++)
                if (ids[i] >= 0 && (size_t)ids[i] < n)
                    vis[ids[i]] = 1;
            /* bf16(sc * scale), then the additive mask (finfo.min) */
            float mx = APUS_GDSA_NEG_INF;
            for (size_t j = 0; j < n; j++) {
                float v = apus_bf16_round(apus_bf16_f32(scb[t * n + j])
                                          * scale);
                v = v + (vis[j] ? 0.0f : APUS_GDSA_NEG_INF);
                sc[j] = v;
                if (v > mx) mx = v;
            }
            /* fp32 softmax (contiguous pairwise sum) -> bf16 */
            float sum = 0.0f;
            for (size_t j = 0; j < n; j++)
                e[j] = expf(sc[j] - mx);
            sum = apus_gmhc_pw_sum(e, n);
            uint16_t *pr = probs ? probs + (t * H + h) * n : NULL;
            for (size_t j = 0; j < n; j++)
                e[j] = e[j] / sum;
            /* o = bf16(p @ v): ascending kv positions, +0.0f init */
            const uint16_t *vc = st->v_cache + h * st->cap * vd;
            for (size_t dv = 0; dv < vd; dv++) {
                float acc = 0.0f;
                for (size_t j = 0; j < n; j++) {
                    float pj = apus_bf16_round(e[j]);
                    acc += pj * apus_bf16_f32(vc[j * vd + dv]);
                }
                out[t * H * vd + h * vd + dv] = apus_bf16_bits(acc);
            }
            if (pr)
                for (size_t j = 0; j < n; j++)
                    pr[j] = apus_bf16_bits(e[j]);
        }
    }
    free(xf);
    free(vis);
    free(e);
    free(sc);
    free(scb);
    free(qh);
}

/* FP8 linear: dequant -> bf16 -> bf16 GEMM (the m3g bitwise path).
 * M7b GLM Metal hook: the fused gfp8blk_gemm shader produces BITWISE the
 * dequant+GEMM composition (c/backend_gmetal.h); per-op fail-soft.
 * P5: the dequant runs on the pool (apus_fp8blk_dequant_mt — elementwise,
 * bitwise at every APUS_THREADS; compute-thread call sites only, the
 * gcache I/O workers keep the single-thread dispatch). */
static void apus_gdsa_fp8_linear(const uint8_t *codes, const float *scales,
                                 const uint16_t *x, float *xf, uint16_t *y,
                                 uint16_t *wbuf, size_t s, size_t O,
                                 size_t K) {
    if (apus_gmetal_hooks.fp8blk_linear
        && apus_gmetal_hooks.fp8blk_linear(codes, scales, x, y, s, O, K) == 0)
        return;
    apus_fp8blk_dequant_mt(codes, scales, wbuf, O, K);
    apus_bf16_gemm_mt(wbuf, x, xf, y, s, O, K);
}

void apus_gdsa_scratch_free(ApusGdsaScratch *sc) {
    if (!sc) return;
    free(sc->wbuf);
    sc->wbuf = NULL;
    sc->wbuf_cap = 0;
}

void apus_gdsa_forward(const ApusGdsaW *W, const uint16_t *x, size_t s,
                       ApusGdsaState *st, uint16_t *out,
                       ApusGdsaInterm *im) {
    apus_gdsa_forward2(W, x, s, st, out, im, NULL);
}

void apus_gdsa_forward2(const ApusGdsaW *W, const uint16_t *x, size_t s,
                        ApusGdsaState *st, uint16_t *out,
                        ApusGdsaInterm *im, const int32_t *forced_topk) {
    apus_gdsa_forward3(W, x, s, st, out, im, forced_topk, NULL);
}

void apus_gdsa_forward3(const ApusGdsaW *W, const uint16_t *x, size_t s,
                        ApusGdsaState *st, uint16_t *out,
                        ApusGdsaInterm *im, const int32_t *forced_topk,
                        ApusGdsaScratch *sc) {
    size_t dim = W->dim, H = W->H, qd = W->qd, vd = W->vd;
    size_t ql = W->ql, kl = W->kl;
    size_t base = st->n;
    size_t width = apus_gdsa_width(W);

    /* wbuf must cover the largest FP8 projection's O*K exactly: q_a
     * [ql,dim], q_b [H*qd,ql], kv_a [kl,dim], o_proj [dim,H*vd]. (The M4b
     * code allocated omax*xmax — 4x the true need at the real shapes.) */
    size_t wneed = ql * dim;
    if (H * qd * ql > wneed) wneed = H * qd * ql;
    if (kl * dim > wneed) wneed = kl * dim;
    if (dim * H * vd > wneed) wneed = dim * H * vd;
    size_t xmax = dim;
    if (ql > xmax) xmax = ql;
    if (kl > xmax) xmax = kl;
    if (H * vd > xmax) xmax = H * vd;
    uint16_t *wbuf;
    if (sc) {
        if (sc->wbuf_cap < wneed) {
            free(sc->wbuf);
            sc->wbuf = (uint16_t *)malloc(wneed * sizeof(uint16_t));
            sc->wbuf_cap = sc->wbuf ? wneed : 0;
        }
        wbuf = sc->wbuf;
    } else {
        wbuf = (uint16_t *)malloc(wneed * sizeof(uint16_t));
    }
    float *xf = (float *)malloc(s * xmax * sizeof(float));

    uint16_t *qa = (uint16_t *)malloc(s * ql * sizeof(uint16_t));
    uint16_t *qres = (uint16_t *)malloc(s * ql * sizeof(uint16_t));
    uint16_t *q = (uint16_t *)malloc(s * H * qd * sizeof(uint16_t));
    uint16_t *kv = (uint16_t *)malloc(s * kl * sizeof(uint16_t));
    uint16_t *kpass = (uint16_t *)malloc(s * kl * sizeof(uint16_t));
    uint16_t *kvb = (uint16_t *)malloc(s * H * (qd + vd)
                                       * sizeof(uint16_t));
    uint16_t *attn = (uint16_t *)malloc(s * H * vd * sizeof(uint16_t));
    int32_t *topk = (int32_t *)malloc(s * width * sizeof(int32_t));

    /* q path (glm5:1168-1169) */
    apus_gdsa_fp8_linear(W->q_a_c, W->q_a_s, x, xf, qa, wbuf, s, ql, dim);
    for (size_t t = 0; t < s; t++)
        apus_gdsa_rmsnorm(qa + t * ql, W->q_a_norm, qres + t * ql, ql,
                          W->eps);
    apus_gdsa_fp8_linear(W->q_b_c, W->q_b_s, qres, xf, q, wbuf, s,
                         H * qd, ql);
    /* kv path (glm5:1171-1173, 1146-1147) */
    apus_gdsa_fp8_linear(W->kv_a_c, W->kv_a_s, x, xf, kv, wbuf, s, kl,
                         dim);
    for (size_t t = 0; t < s; t++)
        apus_gdsa_rmsnorm(kv + t * kl, W->kv_a_norm, kpass + t * kl, kl,
                          W->eps);
    apus_bf16_gemm_mt(W->kv_b, kpass, xf, kvb, s, H * (qd + vd), kl);
    /* append the expanded kv (head-major cache layout). glm5:1147
     * (expand_kv): kvb is [H, qd+vd] per token — the k/v split is PER
     * HEAD, interleaved [k|v] (k at h*(qd+vd), v at h*(qd+vd)+qd), NOT
     * head-major [all k | all v] (the pre-anchor layout; caught by
     * tests/m0/check_vs_hf.py, 2026-09-05 — the gate-7 divergence fix,
     * mirrored from tools/oracle.py dsa_forward). */
    for (size_t t = 0; t < s; t++)
        for (size_t h = 0; h < H; h++) {
            memcpy(st->k_cache + h * st->cap * qd + (base + t) * qd,
                   kvb + t * H * (qd + vd) + h * (qd + vd),
                   qd * sizeof(uint16_t));
            memcpy(st->v_cache + h * st->cap * vd + (base + t) * vd,
                   kvb + t * H * (qd + vd) + h * (qd + vd) + qd,
                   vd * sizeof(uint16_t));
        }
    if (im && im->q_resid)
        memcpy(im->q_resid, qres, s * ql * sizeof(uint16_t));
    if (im && im->q)
        memcpy(im->q, q, s * H * qd * sizeof(uint16_t));
    if (im && im->k_new)
        for (size_t t = 0; t < s; t++)
            for (size_t h = 0; h < H; h++)
                memcpy(im->k_new + t * H * qd + h * qd,
                       kvb + t * H * (qd + vd) + h * (qd + vd),
                       qd * sizeof(uint16_t));
    if (im && im->v_new)
        for (size_t t = 0; t < s; t++)
            for (size_t h = 0; h < H; h++)
                memcpy(im->v_new + t * H * vd + h * vd,
                       kvb + t * H * (qd + vd) + h * (qd + vd) + qd,
                       vd * sizeof(uint16_t));

    /* indexer (glm5:1182-1188) — appends its own caches */
    apus_gdsa_indexer(W, x, qres, s, st, topk,
                      im ? im->idx_scores : NULL,
                      im ? im->idx_knew : NULL,
                      im ? im->idx_gnew : NULL);
    if (im && im->idx_topk)
        memcpy(im->idx_topk, topk, s * width * sizeof(int32_t));
    st->n = base + s;
    /* teacher-forcing (tests/m5g tolerance tier): replace the selection
     * consumed by the attention; the caches above are unaffected. */
    if (forced_topk)
        memcpy(topk, forced_topk, s * width * sizeof(int32_t));
    /* attention + o_proj (glm5:1200-1216) */
    apus_gdsa_attention(W, q, st, s, topk, attn, im ? im->probs : NULL);
    if (im && im->attn)
        memcpy(im->attn, attn, s * H * vd * sizeof(uint16_t));
    apus_gdsa_fp8_linear(W->o_c, W->o_s, attn, xf, out, wbuf, s, dim,
                         H * vd);

    free(topk);
    free(attn);
    free(kvb);
    free(kpass);
    free(kv);
    free(q);
    free(qres);
    free(qa);
    free(xf);
    if (!sc)
        free(wbuf);     /* scratch-owned otherwise */
}

void apus_gdsa_decode_batch(const ApusGdsaW *W, const uint16_t *x, size_t s,
                            ApusGdsaState *st, uint16_t *out,
                            const int32_t *forced_topk) {
    apus_gdsa_decode_batch2(W, x, s, st, out, forced_topk, NULL);
}

void apus_gdsa_decode_batch2(const ApusGdsaW *W, const uint16_t *x, size_t s,
                             ApusGdsaState *st, uint16_t *out,
                             const int32_t *forced_topk,
                             ApusGdsaScratch *sc) {
    const size_t width = apus_gdsa_width(W);
    for (size_t t = 0; t < s; t++)
        apus_gdsa_forward3(W, x + t * W->dim, 1, st, out + t * W->dim, NULL,
                           forced_topk ? forced_topk + t * width : NULL, sc);
}

#endif /* APUS_GDSA_IMPLEMENTATION */
#endif /* APUS_GDSA_H */
