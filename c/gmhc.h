/*
 * c/gmhc.h — GLM-5.3-Flash mHC (Manifold-Constrained Hyper-Connections)
 * residual stream (M4a). C11, libc only. Scalar reference paths (SIMD/mt
 * are M7 perf work — the bitwise gate pins the semantics first).
 *
 * Normative reference: tools/oracle.py hc_pre/hc_post/hc_split_sinkhorn +
 * the HyperHead mean in model_forward (f32 mode), porting
 * reference/inference/modeling_glm5_next.py (glm5:210-216 unweighted
 * RMSNorm, 267-295 HyperConnection, 1317-1328 layer wiring, 1494 head).
 * The Sinkhorn-20 recipe is IDENTICAL to the base engine's c/mhc.h
 * (Phase A), but two GLM differences are bitwise-sensitive vs c/mhc.h:
 *
 *   1. NORM-BEFORE-FN: the flattened streams are normalized by the
 *      UNWEIGHTED RMSNorm (eps = rms_norm_eps = 1e-5, fp32, no weight, no
 *      bf16 rounding — glm5:278) BEFORE the fn matmul (glm5:279). The base
 *      engine multiplied by rsqrt AFTER the matmul (norm_eps 1e-6).
 *      Mathematically equal, FP32-rounding different; GLM order here.
 *   2. hc_head is an UNWEIGHTED MEAN over the stream axis (glm5:1494) —
 *      no fn/scale/base, no sigmoid, no Sinkhorn (unlike the base).
 *
 * Per-token semantics (x4 [n, d] flattened to [n*d], n = hc_mult = 4,
 * all map math fp32 end-to-end — the maps are NEVER bf16-rounded):
 *
 *   maps (hc_pre, glm5:278-290):
 *       flat[i] = x[i] * (1/sqrt(mean(x^2) + norm_eps))   (unweighted)
 *       mixes   = flat @ fn^T          (fp32, sequential-k, NO rounding)
 *       pre[j]    = sigmoid(mixes[j]     * scale[0] + base[j])     + hc_eps
 *       post[j]   = 2 * sigmoid(mixes[n+j]   * scale[1] + base[n+j])
 *       comb[j,k] = mixes[2n + j*n+k] * scale[2] + base[2n + j*n+k]
 *       Sinkhorn-20 on comb (see below)
 *   collapse (hc_pre output, glm5:294): fp32 weighted sum over the stream
 *       axis, ONE bf16 rounding at the end:
 *       y[i] = bf16( sum_j pre[j] * x4[j,i] )
 *   expand (hc_post, glm5:1317-1319): post and comb are CAST TO BF16
 *       FIRST, then
 *       t1[j,i] = bf16( bf16(post[j]) * x[i] )
 *       t2[j,i] = bf16( sum_k bf16(comb[k,j]) * res[k,i] )   (bf16 matmul,
 *                 fp32 accumulate, comb indexed [residual k][output j])
 *       y[j,i]  = bf16( t1[j,i] + t2[j,i] )
 *   head (glm5:1494): y[i] = bf16( mean_j x4[j,i] )
 *
 * Sinkhorn (oracle hc_split_sinkhorn, glm5:286-290) — EXACT order and eps
 * placement (hc_eps = 1e-6, iters = 20), same recipe as c/mhc.h:
 *   1. row softmax (max-subtracted), then PER-ELEMENT + eps;
 *   2. column normalize with eps IN THE DENOMINATOR;
 *   3. 19 x (row normalize /(rowsum+eps); column normalize /(colsum+eps)).
 * The last op is a column normalization: comb is column-stochastic to
 * ~1e-6, row sums deviate — do not "fix".
 *
 * Bitwise contract vs the numpy oracle f32 mode (tests/m4g):
 *   - The sigmoid replicates numpy's numerically-stable form
 *     (oracle.py sigmoid): e = expf(-|x|); x >= 0 ? 1/(1+e) : e/(1+e).
 *     This is NOT the naive 1/(1+expf(-x)) for x < 0 (different rounding).
 *   - Reductions over numpy axes replicate numpy's pairwise summation
 *     order EXACTLY (apus_gmhc_pw_sum: n < 8 sequential from a[0];
 *     n <= 128 eight-accumulator tree; larger n recursive halving to
 *     multiples of 8). Used for the RMSNorm mean (n = n*d) and every
 *     sinkhorn sum (n = 4). Verified bitwise against numpy 2.5.2.
 *   - Matmul-style accumulations (fn matmul, the comb^T@residual expand
 *     matmul) follow the oracle's _mm convention: acc initialized to
 *     +0.0f, strictly ascending k, mul+add two roundings, no FMA
 *     (-ffp-contract=off pinned). The collapse/head stream sums follow
 *     numpy's pairwise order instead (n = 4: start from the FIRST
 *     product, then sequential) — the init difference matters only for
 *     signed zeros and is replicated deliberately.
 *   - expf == numpy float32 exp is a HOST property (verified bitwise on
 *     macOS arm64 and on Linux/x86_64 with numpy pinned to its baseline
 *     exp kernel — tests/m4g probes it at runtime and documents the
 *     tolerance fallback; see tests/m4g/README.md).
 *
 * Usage: #define APUS_GMHC_IMPLEMENTATION in exactly one TU (the
 * implementation section instantiates c/num.h's helpers — guarded, so
 * combining with APUS_NUM_IMPLEMENTATION / APUS_BF16_IMPLEMENTATION in
 * the same TU is safe). Self-contained (c/num.h only).
 */
#ifndef APUS_GMHC_H
#define APUS_GMHC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_GMHC_MULT 4u   /* hc_mult in GLM-5.3-Flash (kernels n-generic) */

/* numpy-stable sigmoid: e = expf(-|x|); x >= 0 ? 1/(1+e) : e/(1+e). */
float apus_gmhc_sigmoid(float x);

/* numpy pairwise_sum order replica (float32): the exact accumulation
 * order of numpy's .sum()/.mean() over a contiguous axis. */
float apus_gmhc_pw_sum(const float *a, size_t n);

/* Unweighted RMSNorm (glm5:210-216): out[i] = x[i] * (1/sqrt(mean(x^2)
 * + eps)), fp32 in/out, NO weight, NO bf16 rounding. out may alias x. */
void apus_gmhc_unweighted_rmsnorm(const float *x, float *out, size_t n,
                                  float eps);

/* Sinkhorn steps on a row-major n x n matrix (exposed individually so the
 * test can verify the reference iteration-for-iteration):
 * row softmax (max-subtracted) then per-element +eps. */
void apus_gmhc_row_softmax_eps(float *c, size_t n, float eps);
/* c[j,k] /= rowsum[j] + eps (eps added to the sum). */
void apus_gmhc_norm_rows_eps(float *c, size_t n, float eps);
/* c[j,k] /= colsum[k] + eps (eps added to the sum). */
void apus_gmhc_norm_cols_eps(float *c, size_t n, float eps);
/* Full driver: row softmax+eps; col norm; (iters-1) x (row norm; col norm). */
void apus_gmhc_sinkhorn(float *c, size_t n, int iters, float eps);

/* Per-token maps (hc_pre minus the collapse): x4 [n*d] fp32 (bf16-valued),
 * fn [(2+n)*n, n*d] row-major fp32 (bf16-valued), scale [3], base [(2+n)*n].
 * norm_eps = rms_norm_eps (1e-5), hc_eps = 1e-6, iters = 20.
 * Outputs: pre [n], post [n], comb [n*n] (post-Sinkhorn), all fp32.
 * mixes: scratch [(2+n)*n], may be NULL for n == APUS_GMHC_MULT. */
void apus_gmhc_maps(const float *x4, size_t d, size_t n,
                    const float *fn, const float *scale, const float *base,
                    float norm_eps, float hc_eps, int iters,
                    float *pre, float *post, float *comb, float *mixes);

/* Collapse (hc_pre output): y[i] = bf16( sum_j pre[j] * x4[j*d+i] ) as
 * BF16 codes. The stream sum follows numpy's pairwise order (n < 8: from
 * the first product, sequential). */
void apus_gmhc_collapse(const float *x4, const float *pre,
                        uint16_t *y, size_t d, size_t n);

/* Expand (hc_post): x [d] and res [n*d] BF16 codes; post [n], comb [n*n]
 * fp32 (rounded to bf16 inside, reference order); y4 [n*d] BF16 codes.
 * y[j,i] = bf16( bf16(bf16(post[j])*x[i]) + bf16(sum_k bf16(comb[k,j])
 *          * res[k,i]) ) — the comb matmul accumulates from +0.0f in
 * ascending k (oracle _mm). */
void apus_gmhc_expand(const uint16_t *x, const uint16_t *res,
                      const float *post, const float *comb,
                      uint16_t *y4, size_t d, size_t n);

/* HyperHead (glm5:1494): y[i] = bf16( mean_j x4[j*d+i] ), unweighted,
 * no maps. x4 [n*d] BF16 codes, y [d] BF16 codes. */
void apus_gmhc_head(const uint16_t *x4, uint16_t *y, size_t d, size_t n);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GMHC_IMPLEMENTATION) && !defined(APUS_GMHC_IMPL_INCLUDED)
#define APUS_GMHC_IMPL_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Instantiate the shared scalar numerics helpers (guarded). */
#define APUS_NUM_IMPLEMENTATION
#include "num.h"

float apus_gmhc_sigmoid(float x) {
    float e = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

/* numpy pairwise_sum (float32): n < 8 sequential from a[0]; n <= 128 an
 * 8-accumulator unrolled block + tree combine + sequential remainder;
 * larger n recurses on halves rounded down to a multiple of 8. */
float apus_gmhc_pw_sum(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        float r = a[0];
        for (size_t i = 1; i < n; i++) r += a[i];
        return r;
    }
    if (n <= 128) {
        float r[8];
        size_t i;
        for (int j = 0; j < 8; j++) r[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + (size_t)j];
        float res = ((r[0] + r[1]) + (r[2] + r[3]))
                  + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    size_t n2 = (n / 2) & ~(size_t)7;
    return apus_gmhc_pw_sum(a, n2) + apus_gmhc_pw_sum(a + n2, n - n2);
}

void apus_gmhc_unweighted_rmsnorm(const float *x, float *out, size_t n,
                                  float eps) {
    float *sq = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) sq[i] = x[i] * x[i];
    float var = apus_gmhc_pw_sum(sq, n) / (float)n;
    free(sq);
    float inv = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * inv;
}

void apus_gmhc_row_softmax_eps(float *c, size_t n, float eps) {
    for (size_t j = 0; j < n; j++) {
        float *r = c + j * n;
        float mx = r[0];
        for (size_t k = 1; k < n; k++) if (r[k] > mx) mx = r[k];
        for (size_t k = 0; k < n; k++) r[k] = expf(r[k] - mx);
        float sum = apus_gmhc_pw_sum(r, n);
        for (size_t k = 0; k < n; k++) r[k] = r[k] / sum + eps;
    }
}

void apus_gmhc_norm_rows_eps(float *c, size_t n, float eps) {
    for (size_t j = 0; j < n; j++) {
        float *r = c + j * n;
        float den = apus_gmhc_pw_sum(r, n) + eps;
        for (size_t k = 0; k < n; k++) r[k] /= den;
    }
}

void apus_gmhc_norm_cols_eps(float *c, size_t n, float eps) {
    for (size_t k = 0; k < n; k++) {
        float sum = c[k];
        for (size_t j = 1; j < n; j++) sum += c[j * n + k];
        float den = sum + eps;
        for (size_t j = 0; j < n; j++) c[j * n + k] /= den;
    }
}

void apus_gmhc_sinkhorn(float *c, size_t n, int iters, float eps) {
    apus_gmhc_row_softmax_eps(c, n, eps);
    apus_gmhc_norm_cols_eps(c, n, eps);
    for (int it = 1; it < iters; it++) {
        apus_gmhc_norm_rows_eps(c, n, eps);
        apus_gmhc_norm_cols_eps(c, n, eps);
    }
}

void apus_gmhc_maps(const float *x4, size_t d, size_t n,
                    const float *fn, const float *scale, const float *base,
                    float norm_eps, float hc_eps, int iters,
                    float *pre, float *post, float *comb, float *mixes) {
    size_t nmix = (2 + n) * n, nx = n * d;
    float stack_mixes[(2 + APUS_GMHC_MULT) * APUS_GMHC_MULT];
    if (!mixes) mixes = stack_mixes;   /* caller passes a buffer for n != 4 */
    /* norm BEFORE the fn matmul (GLM order; the base applied rsqrt after) */
    float *flat = (float *)malloc(nx * sizeof(float));
    apus_gmhc_unweighted_rmsnorm(x4, flat, nx, norm_eps);
    /* mixes = flat @ fn^T: fp32, sequential ascending k, +0.0f init,
     * mul+add two roundings (oracle _mm order), no output rounding */
    for (size_t j = 0; j < nmix; j++) {
        const float *f = fn + j * nx;
        float acc = 0.0f;
        for (size_t i = 0; i < nx; i++) acc += f[i] * flat[i];
        mixes[j] = acc;
    }
    free(flat);
    /* gates + split (mix layout [pre(n) | post(n) | comb(n*n)]) */
    for (size_t j = 0; j < n; j++)
        pre[j] = apus_gmhc_sigmoid(mixes[j] * scale[0] + base[j]) + hc_eps;
    for (size_t j = 0; j < n; j++)
        post[j] = 2.0f * apus_gmhc_sigmoid(mixes[n + j] * scale[1]
                                           + base[n + j]);
    for (size_t j = 0; j < n; j++)
        for (size_t k = 0; k < n; k++) {
            size_t idx = 2 * n + j * n + k;
            comb[j * n + k] = mixes[idx] * scale[2] + base[idx];
        }
    apus_gmhc_sinkhorn(comb, n, iters, hc_eps);
}

void apus_gmhc_collapse(const float *x4, const float *pre,
                        uint16_t *y, size_t d, size_t n) {
    /* numpy pairwise order over the stream axis; n < 8 (GLM n = 4) is
     * sequential from the FIRST product (not from +0.0f). */
    for (size_t i = 0; i < d; i++) {
        float acc = pre[0] * x4[i];
        for (size_t j = 1; j < n; j++) acc += pre[j] * x4[j * d + i];
        y[i] = apus_bf16_bits(acc);
    }
}

void apus_gmhc_expand(const uint16_t *x, const uint16_t *res,
                      const float *post, const float *comb,
                      uint16_t *y4, size_t d, size_t n) {
    for (size_t j = 0; j < n; j++) {
        uint16_t *y = y4 + j * d;
        float pb = apus_bf16_round(post[j]);           /* cast to bf16 first */
        for (size_t i = 0; i < d; i++) {
            float t1 = apus_bf16_round(pb * apus_bf16_f32(x[i]));
            float acc = 0.0f;                          /* oracle _mm init */
            for (size_t k = 0; k < n; k++)
                acc += apus_bf16_round(comb[k * n + j])
                     * apus_bf16_f32(res[k * d + i]);
            float t2 = apus_bf16_round(acc);
            y[i] = apus_bf16_bits(t1 + t2);
        }
    }
}

void apus_gmhc_head(const uint16_t *x4, uint16_t *y, size_t d, size_t n) {
    /* Unweighted mean over the stream axis (numpy pairwise order; n = 4:
     * sequential from the first stream), ONE bf16 rounding. */
    for (size_t i = 0; i < d; i++) {
        float acc = apus_bf16_f32(x4[i]);
        for (size_t j = 1; j < n; j++) acc += apus_bf16_f32(x4[j * d + i]);
        y[i] = apus_bf16_bits(acc / (float)n);
    }
}

#endif /* APUS_GMHC_IMPLEMENTATION */
#endif /* APUS_GMHC_H */
