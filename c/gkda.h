/*
 * c/gkda.h — GLM-5.3-Flash KDA (Kimi Delta Attention) linear-attention
 * sublayer (M4b): fused causal conv k=4 + SiLU, low-rank forget gate,
 * beta gate, the CHUNKED prefill core AND the RECURRENT decode core (the
 * M0 KDA ORDERING CONTRACT — two separately verified orderings, neither
 * may "simplify" into the other), gated o_norm, o_proj. C11, libc only
 * (+ c/bf16.h kernels). Scalar reference paths (SIMD/mt beyond the m3g
 * GEMMs is M7 perf work — it must preserve these orders bitwise).
 *
 * Normative reference: tools/oracle.py kda_* (f32 mode), porting
 * reference/inference/modeling_glm5_next.py (glm5:305-734). Ported across
 * the adapter seam from ../Apus-Ling-3.0-Flash-bf16 c/kda.h (attribution:
 * that repo's M4a per-op kernel discipline) and CORRECTED against the
 * GLM oracle — the donor predates this adapter's pinned numerics. Donor
 * divergences fixed here (each one is bitwise-visible; tests/m4h
 * README.md lists them):
 *
 *   1. l2norm output stays FP32 (the donor rounded it back to bf16); the
 *      reference casts q/k to fp32 FIRST and never rounds (glm5:498-505,
 *      445-447). The l2norm sum-of-squares follows numpy's contiguous
 *      pairwise order (the donor summed sequentially).
 *   2. Every sigmoid/silu uses numpy's numerically-stable form
 *      (apus_gmhc_sigmoid: e = expf(-|x|), branch on sign) — the donor's
 *      naive 1/(1+expf(-x)) rounds differently for x < 0.
 *   3. o_norm eps is rms_norm_eps = 1e-5 (glm5:602,623 — the KDA layer
 *      passes config.rms_norm_eps to Glm5NextTextRMSNormGated), NOT the
 *      donor's 1e-6; the variance mean is a numpy pairwise mean, not a
 *      sequential sum.
 *   4. beta is BF16-ROUNDED after the sigmoid (glm5:698, oracle _B); the
 *      donor kept it fp32.
 *   5. The rank-1 state update multiplies as k * ((v - kv_mem) * beta)
 *      (glm5:474-476 — delta computed first, then the outer product);
 *      the donor computed (beta * k) * u — one rounding earlier, bits
 *      differ.
 *   6. The donor has NO chunked prefill at all (recurrent only, chunk
 *      mode "out of scope"); GLM pins the chunked path for prefill
 *      (docs/ARCHITECTURE.md §4).
 *
 * Semantics (f32-faithful; tests/m0/README.md is the pin):
 *
 *   Conv (glm5:394-414 prefill / 375-391 decode; kept-fp32 module, BF16
 *   storage): per token t, channel c:
 *       acc = 0; for i ascending in 0..k-1: acc += f32(w[c,i]) * win[i]
 *       out[c,t] = bf16( acc * sigmoid_stable(acc) )     (ONE rounding)
 *     win = the last k inputs (zero where t-k+i < 0); the decode step is
 *     the identical per-token body (bitwise == prefill token for token).
 *     Conv state holds BF16 values: the last k-1 PRE-conv inputs per
 *     channel. Engine layout: channel-major state[c*(k-1) + j], j = 0
 *     oldest (the oracle's [k-1, C] token-major layout carries the same
 *     bf16 values; layout is an engine choice).
 *   Forget gate (glm5:305-335, safe-gate path, lower_bound = -5):
 *       fg = bf16(f_b @ bf16(f_a @ x))          (two bf16 matmuls)
 *       g  = -5 * sigmoid_stable( expf(A_log[h]) * (f32(fg) + dt_bias) )
 *     fp32 end-to-end after fg, NEVER bf16-rounded.
 *   Beta (glm5:698): beta = bf16( sigmoid_stable( bf16(b_proj @ x) ) ).
 *   Core, BOTH orderings fp32 (state S [H,D,D] FP32, caller zero-inits):
 *     l2norm (eps 1e-6) AFTER the fp32 cast, BEFORE the D^-0.5 q-scale
 *     (q only): ss = pairwise_sum(x^2); y = x / sqrtf(ss + 1e-6f)
 *     (literal division). numpy reduction orders (verified bitwise
 *     against numpy 2.5.2, tests/m4h README): CONTIGUOUS-axis sums use
 *     numpy pairwise (apus_gmhc_pw_sum); STRIDED-axis sums are PLAIN
 *     SEQUENTIAL from +0.0f; matmul contractions are the oracle _mm
 *     order (acc = +0.0f, ascending k, mul+add two roundings, no FMA).
 *     RECURRENT (decode, glm5:428-479), per token, per head:
 *       S[i,j] *= expf(g[i])
 *       kv[j]   = sum_i S[i,j]*k[i]          (strided: sequential +0.0f)
 *       dl[j]   = (v[j] - kv[j]) * beta
 *       S[i,j] += k[i] * dl[j]
 *       o[j]    = sum_i S[i,j]*q[i]          (strided: sequential +0.0f)
 *     CHUNKED (prefill, glm5:483-579, chunk 64, zero right-padding; the
 *     pad is benign — g is cumsum'ed WITHIN each chunk so padded rows
 *     inherit the last real row's cumulative decay): per chunk, per head:
 *       gc         = cumsum(g) within the chunk (sequential)
 *       decay[i,j,d] = expf(gc[i,d] - gc[j,d])   (j <= i only; the
 *                      reference computes the overflowing j > i entries
 *                      and REPLACES them via masked_fill — the masked
 *                      entries are exactly +0.0f either way, so they are
 *                      never computed here)
 *       attn[i,j]  = -sum_d (kb[i,d]*kk[j,d]) * decay[i,j,d]  (j < i)
 *       attn[i,:i] += attn[i,:i] @ attn[:i,:i]  (UT transform, rows in
 *                    ascending i; each dot sequential ascending m +0.0f)
 *       attn      += I
 *       value      = attn @ v_beta
 *       kcd        = attn @ (k_beta * exp(gc))
 *       inter      = (q * exp(gc)) @ S
 *       intra[i,j] = sum_d (q[i,d]*kk[j,d]) * decay[i,j,d]   (j <= i)
 *       v_new      = value - kcd @ S
 *       out        = inter + intra @ v_new
 *       S[m,n]     = S[m,n]*expf(gc[last,m])
 *                    + sum_j (kk[j,m]*expf(gc[last,m] - gc[j,m]))
 *                      * v_new[j,n]
 *     Core output rounds to bf16 ONCE (oracle _B at glm5:576/478); the
 *     state stays fp32 unrounded.
 *   o_norm (glm5:339-359, 623: eps = rms_norm_eps = 1e-5): strict fp32,
 *     per head: var = pairwise_mean(core^2); y = (core * (1/sqrtf(var+eps)))
 *     then * w then * sigmoid_stable(gate); ONE bf16 rounding at the end.
 *   Projections are bf16 matmuls (m3g apus_bf16_gemm_mt — bitwise == the
 *   oracle _mm at every APUS_THREADS).
 *
 * Usage: #define APUS_GKDA_IMPLEMENTATION in exactly one TU. Needs
 * c/bf16.h (APUS_BF16_IMPLEMENTATION) and c/gmhc.h
 * (APUS_GMHC_IMPLEMENTATION — the numpy-stable sigmoid and the
 * pairwise-sum replica) linked in the same binary.
 */
#ifndef APUS_GKDA_H
#define APUS_GKDA_H

#include <stddef.h>
#include <stdint.h>

#include "bf16.h"
#include "gmhc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Conv kernel width (linear_conv_kernel_dim) and chunk size: part of the
 * contract (glm5:394, 511). */
#define APUS_GKDA_CONV_K 4u
#define APUS_GKDA_CHUNK 64u
/* Norm epsilons: fla l2norm 1e-6 (glm5:420); o_norm eps = rms_norm_eps
 * (glm5:602,623) — 1e-5 in GLM-5.3-Flash. */
#define APUS_GKDA_L2NORM_EPS 1e-6f

/* (a) Depthwise causal conv + SiLU. x/out: [T,C] / [C] BF16 codes;
 * w: [C, APUS_GKDA_CONV_K] codes; state: [C*(K-1)] codes in/out
 * (channel-major, j = 0 oldest; caller zero-inits). The decode step is
 * bitwise == the per-token body of the prefill loop. */
void apus_gkda_conv_prefill(const uint16_t *x, const uint16_t *w,
                            uint16_t *out, size_t C, size_t T,
                            uint16_t *state);
void apus_gkda_conv_step(const uint16_t *x, const uint16_t *w,
                         uint16_t *out, size_t C, uint16_t *state);

/* (b) CHUNKED prefill core (oracle kda_chunk, glm5:483-579). q/k/v:
 * [s, H*D] BF16 codes (post-conv); g: [s, H*D] fp32 (forget gate);
 * beta: [s, H] BF16 codes; S: [H*D*D] fp32 in/out (caller zero-inits);
 * out: [s, H*D] BF16 codes. */
void apus_gkda_chunk(const uint16_t *q, const uint16_t *k,
                     const uint16_t *v, const float *g,
                     const uint16_t *beta, float *S, uint16_t *out,
                     size_t s, size_t H, size_t D);

/* (b) RECURRENT decode core (oracle kda_recurrent, glm5:428-479), one or
 * more tokens stepped through the same state. Same signature/semantics
 * as apus_gkda_chunk; BITWISE the oracle's per-token loop. */
void apus_gkda_recurrent(const uint16_t *q, const uint16_t *k,
                         const uint16_t *v, const float *g,
                         const uint16_t *beta, float *S, uint16_t *out,
                         size_t s, size_t H, size_t D);

/* (c) Gated o_norm (glm5:339-359, eps = rms_norm_eps): core/gate/w/y
 * [s*H*D]; core/gate BF16 codes (core is the bf16-rounded core output),
 * w [D] codes, y BF16 codes. */
void apus_gkda_onorm(const uint16_t *core, const uint16_t *gate,
                     const uint16_t *w, uint16_t *y,
                     size_t s, size_t H, size_t D, float eps);

/* --- composed sublayer (oracle kda_forward, glm5:628-734) -------------*/

typedef struct {
    size_t dim;                 /* hidden size */
    size_t H, D;                /* linear_num_heads, linear_head_dim */
    size_t Dr;                  /* low-rank width (f_a/g_a output) */
    const uint16_t *q_w, *k_w, *v_w;  /* [H*D, dim] BF16 codes */
    const uint16_t *conv_w;           /* [3*H*D, APUS_GKDA_CONV_K] codes */
    const uint16_t *f_a;              /* [Dr, dim] codes */
    const uint16_t *f_b;              /* [H*D, Dr] codes */
    const float *A_log;               /* [H] F32 */
    const float *dt_bias;             /* [H*D] F32 */
    const uint16_t *b_w;              /* [H, dim] codes */
    const uint16_t *g_a;              /* [Dr, dim] codes */
    const uint16_t *g_b;              /* [H*D, Dr] codes */
    const uint16_t *o_norm;           /* [D] codes */
    const uint16_t *o_w;              /* [dim, H*D] codes */
    float lower_bound;                /* gate_lower_bound (-5.0) */
    float eps;                        /* o_norm eps (rms_norm_eps, 1e-5) */
} ApusGkdaW;

typedef struct {
    uint16_t *conv_state;   /* [3*H*D][K-1] codes, channel-major */
    float *rec_state;       /* [H*D*D] fp32 (caller zero-inits) */
} ApusGkdaState;

/* Named intermediates (NULL to skip): mixed [s,3*H*D] codes (post-conv),
 * g [s,H*D] fp32, beta [s,H] codes, core/gate/onorm [s,H*D] codes. */
typedef struct {
    uint16_t *mixed;
    float *g;
    uint16_t *beta;
    uint16_t *core;
    uint16_t *gate;
    uint16_t *onorm;
} ApusGkdaInterm;

/* Full sublayer: x [s, dim] BF16 codes -> out [s, dim] BF16 codes.
 * decode != 0: RECURRENT ordering, any s >= 1 — a multi-token decode batch
 * (M8b, the speculative verify path) steps the conv window and the
 * recurrent state TOKEN BY TOKEN in exact sequential order (the conv
 * prefill loop IS the per-token body chained — apus_gkda_conv_prefill is
 * literally a loop of apus_gkda_conv_step — and apus_gkda_recurrent chains
 * s tokens through the same state), so the batch is bitwise == s
 * one-by-one decode calls by construction. decode == 0: chunked prefill
 * ordering (any s >= 1). States mutated in place. */
void apus_gkda_forward(const ApusGkdaW *W, const uint16_t *x, size_t s,
                       ApusGkdaState *st, int decode, uint16_t *out,
                       ApusGkdaInterm *im);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GKDA_IMPLEMENTATION) && !defined(APUS_GKDA_IMPL_INCLUDED)
#define APUS_GKDA_IMPL_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* --- (a) conv ------------------------------------------------------------*/

/* One conv token, the shared body (decode == prefill per-token, bitwise).
 * win[0..K-1] are the BF16 window codes, win[K-1] the current input. */
static inline uint16_t apus_gkda_conv_tok(const uint16_t *wc,
                                          const uint16_t *win) {
    float acc = 0.0f;
    for (size_t i = 0; i < APUS_GKDA_CONV_K; i++)
        acc += apus_bf16_f32(wc[i]) * apus_bf16_f32(win[i]);
    return apus_bf16_bits(acc * apus_gmhc_sigmoid(acc));
}

void apus_gkda_conv_step(const uint16_t *x, const uint16_t *w,
                         uint16_t *out, size_t C, uint16_t *state) {
    for (size_t c = 0; c < C; c++) {
        uint16_t *st = state + c * (APUS_GKDA_CONV_K - 1);
        uint16_t win[APUS_GKDA_CONV_K];
        uint16_t xc = x[c];                 /* x may alias out */
        for (size_t j = 0; j < APUS_GKDA_CONV_K - 1; j++)
            win[j] = st[j];
        win[APUS_GKDA_CONV_K - 1] = xc;
        out[c] = apus_gkda_conv_tok(w + c * APUS_GKDA_CONV_K, win);
        /* state = last K-1 pre-conv inputs, j = 0 oldest */
        for (size_t j = 0; j + 1 < APUS_GKDA_CONV_K - 1; j++)
            st[j] = st[j + 1];
        st[APUS_GKDA_CONV_K - 2] = xc;
    }
}

void apus_gkda_conv_prefill(const uint16_t *x, const uint16_t *w,
                            uint16_t *out, size_t C, size_t T,
                            uint16_t *state) {
    for (size_t t = 0; t < T; t++)
        apus_gkda_conv_step(x + t * C, w, out + t * C, C, state);
}

/* --- shared core helpers -------------------------------------------------*/

/* l2norm (glm5:417-425): x BF16 codes [D] widened exactly; y fp32 =
 * x / sqrtf(pairwise_sum(x^2) + eps). NO bf16 rounding of the output. */
static void apus_gkda_l2norm_row(const uint16_t *x, float *y, size_t D,
                                 float *sq) {
    for (size_t i = 0; i < D; i++) {
        float v = apus_bf16_f32(x[i]);
        y[i] = v;
        sq[i] = v * v;
    }
    float root = sqrtf(apus_gmhc_pw_sum(sq, D) + APUS_GKDA_L2NORM_EPS);
    for (size_t i = 0; i < D; i++)
        y[i] = y[i] / root;
}

/* --- (b) RECURRENT decode core (oracle kda_recurrent) ---------------------*/

/* One recurrent step for ONE head. S [D*D] fp32 in/out; q,k,v BF16 codes
 * [D]; g fp32 [D]; beta BF16 code; o BF16 code out [D]. */
static void apus_gkda_recurrent_head(float *Sh, const uint16_t *qh,
                                     const uint16_t *kh, const uint16_t *vh,
                                     const float *gh, uint16_t beta,
                                     uint16_t *oh, size_t D, float qscale) {
    float *qf = (float *)malloc(3 * D * sizeof(float));
    float *kf = qf + D, *vf = kf + D;
    float *sq = (float *)malloc(D * sizeof(float));
    apus_gkda_l2norm_row(qh, qf, D, sq);
    apus_gkda_l2norm_row(kh, kf, D, sq);
    for (size_t i = 0; i < D; i++) {
        qf[i] = qf[i] * qscale;
        vf[i] = apus_bf16_f32(vh[i]);
    }
    float *kvm = (float *)malloc(D * sizeof(float));
    float *dl = (float *)malloc(D * sizeof(float));
    float bt = apus_bf16_f32(beta);
    /* decay: S = S .* exp(g) (row i scaled by expf(g[i])) */
    for (size_t i = 0; i < D; i++) {
        float dec = expf(gh[i]);
        float *Si = Sh + i * D;
        for (size_t j = 0; j < D; j++)
            Si[j] = Si[j] * dec;
    }
    /* kv_mem = (S * k).sum(axis=-2): STRIDED sum -> plain sequential
     * from +0.0f (the numpy strided-reduce order, tests/m4h README). */
    for (size_t j = 0; j < D; j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < D; i++)
            acc += Sh[i * D + j] * kf[i];
        kvm[j] = acc;
    }
    /* delta = (v - kv_mem) * beta */
    for (size_t j = 0; j < D; j++)
        dl[j] = (vf[j] - kvm[j]) * bt;
    /* S += k outer delta */
    for (size_t i = 0; i < D; i++) {
        float ki = kf[i];
        float *Si = Sh + i * D;
        for (size_t j = 0; j < D; j++)
            Si[j] = Si[j] + ki * dl[j];
    }
    /* o = (S * q).sum(axis=-2): strided sequential, ONE bf16 rounding */
    for (size_t j = 0; j < D; j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < D; i++)
            acc += Sh[i * D + j] * qf[i];
        oh[j] = apus_bf16_bits(acc);
    }
    free(dl);
    free(kvm);
    free(sq);
    free(qf);
}

void apus_gkda_recurrent(const uint16_t *q, const uint16_t *k,
                         const uint16_t *v, const float *g,
                         const uint16_t *beta, float *S, uint16_t *out,
                         size_t s, size_t H, size_t D) {
    float qscale = (float)pow((double)D, -0.5);
    size_t hd = H * D;
    for (size_t t = 0; t < s; t++)
        for (size_t h = 0; h < H; h++)
            apus_gkda_recurrent_head(S + h * D * D,
                                     q + t * hd + h * D, k + t * hd + h * D,
                                     v + t * hd + h * D, g + t * hd + h * D,
                                     beta[t * H + h], out + t * hd + h * D,
                                     D, qscale);
}

/* --- (b) CHUNKED prefill core (oracle kda_chunk) --------------------------*/

void apus_gkda_chunk(const uint16_t *q, const uint16_t *k,
                     const uint16_t *v, const float *g,
                     const uint16_t *beta, float *S, uint16_t *out,
                     size_t s, size_t H, size_t D) {
    const size_t C = APUS_GKDA_CHUNK;
    float qscale = (float)pow((double)D, -0.5);
    size_t pad = (C - s % C) % C;
    size_t sp = s + pad, N = sp / C, hd = H * D;

    /* Per-head work buffers (reused across heads and chunks). */
    float *qp = (float *)malloc(sp * D * sizeof(float));
    float *kp = (float *)malloc(sp * D * sizeof(float));
    float *vp = (float *)malloc(sp * D * sizeof(float));
    float *gc = (float *)malloc(sp * D * sizeof(float));
    float *kb = (float *)malloc(sp * D * sizeof(float));  /* k_beta */
    float *vb = (float *)malloc(sp * D * sizeof(float));  /* v_beta */
    float *decay = (float *)malloc(C * C * D * sizeof(float));
    float *attn = (float *)malloc(C * C * sizeof(float));
    float *value = (float *)malloc(C * D * sizeof(float));
    float *kcd = (float *)malloc(C * D * sizeof(float));
    float *vnew = (float *)malloc(C * D * sizeof(float));
    float *intra = (float *)malloc(C * C * sizeof(float));
    float *sq = (float *)malloc(D * sizeof(float));

    for (size_t h = 0; h < H; h++) {
        /* Extract + widen the head, l2norm (fp32, no rounding), q-scale,
         * zero right-padding (padded g rows extend the cumsum flat). */
        for (size_t t = 0; t < s; t++) {
            apus_gkda_l2norm_row(q + t * hd + h * D, qp + t * D, D, sq);
            apus_gkda_l2norm_row(k + t * hd + h * D, kp + t * D, D, sq);
            float bt = apus_bf16_f32(beta[t * H + h]);
            for (size_t d = 0; d < D; d++) {
                qp[t * D + d] = qp[t * D + d] * qscale;
                float vv = apus_bf16_f32(v[t * hd + h * D + d]);
                vp[t * D + d] = vv;
                gc[t * D + d] = g[t * hd + h * D + d];
                kb[t * D + d] = kp[t * D + d] * bt;
                vb[t * D + d] = vv * bt;
            }
        }
        for (size_t t = s; t < sp; t++) {
            /* zero right-padding; the within-chunk cumsum below carries
             * the last real row's cumulative decay into the padded rows
             * (glm5:511 + the oracle's padding note) */
            memset(qp + t * D, 0, D * sizeof(float));
            memset(kp + t * D, 0, D * sizeof(float));
            memset(vp + t * D, 0, D * sizeof(float));
            memset(kb + t * D, 0, D * sizeof(float));
            memset(vb + t * D, 0, D * sizeof(float));
            memset(gc + t * D, 0, D * sizeof(float));
        }
        /* gc = cumsum WITHIN each chunk (sequential ascending). */
        for (size_t n = 0; n < N; n++)
            for (size_t i = 1; i < C; i++)
                for (size_t d = 0; d < D; d++)
                    gc[(n * C + i) * D + d] =
                        gc[(n * C + i - 1) * D + d] + gc[(n * C + i) * D + d];

        float *Sh = S + h * D * D;
        for (size_t n = 0; n < N; n++) {
            const float *qcn = qp + n * C * D;
            const float *kcn = kp + n * C * D;
            const float *gcn = gc + n * C * D;
            const float *kbcn = kb + n * C * D;
            const float *vbcn = vb + n * C * D;
            /* decay[i,j,d] = expf(gc[i,d] - gc[j,d]), j <= i only. */
            for (size_t i = 0; i < C; i++)
                for (size_t j = 0; j <= i; j++)
                    for (size_t d = 0; d < D; d++)
                        decay[(i * C + j) * D + d] =
                            expf(gcn[i * D + d] - gcn[j * D + d]);
            /* attn[i,j] = -sum_d (kb[i,d]*k[j,d]) * decay[i,j,d], j < i;
             * masked entries are +0.0f (the reference's masked_fill). */
            memset(attn, 0, C * C * sizeof(float));
            for (size_t i = 1; i < C; i++)
                for (size_t j = 0; j < i; j++) {
                    float acc = 0.0f;
                    for (size_t d = 0; d < D; d++)
                        acc += (kbcn[i * D + d] * kcn[j * D + d])
                             * decay[(i * C + j) * D + d];
                    attn[i * C + j] = -acc;
                }
            /* UT transform: attn[i,:i] += attn[i,:i] @ attn[:i,:i], rows
             * in ascending i; the (row[...,None]*sub).sum(axis=-2) is a
             * STRIDED sum -> plain sequential from +0.0f. The += I comes
             * AFTER the whole solve (glm5:540) — the diagonal is still
             * +0.0f while later rows read it. */
            for (size_t i = 1; i < C; i++) {
                float *row = attn + i * C;
                for (size_t j = 0; j < i; j++) {
                    float acc = 0.0f;
                    for (size_t m = 0; m < i; m++)
                        acc += row[m] * attn[m * C + j];
                    row[j] = row[j] + acc;
                }
            }
            for (size_t i = 0; i < C; i++)
                attn[i * C + i] = attn[i * C + i] + 1.0f;
            /* value = attn @ v_beta; kcd = attn @ (k_beta * exp(gc)) —
             * _mm contractions over the chunk axis, ascending, +0.0f. */
            for (size_t i = 0; i < C; i++)
                for (size_t dv = 0; dv < D; dv++) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < C; j++)
                        acc += attn[i * C + j] * vbcn[j * D + dv];
                    value[i * D + dv] = acc;
                }
            for (size_t i = 0; i < C; i++)
                for (size_t d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < C; j++)
                        acc += attn[i * C + j]
                             * (kbcn[j * D + d] * expf(gcn[j * D + d]));
                    kcd[i * D + d] = acc;
                }
            /* inter = (q * exp(gc)) @ S */
            float *inter = (float *)malloc(C * D * sizeof(float));
            for (size_t i = 0; i < C; i++)
                for (size_t dv = 0; dv < D; dv++) {
                    float acc = 0.0f;
                    for (size_t d = 0; d < D; d++)
                        acc += (qcn[i * D + d] * expf(gcn[i * D + d]))
                             * Sh[d * D + dv];
                    inter[i * D + dv] = acc;
                }
            /* intra[i,j] = sum_d (q[i,d]*k[j,d]) * decay[i,j,d], j <= i
             * (the diagonal is KEPT here — triu(k=1) mask). */
            memset(intra, 0, C * C * sizeof(float));
            for (size_t i = 0; i < C; i++)
                for (size_t j = 0; j <= i; j++) {
                    float acc = 0.0f;
                    for (size_t d = 0; d < D; d++)
                        acc += (qcn[i * D + d] * kcn[j * D + d])
                             * decay[(i * C + j) * D + d];
                    intra[i * C + j] = acc;
                }
            /* v_new = value - kcd @ S */
            for (size_t i = 0; i < C; i++)
                for (size_t dv = 0; dv < D; dv++) {
                    float acc = 0.0f;
                    for (size_t d = 0; d < D; d++)
                        acc += kcd[i * D + d] * Sh[d * D + dv];
                    vnew[i * D + dv] = value[i * D + dv] - acc;
                }
            /* out = inter + intra @ v_new (bf16 rounding, real rows) */
            for (size_t i = 0; i < C; i++) {
                size_t t = n * C + i;
                if (t >= s)
                    break;
                uint16_t *orow = out + t * hd + h * D;
                for (size_t dv = 0; dv < D; dv++) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < C; j++)
                        acc += intra[i * C + j] * vnew[j * D + dv];
                    orow[dv] = apus_bf16_bits(inter[i * D + dv] + acc);
                }
            }
            free(inter);
            /* S[m,dv] = S[m,dv]*expf(gc[last,m])
             *           + sum_j (k[j,m]*expf(gc[last,m]-gc[j,m]))
             *             * v_new[j,dv] */
            const float *gl = gcn + (C - 1) * D;
            for (size_t m = 0; m < D; m++) {
                float egl = expf(gl[m]);
                for (size_t dv = 0; dv < D; dv++) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < C; j++)
                        acc += (kcn[j * D + m]
                                * expf(gl[m] - gcn[j * D + m]))
                             * vnew[j * D + dv];
                    Sh[m * D + dv] = Sh[m * D + dv] * egl + acc;
                }
            }
        }
    }

    free(sq);
    free(intra);
    free(vnew);
    free(kcd);
    free(value);
    free(attn);
    free(decay);
    free(vb);
    free(kb);
    free(gc);
    free(vp);
    free(kp);
    free(qp);
}

/* --- (c) gated o_norm -----------------------------------------------------*/

void apus_gkda_onorm(const uint16_t *core, const uint16_t *gate,
                     const uint16_t *w, uint16_t *y,
                     size_t s, size_t H, size_t D, float eps) {
    float *sq = (float *)malloc(D * sizeof(float));
    float *w32 = (float *)malloc(D * sizeof(float));
    for (size_t d = 0; d < D; d++)
        w32[d] = apus_bf16_f32(w[d]);
    for (size_t r = 0; r < s * H; r++) {
        const uint16_t *cr = core + r * D;
        const uint16_t *gr = gate + r * D;
        uint16_t *yr = y + r * D;
        for (size_t d = 0; d < D; d++) {
            float v = apus_bf16_f32(cr[d]);
            sq[d] = v * v;
        }
        float var = apus_gmhc_pw_sum(sq, D) / (float)D;
        float inv = 1.0f / sqrtf(var + eps);
        for (size_t d = 0; d < D; d++) {
            float t = apus_bf16_f32(cr[d]) * inv;
            t = w32[d] * t;
            t = t * apus_gmhc_sigmoid(apus_bf16_f32(gr[d]));
            yr[d] = apus_bf16_bits(t);
        }
    }
    free(w32);
    free(sq);
}

/* --- composed forward -----------------------------------------------------*/

/* bf16 gemm into a [s, O] codes block (mt dispatch; bitwise at every
 * APUS_THREADS). Scratch xf: s*K floats. */
static void apus_gkda_linear(const uint16_t *w, const uint16_t *x,
                             float *xf, uint16_t *y,
                             size_t s, size_t O, size_t K) {
    apus_bf16_gemm_mt(w, x, xf, y, s, O, K);
}

void apus_gkda_forward(const ApusGkdaW *W, const uint16_t *x, size_t s,
                       ApusGkdaState *st, int decode, uint16_t *out,
                       ApusGkdaInterm *im) {
    size_t dim = W->dim, H = W->H, D = W->D, Dr = W->Dr;
    size_t qkv = H * D;
    uint16_t *mixed = (uint16_t *)malloc(s * 3 * qkv * sizeof(uint16_t));
    uint16_t *mq = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *mk = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *mv = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *fg1 = (uint16_t *)malloc(s * Dr * sizeof(uint16_t));
    uint16_t *fg2 = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *bt = (uint16_t *)malloc(s * H * sizeof(uint16_t));
    uint16_t *ga = (uint16_t *)malloc(s * Dr * sizeof(uint16_t));
    uint16_t *gate = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *onorm = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *core = (uint16_t *)malloc(s * qkv * sizeof(uint16_t));
    uint16_t *beta = (uint16_t *)malloc(s * H * sizeof(uint16_t));
    float *g = (float *)malloc(s * qkv * sizeof(float));
    size_t xmax = dim > Dr ? dim : Dr;
    if (qkv > xmax) xmax = qkv;
    float *xf = (float *)malloc(s * xmax * sizeof(float));

    /* q/k/v projections + concat (glm5:643-650; mixed rows are the
     * interleaved [q | k | v] per token) */
    apus_gkda_linear(W->q_w, x, xf, mq, s, qkv, dim);
    apus_gkda_linear(W->k_w, x, xf, mk, s, qkv, dim);
    apus_gkda_linear(W->v_w, x, xf, mv, s, qkv, dim);
    for (size_t t = 0; t < s; t++) {
        memcpy(mixed + t * 3 * qkv, mq + t * qkv, qkv * sizeof(uint16_t));
        memcpy(mixed + t * 3 * qkv + qkv, mk + t * qkv,
               qkv * sizeof(uint16_t));
        memcpy(mixed + t * 3 * qkv + 2 * qkv, mv + t * qkv,
               qkv * sizeof(uint16_t));
    }
    /* fused causal conv + silu (glm5:659-684): one call for both phases —
     * the decode batch steps the window token by token, and the prefill
     * loop IS the chained per-token body (bitwise == s one-by-one
     * conv_step calls, s == 1 included). */
    apus_gkda_conv_prefill(mixed, W->conv_w, mixed, 3 * qkv, s,
                           st->conv_state);
    /* split back (glm5:686-690) */
    for (size_t t = 0; t < s; t++) {
        memcpy(mq + t * qkv, mixed + t * 3 * qkv, qkv * sizeof(uint16_t));
        memcpy(mk + t * qkv, mixed + t * 3 * qkv + qkv,
               qkv * sizeof(uint16_t));
        memcpy(mv + t * qkv, mixed + t * 3 * qkv + 2 * qkv,
               qkv * sizeof(uint16_t));
    }
    /* forget gate (glm5:697): fg = bf16(f_b @ bf16(f_a @ x));
     * g = lower_bound * sigmoid(exp(A_log) * (fg + dt_bias)) — fp32,
     * never rounded */
    apus_gkda_linear(W->f_a, x, xf, fg1, s, Dr, dim);
    apus_gkda_linear(W->f_b, fg1, xf, fg2, s, qkv, Dr);
    for (size_t t = 0; t < s; t++)
        for (size_t hh = 0; hh < H; hh++) {
            float dec = expf(W->A_log[hh]);
            for (size_t d = 0; d < D; d++) {
                size_t i = hh * D + d;
                float z = apus_bf16_f32(fg2[t * qkv + i]) + W->dt_bias[i];
                g[t * qkv + i] =
                    W->lower_bound * apus_gmhc_sigmoid(dec * z);
            }
        }
    /* beta (glm5:698): bf16(sigmoid(bf16(b_proj @ x))) */
    apus_gkda_linear(W->b_w, x, xf, bt, s, H, dim);
    for (size_t i = 0; i < s * H; i++)
        beta[i] = apus_bf16_bits(apus_gmhc_sigmoid(apus_bf16_f32(bt[i])));
    /* core (glm5:701-724): chunked prefill OR recurrent decode */
    if (decode)
        apus_gkda_recurrent(mq, mk, mv, g, beta,
                            st->rec_state, core, s, H, D);
    else
        apus_gkda_chunk(mq, mk, mv, g, beta,
                        st->rec_state, core, s, H, D);
    /* output gate + o_norm + o_proj (glm5:730-732) */
    apus_gkda_linear(W->g_a, x, xf, ga, s, Dr, dim);
    apus_gkda_linear(W->g_b, ga, xf, gate, s, qkv, Dr);
    apus_gkda_onorm(core, gate, W->o_norm, onorm, s, H, D, W->eps);
    apus_gkda_linear(W->o_w, onorm, xf, out, s, dim, qkv);

    if (im) {
        if (im->mixed) memcpy(im->mixed, mixed, s * 3 * qkv * sizeof(uint16_t));
        if (im->g) memcpy(im->g, g, s * qkv * sizeof(float));
        if (im->beta) memcpy(im->beta, beta, s * H * sizeof(uint16_t));
        if (im->core) memcpy(im->core, core, s * qkv * sizeof(uint16_t));
        if (im->gate) memcpy(im->gate, gate, s * qkv * sizeof(uint16_t));
        if (im->onorm) memcpy(im->onorm, onorm, s * qkv * sizeof(uint16_t));
    }
    free(xf);
    free(g);
    free(beta);
    free(core);
    free(onorm);
    free(gate);
    free(ga);
    free(bt);
    free(fg2);
    free(fg1);
    free(mv);
    free(mk);
    free(mq);
    free(mixed);
}

#endif /* APUS_GKDA_IMPLEMENTATION */
#endif /* APUS_GKDA_H */
