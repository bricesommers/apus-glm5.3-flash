/*
 * c/bf16.h — BF16 GEMV/GEMM kernels for GLM-5.3-Flash (M3). C11, libc only
 * (+ arm_neon.h on ARM, immintrin.h via c/x86.h on x86-64).
 *
 * Ported across the adapter seam from ../Apus-Ling-3.0-Flash-bf16 c/bf16.h
 * (attribution: that repo's M3 kernel discipline, itself mirroring the Apus
 * M3 c/fp4.h pattern). Trimmed for this milestone: the donor's M9b ILP
 * reorder kernels and hot wrappers are NOT ported (perf classes are
 * later-milestone work here); what lands is the scalar anchor + the
 * bitwise NEON / AVX2 / mt kernels. M7b added the GLM Metal hook table
 * (c/backend_gmetal.h): the mt entry points try the GPU hook first
 * (bitwise-identical sequential-k semantics) and fall back per-op to the
 * pinned CPU kernels below.
 *
 * Numerics contract (the normative in-engine anchor for every BF16 linear —
 * EXACTLY the M0 oracle's bf16 matmul semantics, tests/m0/README.md:
 * "BF16 matmul with FP32 accumulate", deterministic sequential order):
 *
 *   Storage:  W [O, K] BF16 row-major, x [M, K] BF16 row-major,
 *             y [M, O] BF16 row-major.
 *   Widen:    bf16 -> fp32 is EXACT: f32 bits = (uint32)code << 16
 *             (subnormals, inf, NaN payloads included).
 *   Narrow:   fp32 -> bf16 is round-to-nearest-even on the low 16 bits
 *             (u += 0x7FFF + ((u>>16)&1), then truncate; NaN passes through
 *             as the high 16 bits) — identical to the M0 oracle's
 *             bf16_round for every non-NaN input.
 *   GEMV/GEMM (scalar anchor — the semantic definition):
 *             acc = 0.0f
 *             for k in 0..K-1 (SEQUENTIAL, strictly ascending):
 *                 p    = f32(W[o,k]) * f32(x[m,k])   (one IEEE fp32 rounding)
 *                 acc += p                           (a second rounding)
 *             y[m,o] = bf16_narrow(acc)
 *             Two roundings per element, NO FMA anywhere; contraction is
 *             pinned off (-ffp-contract=off) so the compiler cannot fuse.
 *             This mirrors the oracle's _mm over bf16-valued fp32 operands
 *             (fp32 accumulate, bf16 out) with a fixed summation order —
 *             the oracle's own order IS this sequential-k loop, so the
 *             scalar anchor is BITWISE equal to the oracle f32-mode matmul.
 *   NEON kernels: BITWISE identical to the scalar anchor by construction:
 *             widening is exact, the per-element products are computed 4-wide
 *             (vmulq_f32 — the same single rounding as the scalar mul) and
 *             staged, and every output's adds run strictly in ascending k
 *             order. ILP comes only from interleaving INDEPENDENT output
 *             chains (8 rows at a time), never from reassociation. No
 *             reorder class is consumed on this platform.
 *   AVX2 kernels (M12a-2 pattern, c/x86.h): same staged-product, strictly-
 *             sequential-adds pattern at 8-wide staging — BITWISE identical
 *             to the scalar anchor (proven in tests/m3g on x86; off x86 the
 *             entry points are absent and the mt path falls back to scalar).
 *   Threaded (mt): bitwise identical to the corresponding single-thread
 *             kernel at EVERY pool size (APUS_THREADS=1 included): x is
 *             widened once by the calling thread (exact), then output rows
 *             are partitioned contiguously over the c/pool.h lanes and each
 *             y[m,o] is computed entirely by one lane with the identical
 *             per-output accumulation order.
 *   Invariants: inputs are finite BF16 in normative use. IEEE propagation
 *             is deterministic and identical across all paths: inf and NaN
 *             widen/narrow exactly as above, 0*inf -> NaN, fp32 accumulation
 *             overflow -> +-inf -> bf16 inf. K has no alignment requirement
 *             (SIMD tails are scalar, same roundings).
 *
 * Usage: #define APUS_BF16_IMPLEMENTATION in exactly one TU (the
 * implementation section also instantiates c/num.h's helpers — guarded, so
 * combining with APUS_NUM_IMPLEMENTATION or APUS_FP4_IMPLEMENTATION in the
 * same TU is safe). Scalar paths are always compiled; NEON paths are
 * compiled when __ARM_NEON is defined and are proven bitwise against the
 * scalar paths in tests/m3g.
 */
#ifndef APUS_BF16_H
#define APUS_BF16_H

#include <stddef.h>
#include <stdint.h>

#include "num.h"    /* apus_bf16_bits / apus_bf16_f32 (impl below) */
#include "pool.h"
#include "x86.h"    /* AVX2 runtime dispatch (no-op off x86-64) */
#include "backend_gmetal.h"    /* M7b GLM Metal hook table (all-NULL default) */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* k staging granularity of the SIMD kernels (products staged per 32-wide
 * chunk before the strictly-sequential adds). Part of the implementation,
 * not the numerics contract. */
#define APUS_BF16_CHUNK 32u

/* --- scalar anchor (normative) -------------------------------------------*/

/* Row widen: out[i] = apus_bf16_f32(b[i]), i < n. Exact. */
void apus_bf16_widen_scalar(const uint16_t *b, float *out, size_t n);

/* y[o] = bf16_narrow( sum_k f32(W[o,k]) * f32(x[k]) ), sequential k,
 * mul+add (two roundings per element). w: [O,K], x: [K], y: [O]. */
void apus_bf16_gemv_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t O, size_t K);

/* Same per output, over M activation rows. w: [O,K], x: [M,K], y: [M,O].
 * Every y[m,o] is bitwise the GEMV result for row m (M-independence). */
void apus_bf16_gemm_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t M, size_t O, size_t K);

/* --- NEON kernels (BITWISE == scalar anchor) ------------------------------*/
#ifdef __ARM_NEON
void apus_bf16_widen_neon(const uint16_t *b, float *out, size_t n);
/* xf: scratch, K floats (widened x; filled by the kernel). */
void apus_bf16_gemv_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t O, size_t K);
/* xf: scratch, M*K floats. */
void apus_bf16_gemm_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t M, size_t O, size_t K);
#endif

/* --- AVX2 kernels (BITWISE == scalar anchor; c/x86.h contract) ------------*/
#if APUS_X86
void apus_bf16_gemv_avx2(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t O, size_t K);
void apus_bf16_gemm_avx2(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t M, size_t O, size_t K);
#endif

/* --- threaded variants (c/pool.h; bitwise at every thread count) ---------*/

/* xf: scratch, K floats. Bitwise == apus_bf16_gemv_neon / _avx2 (or the
 * scalar anchor off SIMD) for any APUS_THREADS. */
void apus_bf16_gemv_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t O, size_t K);
/* xf: scratch, M*K floats. Bitwise == apus_bf16_gemm_neon / _avx2 (or
 * scalar) for any APUS_THREADS. */
void apus_bf16_gemm_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t M, size_t O, size_t K);

#ifdef __cplusplus
}
#endif

#endif /* APUS_BF16_H */

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_BF16_IMPLEMENTATION) && !defined(APUS_BF16_IMPL_INCLUDED)
#define APUS_BF16_IMPL_INCLUDED

#include <stdio.h>
#include <string.h>

/* Instantiate the shared scalar numerics helpers (guarded — a TU that also
 * defines APUS_NUM_IMPLEMENTATION or APUS_FP4_IMPLEMENTATION is fine). */
#define APUS_NUM_IMPLEMENTATION
#include "num.h"

/* --- M7b GLM Metal backend hook table (c/backend_gmetal.h) ----------------
 * Defined here because the bf16.h implementation TU is linked into every
 * GLM engine/test binary. All-NULL = the pinned CPU kernels (default);
 * apus_gmetal_enable() fills it. The weak stubs let the plain CPU binary
 * link: c/backend_gmetal.mm (metal=1 build) provides strong definitions. */
ApusGmetalHooks apus_gmetal_hooks;

#if defined(__GNUC__)
__attribute__((weak)) int apus_gmetal_enable(char *err, size_t errcap) {
    if (err && errcap)
        snprintf(err, errcap,
                 "GLM metal backend not compiled in (make metal=1)");
    return -1;
}
__attribute__((weak)) void apus_gmetal_disable(void) {}
__attribute__((weak)) int apus_gmetal_is_enabled(void) { return 0; }
__attribute__((weak)) uint64_t apus_gmetal_bytes_wrapped(void) { return 0; }
__attribute__((weak)) uint64_t apus_gmetal_bytes_pinned(void) { return 0; }
__attribute__((weak)) uint64_t apus_gmetal_dispatches(void) { return 0; }
__attribute__((weak)) int apus_gmetal_register_region(const void *ptr,
                                                      size_t len) {
    (void)ptr; (void)len; return 1;
}
__attribute__((weak)) void apus_gmetal_unregister_region(const void *ptr) {
    (void)ptr;
}
__attribute__((weak)) uint64_t apus_gmetal_offload_bf16(void) { return 0; }
__attribute__((weak)) uint64_t apus_gmetal_offload_fp8blk(void) { return 0; }
__attribute__((weak)) int apus_gmetal_bf16_gemm(const uint16_t *w,
        const uint16_t *x, uint16_t *y, size_t M, size_t O, size_t K) {
    (void)w; (void)x; (void)y; (void)M; (void)O; (void)K; return 1;
}
__attribute__((weak)) int apus_gmetal_fp8blk_linear(const uint8_t *codes,
        const float *scales, const uint16_t *x, uint16_t *y,
        size_t M, size_t O, size_t K) {
    (void)codes; (void)scales; (void)x; (void)y; (void)M; (void)O; (void)K;
    return 1;
}
#endif /* __GNUC__ */

void apus_bf16_widen_scalar(const uint16_t *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = apus_bf16_f32(b[i]);
}

void apus_bf16_gemv_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t O, size_t K) {
    for (size_t o = 0; o < O; o++) {
        const uint16_t *wr = w + o * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++)
            acc += apus_bf16_f32(wr[k]) * apus_bf16_f32(x[k]);
        y[o] = apus_bf16_bits(acc);
    }
}

void apus_bf16_gemm_scalar(const uint16_t *w, const uint16_t *x,
                           uint16_t *y, size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_gemv_scalar(w, x + m * K, y + m * O, O, K);
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON

void apus_bf16_widen_neon(const uint16_t *b, float *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(b + i);
        vst1q_f32(out + i, vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(h), 16)));
        vst1q_f32(out + i + 4, vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(h), 16)));
    }
    for (; i < n; i++)
        out[i] = apus_bf16_f32(b[i]);
}

/* Stage the 32 products f32(wr[k..k+31]) * xf[k..k+31] into st, each with
 * the same single IEEE fp32 rounding as the scalar mul (plain vmulq, never
 * FMA). Widening is exact (vshll 16). */
static inline void apus_bf16_prod_chunk_neon(const uint16_t *wr,
                                             const float *xf, float *st) {
    for (int i = 0; i < (int)APUS_BF16_CHUNK; i += 8) {
        uint16x8_t h = vld1q_u16(wr + i);
        float32x4_t w0 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_low_u16(h), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(
            vshll_n_u16(vget_high_u16(h), 16));
        vst1q_f32(st + i,     vmulq_f32(w0, vld1q_f32(xf + i)));
        vst1q_f32(st + i + 4, vmulq_f32(w1, vld1q_f32(xf + i + 4)));
    }
}

/* One output dot, single sequential chain (row tails). BITWISE the scalar
 * anchor: staged single-rounded products, adds strictly in ascending k. */
static float apus_bf16_dot_neon(const uint16_t *wr, const float *xf,
                                size_t K, float *st) {
    float acc = 0.0f;
    size_t k = 0;
    for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
        apus_bf16_prod_chunk_neon(wr + k, xf + k, st);
        for (int i = 0; i < (int)APUS_BF16_CHUNK; i++)
            acc += st[i];
    }
    for (; k < K; k++)
        acc += apus_bf16_f32(wr[k]) * xf[k];
    return acc;
}

/* GEMV rows [o0, o1): 8 independent row chains for ILP (each chain IS the
 * scalar sequential sum for its row — no reassociation), 4-chain and
 * single-chain tails for the remainder. */
static void apus_bf16_gemv_rows_neon(const uint16_t *w, const float *xf,
                                     uint16_t *y, size_t K,
                                     size_t o0, size_t o1) {
    float st[8][APUS_BF16_CHUNK];
    size_t o = o0;
    for (; o + 8 <= o1; o += 8) {
        const uint16_t *wr[8];
        for (int r = 0; r < 8; r++) wr[r] = w + (o + r) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        size_t k = 0;
        for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
            apus_bf16_prod_chunk_neon(wr[0] + k, xf + k, st[0]);
            apus_bf16_prod_chunk_neon(wr[1] + k, xf + k, st[1]);
            apus_bf16_prod_chunk_neon(wr[2] + k, xf + k, st[2]);
            apus_bf16_prod_chunk_neon(wr[3] + k, xf + k, st[3]);
            apus_bf16_prod_chunk_neon(wr[4] + k, xf + k, st[4]);
            apus_bf16_prod_chunk_neon(wr[5] + k, xf + k, st[5]);
            apus_bf16_prod_chunk_neon(wr[6] + k, xf + k, st[6]);
            apus_bf16_prod_chunk_neon(wr[7] + k, xf + k, st[7]);
            for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
                a4 += st[4][i];
                a5 += st[5][i];
                a6 += st[6][i];
                a7 += st[7][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_bf16_f32(wr[0][k]) * xf[k];
            a1 += apus_bf16_f32(wr[1][k]) * xf[k];
            a2 += apus_bf16_f32(wr[2][k]) * xf[k];
            a3 += apus_bf16_f32(wr[3][k]) * xf[k];
            a4 += apus_bf16_f32(wr[4][k]) * xf[k];
            a5 += apus_bf16_f32(wr[5][k]) * xf[k];
            a6 += apus_bf16_f32(wr[6][k]) * xf[k];
            a7 += apus_bf16_f32(wr[7][k]) * xf[k];
        }
        y[o + 0] = apus_bf16_bits(a0);
        y[o + 1] = apus_bf16_bits(a1);
        y[o + 2] = apus_bf16_bits(a2);
        y[o + 3] = apus_bf16_bits(a3);
        y[o + 4] = apus_bf16_bits(a4);
        y[o + 5] = apus_bf16_bits(a5);
        y[o + 6] = apus_bf16_bits(a6);
        y[o + 7] = apus_bf16_bits(a7);
    }
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
            apus_bf16_prod_chunk_neon(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_neon(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_neon(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_neon(w3 + k, xf + k, st[3]);
            for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_bf16_f32(w0[k]) * xf[k];
            a1 += apus_bf16_f32(w1[k]) * xf[k];
            a2 += apus_bf16_f32(w2[k]) * xf[k];
            a3 += apus_bf16_f32(w3[k]) * xf[k];
        }
        y[o + 0] = apus_bf16_bits(a0);
        y[o + 1] = apus_bf16_bits(a1);
        y[o + 2] = apus_bf16_bits(a2);
        y[o + 3] = apus_bf16_bits(a3);
    }
    for (; o < o1; o++)
        y[o] = apus_bf16_bits(
            apus_bf16_dot_neon(w + o * K, xf, K, st[0]));
}

/* GEMM rows [o0, o1) for all m: m-groups of 4 (the widened W chunk is
 * shared across the 4 activation rows), single-chain tail group. Every
 * output keeps the scalar anchor's strictly-sequential per-output order, so
 * values are M- and thread-partition-independent. */
static void apus_bf16_gemm_rows_neon(const uint16_t *w, const float *xf,
                                     uint16_t *y, size_t M, size_t O,
                                     size_t K, size_t o0, size_t o1) {
    float st[4][APUS_BF16_CHUNK];
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        if (mc == 4) {
            const float *x0 = xf + (m0 + 0) * K;
            const float *x1 = xf + (m0 + 1) * K;
            const float *x2 = xf + (m0 + 2) * K;
            const float *x3 = xf + (m0 + 3) * K;
            for (size_t o = o0; o < o1; o++) {
                const uint16_t *wr = w + o * K;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                size_t k = 0;
                for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
                    for (int i = 0; i < (int)APUS_BF16_CHUNK; i += 8) {
                        uint16x8_t h = vld1q_u16(wr + k + i);
                        float32x4_t wv0 = vreinterpretq_f32_u32(
                            vshll_n_u16(vget_low_u16(h), 16));
                        float32x4_t wv1 = vreinterpretq_f32_u32(
                            vshll_n_u16(vget_high_u16(h), 16));
                        vst1q_f32(st[0] + i, vmulq_f32(wv0,
                            vld1q_f32(x0 + k + i)));
                        vst1q_f32(st[0] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x0 + k + i + 4)));
                        vst1q_f32(st[1] + i, vmulq_f32(wv0,
                            vld1q_f32(x1 + k + i)));
                        vst1q_f32(st[1] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x1 + k + i + 4)));
                        vst1q_f32(st[2] + i, vmulq_f32(wv0,
                            vld1q_f32(x2 + k + i)));
                        vst1q_f32(st[2] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x2 + k + i + 4)));
                        vst1q_f32(st[3] + i, vmulq_f32(wv0,
                            vld1q_f32(x3 + k + i)));
                        vst1q_f32(st[3] + i + 4, vmulq_f32(wv1,
                            vld1q_f32(x3 + k + i + 4)));
                    }
                    for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                        a0 += st[0][i];
                        a1 += st[1][i];
                        a2 += st[2][i];
                        a3 += st[3][i];
                    }
                }
                for (; k < K; k++) {
                    float wv = apus_bf16_f32(wr[k]);
                    a0 += wv * x0[k];
                    a1 += wv * x1[k];
                    a2 += wv * x2[k];
                    a3 += wv * x3[k];
                }
                y[(m0 + 0) * O + o] = apus_bf16_bits(a0);
                y[(m0 + 1) * O + o] = apus_bf16_bits(a1);
                y[(m0 + 2) * O + o] = apus_bf16_bits(a2);
                y[(m0 + 3) * O + o] = apus_bf16_bits(a3);
            }
        } else {
            for (size_t r = 0; r < mc; r++) {
                const float *xr = xf + (m0 + r) * K;
                for (size_t o = o0; o < o1; o++)
                    y[(m0 + r) * O + o] = apus_bf16_bits(
                        apus_bf16_dot_neon(w + o * K, xr, K, st[0]));
            }
        }
    }
}

void apus_bf16_gemv_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t O, size_t K) {
    apus_bf16_widen_neon(x, xf, K);
    apus_bf16_gemv_rows_neon(w, xf, y, K, 0, O);
}

void apus_bf16_gemm_neon(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_neon(x + m * K, xf + m * K, K);
    apus_bf16_gemm_rows_neon(w, xf, y, M, O, K, 0, O);
}

#endif /* __ARM_NEON */

/* -------------------------------------------------------------------------*/
#if APUS_X86

/* Row widen: out[i] = f32(b[i]), i < n. Exact, scalar tail. */
APUS_TGT_AVX2
static void apus_bf16_widen_avx2(const uint16_t *b, float *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, apus_bf16_widen8_x86(b + i));
    for (; i < n; i++)
        out[i] = apus_bf16_f32(b[i]);
}

/* Stage the 32 products f32(wr[k..k+31]) * xf[k..k+31] into st, each with
 * the same single IEEE fp32 rounding as the scalar mul (plain mul, never
 * FMA). Widening is exact. */
APUS_TGT_AVX2
static inline void apus_bf16_prod_chunk_avx2(const uint16_t *wr,
                                             const float *xf, float *st) {
    for (int i = 0; i < (int)APUS_BF16_CHUNK; i += 8)
        _mm256_storeu_ps(st + i, _mm256_mul_ps(apus_bf16_widen8_x86(wr + i),
                                               _mm256_loadu_ps(xf + i)));
}

/* One output dot, single sequential chain (row tails). BITWISE the scalar
 * anchor: staged single-rounded products, adds strictly in ascending k. */
APUS_TGT_AVX2
static float apus_bf16_dot_avx2(const uint16_t *wr, const float *xf,
                                size_t K, float *st) {
    float acc = 0.0f;
    size_t k = 0;
    for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
        apus_bf16_prod_chunk_avx2(wr + k, xf + k, st);
        for (int i = 0; i < (int)APUS_BF16_CHUNK; i++)
            acc += st[i];
    }
    for (; k < K; k++)
        acc += apus_bf16_f32(wr[k]) * xf[k];
    return acc;
}

/* GEMV rows [o0, o1): 8 independent row chains (each chain IS the scalar
 * sequential sum for its row — no reassociation; the same chain structure
 * as the NEON kernel), 4-chain and single-chain tails. NAMED accumulators
 * (the Rosetta spilling trap, c/x86.h header note). */
APUS_TGT_AVX2
static void apus_bf16_gemv_rows_avx2(const uint16_t *w, const float *xf,
                                     uint16_t *y, size_t K,
                                     size_t o0, size_t o1) {
    float st[8][APUS_BF16_CHUNK];
    size_t o = o0;
    for (; o + 8 <= o1; o += 8) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        const uint16_t *w4 = w + (o + 4) * K;
        const uint16_t *w5 = w + (o + 5) * K;
        const uint16_t *w6 = w + (o + 6) * K;
        const uint16_t *w7 = w + (o + 7) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        size_t k = 0;
        for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
            apus_bf16_prod_chunk_avx2(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_avx2(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_avx2(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_avx2(w3 + k, xf + k, st[3]);
            apus_bf16_prod_chunk_avx2(w4 + k, xf + k, st[4]);
            apus_bf16_prod_chunk_avx2(w5 + k, xf + k, st[5]);
            apus_bf16_prod_chunk_avx2(w6 + k, xf + k, st[6]);
            apus_bf16_prod_chunk_avx2(w7 + k, xf + k, st[7]);
            for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
                a4 += st[4][i];
                a5 += st[5][i];
                a6 += st[6][i];
                a7 += st[7][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_bf16_f32(w0[k]) * xf[k];
            a1 += apus_bf16_f32(w1[k]) * xf[k];
            a2 += apus_bf16_f32(w2[k]) * xf[k];
            a3 += apus_bf16_f32(w3[k]) * xf[k];
            a4 += apus_bf16_f32(w4[k]) * xf[k];
            a5 += apus_bf16_f32(w5[k]) * xf[k];
            a6 += apus_bf16_f32(w6[k]) * xf[k];
            a7 += apus_bf16_f32(w7[k]) * xf[k];
        }
        float accs[8] = { a0, a1, a2, a3, a4, a5, a6, a7 };
        apus_bf16_narrow8_x86(accs, y + o);
    }
    for (; o + 4 <= o1; o += 4) {
        const uint16_t *w0 = w + (o + 0) * K;
        const uint16_t *w1 = w + (o + 1) * K;
        const uint16_t *w2 = w + (o + 2) * K;
        const uint16_t *w3 = w + (o + 3) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        size_t k = 0;
        for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
            apus_bf16_prod_chunk_avx2(w0 + k, xf + k, st[0]);
            apus_bf16_prod_chunk_avx2(w1 + k, xf + k, st[1]);
            apus_bf16_prod_chunk_avx2(w2 + k, xf + k, st[2]);
            apus_bf16_prod_chunk_avx2(w3 + k, xf + k, st[3]);
            for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                a0 += st[0][i];
                a1 += st[1][i];
                a2 += st[2][i];
                a3 += st[3][i];
            }
        }
        for (; k < K; k++) {
            a0 += apus_bf16_f32(w0[k]) * xf[k];
            a1 += apus_bf16_f32(w1[k]) * xf[k];
            a2 += apus_bf16_f32(w2[k]) * xf[k];
            a3 += apus_bf16_f32(w3[k]) * xf[k];
        }
        y[o + 0] = apus_bf16_bits(a0);
        y[o + 1] = apus_bf16_bits(a1);
        y[o + 2] = apus_bf16_bits(a2);
        y[o + 3] = apus_bf16_bits(a3);
    }
    for (; o < o1; o++)
        y[o] = apus_bf16_bits(
            apus_bf16_dot_avx2(w + o * K, xf, K, st[0]));
}

/* GEMM rows [o0, o1) for all m: m-groups of 4 (the widened W chunk is
 * shared across the 4 activation rows), single-chain tail group. Every
 * output keeps the scalar anchor's strictly-sequential per-output order, so
 * values are M- and thread-partition-independent. */
APUS_TGT_AVX2
static void apus_bf16_gemm_rows_avx2(const uint16_t *w, const float *xf,
                                     uint16_t *y, size_t M, size_t O,
                                     size_t K, size_t o0, size_t o1) {
    float st[4][APUS_BF16_CHUNK];
    for (size_t m0 = 0; m0 < M; m0 += 4) {
        size_t mc = M - m0 < 4 ? M - m0 : 4;
        if (mc == 4) {
            const float *x0 = xf + (m0 + 0) * K;
            const float *x1 = xf + (m0 + 1) * K;
            const float *x2 = xf + (m0 + 2) * K;
            const float *x3 = xf + (m0 + 3) * K;
            for (size_t o = o0; o < o1; o++) {
                const uint16_t *wr = w + o * K;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                size_t k = 0;
                for (; k + APUS_BF16_CHUNK <= K; k += APUS_BF16_CHUNK) {
                    for (int i = 0; i < (int)APUS_BF16_CHUNK; i += 8) {
                        __m256 wv = apus_bf16_widen8_x86(wr + k + i);
                        _mm256_storeu_ps(st[0] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x0 + k + i)));
                        _mm256_storeu_ps(st[1] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x1 + k + i)));
                        _mm256_storeu_ps(st[2] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x2 + k + i)));
                        _mm256_storeu_ps(st[3] + i, _mm256_mul_ps(wv,
                            _mm256_loadu_ps(x3 + k + i)));
                    }
                    for (int i = 0; i < (int)APUS_BF16_CHUNK; i++) {
                        a0 += st[0][i];
                        a1 += st[1][i];
                        a2 += st[2][i];
                        a3 += st[3][i];
                    }
                }
                for (; k < K; k++) {
                    float wv = apus_bf16_f32(wr[k]);
                    a0 += wv * x0[k];
                    a1 += wv * x1[k];
                    a2 += wv * x2[k];
                    a3 += wv * x3[k];
                }
                y[(m0 + 0) * O + o] = apus_bf16_bits(a0);
                y[(m0 + 1) * O + o] = apus_bf16_bits(a1);
                y[(m0 + 2) * O + o] = apus_bf16_bits(a2);
                y[(m0 + 3) * O + o] = apus_bf16_bits(a3);
            }
        } else {
            for (size_t r = 0; r < mc; r++) {
                const float *xr = xf + (m0 + r) * K;
                for (size_t o = o0; o < o1; o++)
                    y[(m0 + r) * O + o] = apus_bf16_bits(
                        apus_bf16_dot_avx2(w + o * K, xr, K, st[0]));
            }
        }
    }
}

void apus_bf16_gemv_avx2(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t O, size_t K) {
    apus_bf16_widen_avx2(x, xf, K);
    apus_bf16_gemv_rows_avx2(w, xf, y, K, 0, O);
}

void apus_bf16_gemm_avx2(const uint16_t *w, const uint16_t *x, float *xf,
                         uint16_t *y, size_t M, size_t O, size_t K) {
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_avx2(x + m * K, xf + m * K, K);
    apus_bf16_gemm_rows_avx2(w, xf, y, M, O, K, 0, O);
}

#endif /* APUS_X86 */

/* --- threaded variants -----------------------------------------------------*/

typedef struct {
    const uint16_t *w;
    float *xf;          /* widened x: K floats (GEMV) / M*K floats (GEMM) */
    uint16_t *y;
    size_t M, O, K;
} ApusBf16Job;

#ifdef __ARM_NEON
/* Row bodies delegate to the shared NEON row functions, so mt output is
 * bitwise the single-thread NEON kernel for any row partition. */
static void apus_bf16_gemv_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemv_rows_neon(j->w, j->xf, j->y, j->K, o0, o1);
}
static void apus_bf16_gemm_neon_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemm_rows_neon(j->w, j->xf, j->y, j->M, j->O, j->K, o0, o1);
}
#else
/* Off-NEON fallback: the scalar anchor itself over the row range (xf is
 * exact widening, so this is bitwise apus_bf16_gem*_scalar). Also the
 * x86 runtime fallback when the CPU lacks AVX2 (APUS_X86). */
static void apus_bf16_gemv_scalar_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    for (size_t o = o0; o < o1; o++) {
        const uint16_t *wr = j->w + o * j->K;
        float acc = 0.0f;
        for (size_t k = 0; k < j->K; k++)
            acc += apus_bf16_f32(wr[k]) * j->xf[k];
        j->y[o] = apus_bf16_bits(acc);
    }
}
static void apus_bf16_gemm_scalar_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    for (size_t m = 0; m < j->M; m++) {
        const float *xr = j->xf + m * j->K;
        for (size_t o = o0; o < o1; o++) {
            const uint16_t *wr = j->w + o * j->K;
            float acc = 0.0f;
            for (size_t k = 0; k < j->K; k++)
                acc += apus_bf16_f32(wr[k]) * xr[k];
            j->y[m * j->O + o] = apus_bf16_bits(acc);
        }
    }
}
#if APUS_X86
/* AVX2 row bodies — BITWISE == the scalar rows, so the mt output is the
 * scalar anchor's bits at every pool size. */
static void apus_bf16_gemv_avx2_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemv_rows_avx2(j->w, j->xf, j->y, j->K, o0, o1);
}
static void apus_bf16_gemm_avx2_rows(void *vjob, size_t o0, size_t o1) {
    const ApusBf16Job *j = vjob;
    apus_bf16_gemm_rows_avx2(j->w, j->xf, j->y, j->M, j->O, j->K, o0, o1);
}
#endif
#endif

void apus_bf16_gemv_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t O, size_t K) {
    /* M7b GLM Metal hook (when enabled): BITWISE-identical sequential-k
     * semantics (c/backend_gmetal.h); on any hook failure the pinned CPU
     * kernel below runs (per-op fail-soft). */
    if (apus_gmetal_hooks.bf16_gemm
        && apus_gmetal_hooks.bf16_gemm(w, x, y, 1, O, K) == 0)
        return;
    ApusBf16Job job = { w, xf, y, 1, O, K };
#ifdef __ARM_NEON
    apus_bf16_widen_neon(x, xf, K);
    apus_pool_run(O, apus_bf16_gemv_neon_rows, &job);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_bf16_widen_avx2(x, xf, K);
        apus_pool_run(O, apus_bf16_gemv_avx2_rows, &job);
    } else {
        apus_bf16_widen_scalar(x, xf, K);
        apus_pool_run(O, apus_bf16_gemv_scalar_rows, &job);
    }
#else
    apus_bf16_widen_scalar(x, xf, K);
    apus_pool_run(O, apus_bf16_gemv_scalar_rows, &job);
#endif
}

void apus_bf16_gemm_mt(const uint16_t *w, const uint16_t *x, float *xf,
                       uint16_t *y, size_t M, size_t O, size_t K) {
    /* M7b GLM Metal hook (when enabled): BITWISE-identical sequential-k
     * semantics (c/backend_gmetal.h); per-op fail-soft to the CPU kernel. */
    if (apus_gmetal_hooks.bf16_gemm
        && apus_gmetal_hooks.bf16_gemm(w, x, y, M, O, K) == 0)
        return;
    ApusBf16Job job = { w, xf, y, M, O, K };
#ifdef __ARM_NEON
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_neon(x + m * K, xf + m * K, K);
    apus_pool_run(O, apus_bf16_gemm_neon_rows, &job);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        for (size_t m = 0; m < M; m++)
            apus_bf16_widen_avx2(x + m * K, xf + m * K, K);
        apus_pool_run(O, apus_bf16_gemm_avx2_rows, &job);
    } else {
        for (size_t m = 0; m < M; m++)
            apus_bf16_widen_scalar(x + m * K, xf + m * K, K);
        apus_pool_run(O, apus_bf16_gemm_scalar_rows, &job);
    }
#else
    for (size_t m = 0; m < M; m++)
        apus_bf16_widen_scalar(x + m * K, xf + m * K, K);
    apus_pool_run(O, apus_bf16_gemm_scalar_rows, &job);
#endif
}

#endif /* APUS_BF16_IMPLEMENTATION */
