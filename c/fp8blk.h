/*
 * c/fp8blk.h — GLM-5.3-Flash FP8-E4M3 + F32 weight_scale_inv 128x128-block
 * dequant -> BF16 (M3). C11, libc only (+ arm_neon.h on ARM, immintrin.h
 * via c/x86.h on x86-64).
 *
 * Numerics contract (normative: tools/oracle.py fp8_dequant + bf16_round,
 * pinned in tests/m0/README.md; storage semantics pinned by
 * tests/m1/test_3_schema.py):
 *
 *   Storage:  W  [O, K] FP8-E4M3 codes, row-major.
 *             WS [ceil(O/128) * ceil(K/128)] F32 scales, row-major over the
 *             block grid: the scale of element (o,k) is
 *             WS[(o/128) * nkb + (k/128)], nkb = ceil(K/128). PLAIN F32
 *             scales (NOT the V4/base UE8M0 power-of-2 format).
 *   Per element (the oracle's f32-mode dequant, ONE IEEE fp32 rounding):
 *             p   = e4m3(code) * ws       (single fp32 multiply)
 *             out = bf16_rne(p)           (c/num.h apus_bf16_bits)
 *   E4M3 decode is EXACT (c/num.h apus_e4m3_dequant_f32). NaN codes
 *   0x7F/0xFF decode as +-480 (e=15,m=7 as a normal) — they are refused by
 *   the converter (tests/m1) and never occur in normative data; the SIMD
 *   expansions decode them identically, so the paths stay bitwise even on
 *   out-of-contract input.
 *   O and K are NOT required to be multiples of 128: edge blocks are
 *   partial (the scale of a partial block still covers its elements) —
 *   the real checkpoint has ceil-shaped scale tensors (tests/m1).
 *
 *   NEON/AVX2 paths are BITWISE identical to the scalar path: the E4M3
 *   expand is exact (the FP16 bit-placement trick — E4M3 values are exact
 *   in FP16, decode = f32(h16) * 256; c/x86.h proves the x86 variants on
 *   all 256 codes), the scale multiply is one vmulq/_mm256_mul_ps rounding
 *   (same as the scalar mul), and the BF16 narrow is the scalar RNE
 *   bit-trick in integer lanes. No FMA, -ffp-contract=off discipline.
 *
 * The BF16 output feeds c/bf16.h's GEMV/GEMM — the composition
 * (dequant -> BF16 -> BF16 matmul fp32-acc -> BF16) is exactly the oracle's
 * fp8_linear (tests/m3g gates the composition bitwise).
 *
 * Usage: #define APUS_FP8BLK_IMPLEMENTATION in exactly one TU (also
 * instantiates c/num.h's helpers — guarded). Scalar always compiled; NEON
 * when __ARM_NEON; AVX2 runtime-dispatched on x86-64.
 */
#ifndef APUS_FP8BLK_H
#define APUS_FP8BLK_H

#include <stddef.h>
#include <stdint.h>

#include "num.h"    /* apus_e4m3_dequant_f32 / apus_bf16_bits (impl below) */
#include "x86.h"    /* AVX2 runtime dispatch + expand/narrow (no-op off x86) */
#include "pool.h"   /* apus_pool_run (dequant_mt) */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_FP8BLK_GROUP 128u   /* weight_scale_inv block edge (O and K) */

/* Number of 128-blocks covering n (ceil(n/128)). */
size_t apus_fp8blk_nblocks(size_t n);

/* Dequantize W [O,K] E4M3 + WS F32 128x128-block scales -> OUT [O,K] BF16
 * codes. All three paths are bitwise identical. */
void apus_fp8blk_dequant_scalar(const uint8_t *w, const float *ws,
                                uint16_t *out, size_t O, size_t K);
#ifdef __ARM_NEON
void apus_fp8blk_dequant_neon(const uint8_t *w, const float *ws,
                              uint16_t *out, size_t O, size_t K);
#endif
#if APUS_X86
void apus_fp8blk_dequant_avx2(const uint8_t *w, const float *ws,
                              uint16_t *out, size_t O, size_t K);
#endif

/* Dispatch: NEON (compile-time) / AVX2 (runtime, scalar fallback) / scalar.
 * Numerics-neutral — every path produces the same bits. */
void apus_fp8blk_dequant(const uint8_t *w, const float *ws,
                         uint16_t *out, size_t O, size_t K);

/* Threaded variant (c/pool.h): output rows partitioned contiguously over
 * the pool lanes. Dequant is elementwise — there is no accumulation order
 * — so the result is BITWISE the single-thread dispatch at every
 * APUS_THREADS. Call only from the compute thread (the c/pool.h pool is
 * one-job-at-a-time; the gcache I/O workers keep the plain dispatch). */
void apus_fp8blk_dequant_mt(const uint8_t *w, const float *ws,
                            uint16_t *out, size_t O, size_t K);

#ifdef __cplusplus
}
#endif

#endif /* APUS_FP8BLK_H */

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_FP8BLK_IMPLEMENTATION) && !defined(APUS_FP8BLK_IMPL_INCLUDED)
#define APUS_FP8BLK_IMPL_INCLUDED

#include <string.h>

/* Instantiate the shared scalar numerics helpers (guarded — a TU that also
 * defines APUS_NUM_IMPLEMENTATION or APUS_FP4_IMPLEMENTATION is fine). */
#define APUS_NUM_IMPLEMENTATION
#include "num.h"

size_t apus_fp8blk_nblocks(size_t n) {
    return (n + APUS_FP8BLK_GROUP - 1) / APUS_FP8BLK_GROUP;
}

/* Rows [o0, o1) — the per-row body shared by the single-thread entry and
 * the mt row partition (same code, so both produce the same bits). */
static void apus_fp8blk_rows_scalar(const uint8_t *w, const float *ws,
                                    uint16_t *out, size_t K, size_t nkb,
                                    size_t o0, size_t o1) {
    for (size_t o = o0; o < o1; o++) {
        const uint8_t *wr = w + o * K;
        const float *sr = ws + (o / APUS_FP8BLK_GROUP) * nkb;
        uint16_t *orow = out + o * K;
        for (size_t kb = 0; kb < nkb; kb++) {
            float s = sr[kb];
            size_t lo = kb * APUS_FP8BLK_GROUP;
            size_t hi = lo + APUS_FP8BLK_GROUP;
            if (hi > K) hi = K;
            for (size_t k = lo; k < hi; k++)
                orow[k] = apus_bf16_bits(apus_e4m3_dequant_f32(wr[k]) * s);
        }
    }
}

void apus_fp8blk_dequant_scalar(const uint8_t *w, const float *ws,
                                uint16_t *out, size_t O, size_t K) {
    apus_fp8blk_rows_scalar(w, ws, out, K, apus_fp8blk_nblocks(K), 0, O);
}

/* -------------------------------------------------------------------------*/
#ifdef __ARM_NEON

/* Expand 16 E4M3 codes -> 4 float32x4, EXACT (the M9a FP16 bit-placement
 * trick: h16 = (c&0x80)<<8 | (c&0x7F)<<7 read as FP16 is exactly 2^-8 x the
 * E4M3 value for normals AND subnormals; f32(h16) * 256 is exact for all
 * 256 codes, NaN codes decode as +-480 like apus_e4m3_dequant_f32). */
static inline void apus_fp8blk_expand16_neon(const uint8_t *p,
                                             float32x4_t v[4]) {
    uint8x16_t b = vld1q_u8(p);
    uint16x8_t cl = vmovl_u8(vget_low_u8(b)), ch = vmovl_u8(vget_high_u8(b));
    uint16x8_t hl = vorrq_u16(vshlq_n_u16(vandq_u16(cl, vdupq_n_u16(0x80u)), 8),
                              vshlq_n_u16(vandq_u16(cl, vdupq_n_u16(0x7Fu)), 7));
    uint16x8_t hh = vorrq_u16(vshlq_n_u16(vandq_u16(ch, vdupq_n_u16(0x80u)), 8),
                              vshlq_n_u16(vandq_u16(ch, vdupq_n_u16(0x7Fu)), 7));
    v[0] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hl))),
                       256.0f);
    v[1] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hl))),
                       256.0f);
    v[2] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(hh))),
                       256.0f);
    v[3] = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(hh))),
                       256.0f);
}

/* Narrow 4 FP32 -> 4 BF16 codes: the scalar RNE bit-trick in integer lanes
 * (u += 0x7FFF + ((u>>16)&1), truncate; NaN passes through as the high 16
 * bits). INTEGER ops only — bitwise == apus_bf16_bits per lane. */
static inline uint16x4_t apus_fp8blk_narrow4_neon(float32x4_t x) {
    uint32x4_t u = vreinterpretq_u32_f32(x);
    uint32x4_t mag = vandq_u32(u, vdupq_n_u32(0x7fffffffu));
    uint32x4_t isnan = vcgtq_u32(mag, vdupq_n_u32(0x7f800000u));
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7FFFu),
        vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u)));
    uint32x4_t code = vshrq_n_u32(vaddq_u32(u, bias), 16);
    uint32x4_t ncode = vshrq_n_u32(u, 16);
    return vmovn_u32(vbslq_u32(isnan, ncode, code));
}

static void apus_fp8blk_rows_neon(const uint8_t *w, const float *ws,
                                  uint16_t *out, size_t K, size_t nkb,
                                  size_t o0, size_t o1) {
    for (size_t o = o0; o < o1; o++) {
        const uint8_t *wr = w + o * K;
        const float *sr = ws + (o / APUS_FP8BLK_GROUP) * nkb;
        uint16_t *orow = out + o * K;
        for (size_t kb = 0; kb < nkb; kb++) {
            float32x4_t s = vdupq_n_f32(sr[kb]);
            size_t lo = kb * APUS_FP8BLK_GROUP;
            size_t hi = lo + APUS_FP8BLK_GROUP;
            if (hi > K) hi = K;
            size_t k = lo;
            for (; k + 16 <= hi; k += 16) {
                float32x4_t v[4];
                apus_fp8blk_expand16_neon(wr + k, v);
                for (int j = 0; j < 4; j++)
                    vst1_u16(orow + k + 4 * (size_t)j,
                        apus_fp8blk_narrow4_neon(vmulq_f32(v[j], s)));
            }
            for (; k < hi; k++)
                orow[k] = apus_bf16_bits(apus_e4m3_dequant_f32(wr[k])
                                         * sr[kb]);
        }
    }
}

void apus_fp8blk_dequant_neon(const uint8_t *w, const float *ws,
                              uint16_t *out, size_t O, size_t K) {
    apus_fp8blk_rows_neon(w, ws, out, K, apus_fp8blk_nblocks(K), 0, O);
}

#endif /* __ARM_NEON */

/* -------------------------------------------------------------------------*/
#if APUS_X86

/* The AVX2 worker is a target-attributed static (the c/fp8.h pattern): the
 * plain extern entry below delegates to it under the runtime gate, so no
 * target-mismatch inlining can occur. */
APUS_TGT_AVX2
static void apus_fp8blk_rows_avx2(const uint8_t *w, const float *ws,
                                  uint16_t *out, size_t K, size_t nkb,
                                  size_t o0, size_t o1) {
    for (size_t o = o0; o < o1; o++) {
        const uint8_t *wr = w + o * K;
        const float *sr = ws + (o / APUS_FP8BLK_GROUP) * nkb;
        uint16_t *orow = out + o * K;
        for (size_t kb = 0; kb < nkb; kb++) {
            __m256 s = _mm256_set1_ps(sr[kb]);
            size_t lo = kb * APUS_FP8BLK_GROUP;
            size_t hi = lo + APUS_FP8BLK_GROUP;
            if (hi > K) hi = K;
            size_t k = lo;
            for (; k + 16 <= hi; k += 16) {
                float v[16], p[16];
                /* exact expand (c/x86.h; F16C and integer variants proven
                 * bitwise on all 256 codes in tests/m12) */
                apus_e4m3_expand16_x86(wr + k, v);
                /* ONE fp32 rounding per element (same as the scalar mul) */
                _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(v), s));
                _mm256_storeu_ps(p + 8,
                                 _mm256_mul_ps(_mm256_loadu_ps(v + 8), s));
                apus_bf16_narrow8_x86(p, orow + k);
                apus_bf16_narrow8_x86(p + 8, orow + k + 8);
            }
            for (; k < hi; k++)
                orow[k] = apus_bf16_bits(apus_e4m3_dequant_f32(wr[k])
                                         * sr[kb]);
        }
    }
}

void apus_fp8blk_dequant_avx2(const uint8_t *w, const float *ws,
                              uint16_t *out, size_t O, size_t K) {
    apus_fp8blk_rows_avx2(w, ws, out, K, apus_fp8blk_nblocks(K), 0, O);
}

#endif /* APUS_X86 */

void apus_fp8blk_dequant(const uint8_t *w, const float *ws,
                         uint16_t *out, size_t O, size_t K) {
#ifdef __ARM_NEON
    apus_fp8blk_dequant_neon(w, ws, out, O, K);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_fp8blk_dequant_avx2(w, ws, out, O, K);
    } else {
        apus_fp8blk_dequant_scalar(w, ws, out, O, K);
    }
#else
    apus_fp8blk_dequant_scalar(w, ws, out, O, K);
#endif
}

/* --- threaded variant -------------------------------------------------------
 * Elementwise: lanes write disjoint output rows with the same per-row code,
 * so the result is bitwise the single-thread dispatch at every pool size. */

typedef struct {
    const uint8_t *w;
    const float   *ws;
    uint16_t      *out;
    size_t        K, nkb;
} ApusFp8blkJob;

#ifdef __ARM_NEON
static void apus_fp8blk_neon_rows_job(void *vjob, size_t o0, size_t o1) {
    const ApusFp8blkJob *j = vjob;
    apus_fp8blk_rows_neon(j->w, j->ws, j->out, j->K, j->nkb, o0, o1);
}
#else
static void apus_fp8blk_scalar_rows_job(void *vjob, size_t o0, size_t o1) {
    const ApusFp8blkJob *j = vjob;
    apus_fp8blk_rows_scalar(j->w, j->ws, j->out, j->K, j->nkb, o0, o1);
}
#if APUS_X86
static void apus_fp8blk_avx2_rows_job(void *vjob, size_t o0, size_t o1) {
    const ApusFp8blkJob *j = vjob;
    apus_fp8blk_rows_avx2(j->w, j->ws, j->out, j->K, j->nkb, o0, o1);
}
#endif
#endif

void apus_fp8blk_dequant_mt(const uint8_t *w, const float *ws,
                            uint16_t *out, size_t O, size_t K) {
    ApusFp8blkJob job = { w, ws, out, K, apus_fp8blk_nblocks(K) };
#ifdef __ARM_NEON
    apus_pool_run(O, apus_fp8blk_neon_rows_job, &job);
#elif APUS_X86
    if (apus_x86_have_avx2()) {
        atomic_fetch_add(&apus_x86_hits, 1);
        apus_pool_run(O, apus_fp8blk_avx2_rows_job, &job);
    } else {
        apus_pool_run(O, apus_fp8blk_scalar_rows_job, &job);
    }
#else
    apus_pool_run(O, apus_fp8blk_scalar_rows_job, &job);
#endif
}

#endif /* APUS_FP8BLK_IMPLEMENTATION */
