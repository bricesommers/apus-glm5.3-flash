/*
 * c/num.h — shared scalar numerics helpers (M3 re-home). C11, libc only.
 *
 * These helpers were born in c/fp4.h (the DeepSeek-V4 MXFP4 kernel header)
 * and are used by the retained inherited engine (c/fp4.h, c/fp8.h, c/attn.h,
 * c/blas.h via the fp4.h instantiation) AND by the GLM-5.3-Flash kernel
 * layer (c/fp8blk.h, c/bf16.h). They live here so c/fp4.h can shrink to
 * MXFP4-only without breaking any consumer.
 *
 * Contents:
 *   - BF16 rounding/packing (RNE via the 0x7FFF + LSB bias bit trick —
 *     the SAME semantics as the M0 oracle's bf16_round, tests/m0/README.md);
 *   - FP8-E4M3 scalar codec (RNE quantize clamped to +-448; exact decode —
 *     NaN codes 0x7F/0xFF decode as +-480, matching the numpy table's
 *     non-NaN entries and documented in c/fp8blk.h);
 *   - UE8M0 byte -> FP32 scale (V4-only scale format; NOT used by the GLM
 *     path, whose weight_scale_inv is plain F32 — kept here because fp4.h,
 *     fp8.h and blas.h all share it);
 *   - the V4 act-quant helpers (per-128-along-K FP8-E4M3 activation
 *     quantization, ue8m0 scale variant). The apus_fp4_act_* names are kept
 *     verbatim so no call site changes; they are inherited-engine machinery,
 *     not part of the GLM numerics contract.
 *
 * Usage: #define APUS_NUM_IMPLEMENTATION in exactly one TU. c/fp4.h's
 * implementation section does this itself (the retained engine's historical
 * instantiation point), so every pre-M3 TU is unaffected; new GLM TUs
 * define APUS_NUM_IMPLEMENTATION directly.
 */
#ifndef APUS_NUM_H
#define APUS_NUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APUS_FP4_ACT_GROUP  128u  /* activation scale group along K (V4) */

/* UE8M0 byte -> FP32 scale 2^(b-127). Exact for all b: b==0 gives the FP32
 * subnormal 2^-127; b==255 gives +inf (2^128 overflows FP32), matching the
 * M1-pinned numpy semantics. */
float apus_ue8m0_f32(uint8_t b);

/* Quantize one FP32 value to FP8-E4M3 (round-to-nearest-even, clamped to the
 * finite range +-448). Input must be finite. */
uint8_t apus_e4m3_quant_f32(float x);

/* FP8-E4M3 code -> FP32 (exact). NaN codes 0x7F/0xFF decode as +-480
 * (e=15, m=7 evaluated as a normal); the GLM converter refuses NaN codes at
 * conversion time, so they never occur in normative data. */
float apus_e4m3_dequant_f32(uint8_t c);

/* Round FP32 to BF16 (round-to-nearest-even) and back to FP32. Provided so
 * callers can mirror the reference's BF16 act_quant input. */
float apus_bf16_round(float x);

/* BF16 bit pack/unpack. apus_bf16_bits(x) == bits(apus_bf16_round(x)) >> 16;
 * apus_bf16_f32 widening is exact. NaN passes through as the high 16 bits. */
uint16_t apus_bf16_bits(float x);
float apus_bf16_f32(uint16_t b);

/* Number of per-128 activation scale blocks for K (ceil(K/128)). */
size_t apus_fp4_act_blocks(size_t K);

/* Per-128-along-K FP8-E4M3 activation quantization, ue8m0 (power-of-2) scale
 * variant — the V4 normative act_quant(scale_fmt="ue8m0") port.
 * x: K floats; codes: K bytes out; scales: apus_fp4_act_blocks(K) floats out.
 * Requires K % 32 == 0. */
void apus_fp4_act_quant_scalar(const float *x, size_t K,
                               uint8_t *codes, float *scales);

#ifdef __cplusplus
}
#endif

#endif /* APUS_NUM_H */

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): c/fp4.h's
 * implementation section re-includes this header after defining
 * APUS_NUM_IMPLEMENTATION. */
#if defined(APUS_NUM_IMPLEMENTATION) && !defined(APUS_NUM_IMPL_INCLUDED)
#define APUS_NUM_IMPL_INCLUDED

#include <math.h>
#include <string.h>

float apus_ue8m0_f32(uint8_t b) {
    /* Bit construction avoids out-of-range double->float conversion (which
     * would be UB at b==255). b==0 -> 2^-127 (subnormal); b in 1..254 ->
     * 2^(b-127); b==255 -> +inf, matching np.exp2(128).astype(f32). */
    uint32_t bits = (b == 0) ? 0x00400000u : ((uint32_t)b << 23);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint32_t apus_rne_shift(uint32_t m, int sh) {
    /* m >> sh with round-to-nearest-even. 1 <= sh <= 24. */
    uint32_t half = 1u << (sh - 1);
    uint32_t mask = (1u << sh) - 1u;
    uint32_t r = m & mask;
    uint32_t v = m >> sh;
    if (r > half || (r == half && (v & 1u))) v++;
    return v;
}

uint8_t apus_e4m3_quant_f32(float x) {
    if (x > 448.0f) x = 448.0f;
    else if (x < -448.0f) x = -448.0f;
    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t sign = (u >> 31) << 7;
    uint32_t a = u & 0x7fffffffu;
    if (a == 0) return (uint8_t)sign;
    int e = (int)(a >> 23) - 127;
    uint32_t m23 = (a & 0x007fffffu) | 0x00800000u;
    uint32_t code;
    if (e >= -6) {
        /* normal: keep 3 mantissa bits, RNE; carry bumps the exponent */
        uint32_t v = apus_rne_shift(m23, 20); /* 8..16 */
        if (v == 16) { v = 8; e++; }
        if (e > 8) code = 0x7Eu;              /* saturate to 448 (finite max) */
        else code = ((uint32_t)(e + 7) << 3) | (v & 7u);
    } else if (e >= -10) {
        /* subnormal grid, quantum 2^-9; code 8 is continuous with min normal */
        code = apus_rne_shift(m23, 14 - e);   /* 0..8 */
    } else {
        code = 0;
    }
    return (uint8_t)(sign | code);
}

float apus_e4m3_dequant_f32(uint8_t c) {
    int e = (c >> 3) & 0xF, m = c & 7;
    float v = (e != 0) ? ldexpf((float)(8 + m), e - 10)   /* (1+m/8)*2^(e-7) */
                       : ldexpf((float)m, -9);            /* m*2^-9 subnormal */
    return (c & 0x80) ? -v : v;
}

float apus_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x; /* NaN: leave alone */
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

uint16_t apus_bf16_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(u >> 16); /* NaN */
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float apus_bf16_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

size_t apus_fp4_act_blocks(size_t K) {
    return (K + APUS_FP4_ACT_GROUP - 1) / APUS_FP4_ACT_GROUP;
}

void apus_fp4_act_quant_scalar(const float *x, size_t K,
                               uint8_t *codes, float *scales) {
    size_t nab = apus_fp4_act_blocks(K);
    for (size_t b = 0; b < nab; b++) {
        size_t lo = b * APUS_FP4_ACT_GROUP;
        size_t hi = lo + APUS_FP4_ACT_GROUP;
        if (hi > K) hi = K;
        float amax = 0.0f;
        for (size_t i = lo; i < hi; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        /* scale = 2^ceil(log2(amax * (1/448))) via the reference bit trick
         * (kernel.py fast_log2_ceil/fast_pow2): exponent of the FP32 product
         * plus 1 when any mantissa bit is set. amax >= 1e-4 keeps the product
         * normal, so no subnormal case to handle. */
        float p = amax * (1.0f / 448.0f);
        uint32_t pb;
        memcpy(&pb, &p, 4);
        int e = (int)((pb >> 23) & 0xFF) - 127 + ((pb & 0x007fffffu) != 0);
        uint32_t sbits = (uint32_t)(e + 127) << 23;
        float s;
        memcpy(&s, &sbits, 4);
        scales[b] = s;
        for (size_t i = lo; i < hi; i++)
            codes[i] = apus_e4m3_quant_f32(x[i] / s);
    }
}

#endif /* APUS_NUM_IMPLEMENTATION */
