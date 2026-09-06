/*
 * c/backend_gmetal.mm — Metal backend for the GLM-5.3-Flash dense compute
 * (M7b). Objective-C++ (clang++); built only into bin/apus_metal and the
 * tests/m7b binaries (make metal=1 / test-m7b); the plain CPU binary
 * never links it and is behaviorally untouched. The GLM twin of
 * c/backend_metal.mm (the V4 backend, untouched): same zero-copy
 * unified-memory machinery, GLM kernels and a different buffer policy.
 *
 * What runs on the GPU (all FP32 shader math, fast-math DISABLED — no
 * contraction, no reassociation, so acc = acc + wv*xv is TWO IEEE
 * roundings exactly like the pinned CPU kernels):
 *   - gbf16_gemm: BF16 GEMV/GEMM, the c/bf16.h semantics — inline exact
 *     widen (bits << 16), SEQUENTIAL-k mul+add per output (the CPU
 *     kernels' canonical order — no reorder class needed, GPU == CPU
 *     BITWISE), RNE bf16 narrow of the output (c/num.h bit trick).
 *     One thread per (m, o); M chunked so the grid stays < 2^28 threads.
 *   - gfp8blk_gemm: the m3g composition fused — EXACT E4M3 decode
 *     (NaN codes as +-480), one fp32 multiply by the F32
 *     weight_scale_inv of the 128x128 block (ceil shapes), RNE bf16
 *     round in an fp32 container (== narrow+widen), then the same
 *     sequential accumulation. BITWISE == apus_fp8blk_dequant +
 *     apus_bf16_gemm_mt; skips materializing the BF16 weight entirely.
 * Everything else (KDA conv/recurrent delta rule, the Lightning indexer
 * pool rebuild/scoring, mHC/Sinkhorn, RMSNorm, router scoring, SwiGLU,
 * sampling) stays on the CPU this milestone — see tests/m7b/README.md.
 *
 * Floors: APUS_GMETAL_MIN_KB (bf16, default OFF at real scale — the P4
 * probes measured every bf16 decode shape a net loss on the 32 GB dev
 * machine; set the env to re-enable) and APUS_GMETAL_FP8_MIN_KB (fused
 * path, default 0 — skipping the per-call dequant materialization pays
 * at every size, fixture AND real scale). Below a floor the hook returns
 * "unsupported" and the caller runs the pinned CPU kernel (per-op
 * fail-soft). APUS_GMETAL_MAX_M (default 1) keeps prefill GEMMs on the
 * CPU (the shaders have no m-blocking — the header has the numbers).
 *
 * Buffers: zero-copy only, two persistence classes (the header has the
 * full contract): REGISTERED model-owned weight regions (wrapped once at
 * apus_gmetal_register_region, reused thereafter — P4; only memory the
 * model owns for its whole lifetime may be registered) and EPHEMERAL
 * per-op wraps for everything else (gcache expert payloads, activations,
 * outputs — the M7b policy: nothing held across ops, so the M6
 * layer_end/RSS-guard boundary can never alias a dead payload).
 *
 * Synchronous execution: one command buffer per op chunk, commit +
 * waitUntilCompleted. The engine's forward is single-threaded; no locks.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend_gmetal.h"

/* ========================================================================
 * Shader source (compiled at apus_gmetal_enable with fastMathEnabled = NO).
 * The kernels mirror c/bf16.h (apus_bf16_gemv_scalar) and c/fp8blk.h
 * (apus_fp8blk_dequant_scalar) step for step; the sequential-k order with
 * two roundings per element IS the CPU canonical order, so GPU == CPU
 * bitwise — verified in tests/m7b/test_gkernels.c.
 * ====================================================================== */

static NSString *const kGShaderSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

/* apus_bf16_f32: exact widen, f32 bits = (uint32)code << 16. */
inline float bf16w(ushort b) {
    return as_type<float>((uint)b << 16);
}

/* apus_bf16_bits: RNE to BF16, returned as the 16-bit code; NaN passes
 * through as the high 16 bits. */
inline ushort bf16n(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)(u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

/* apus_bf16_round: RNE to BF16 kept in an FP32 container (== narrow then
 * exact widen); NaN untouched. */
inline float bf16r(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    return as_type<float>(u);
}

/* apus_e4m3_dequant_f32, exact (NaN codes decode as +-480, as in c/num.h
 * via the FP16 bit-placement identity: h16 = (c&0x80)<<8 | (c&0x7F)<<7 is
 * exactly 2^-8 x the E4M3 value for normals AND subnormals). */
inline float e4m3_deq(uint c) {
    uint h = ((c & 0x80u) << 8) | ((c & 0x7Fu) << 7);
    return (float)as_type<half>((ushort)h) * 256.0f;   /* exact widen + scale */
}

/* BF16 GEMV/GEMM (c/bf16.h apus_bf16_gemv_scalar order): one thread per
 * (m, o), sequential ascending k, acc += wv*xv (two roundings per
 * element, NO FMA — fast math is off), RNE bf16 out. */
kernel void gbf16_gemm(device const uchar *w8 [[buffer(0)]],
                       device const uchar *x8 [[buffer(1)]],
                       device uchar *y8 [[buffer(2)]],
                       constant uint &O [[buffer(3)]],
                       constant uint &K [[buffer(4)]],
                       constant ulong &woff [[buffer(5)]],
                       constant ulong &xoff [[buffer(6)]],
                       constant ulong &yoff [[buffer(7)]],
                       uint gid [[thread_position_in_grid]]) {
    device const ushort *w = (device const ushort *)(w8 + woff);
    device const ushort *x = (device const ushort *)(x8 + xoff);
    device ushort *y = (device ushort *)(y8 + yoff);
    uint o = gid % O;
    uint m = gid / O;
    ulong wb = (ulong)o * K, xb = (ulong)m * K;
    float acc = 0.0f;
    for (uint k = 0u; k < K; k++)
        acc = acc + bf16w(w[wb + k]) * bf16w(x[xb + k]);
    y[(ulong)m * O + o] = bf16n(acc);
}

/* Fused FP8-block dequant + BF16 GEMM (c/fp8blk.h + c/bf16.h
 * composition): per element, rne(e4m3(code) * scale) — one fp32 rounding
 * then the RNE narrow — then the same sequential mul+add as gbf16_gemm.
 * ws: [ceil(O/128) * nkb] F32, nkb = ceil(K/128); edge blocks partial. */
kernel void gfp8blk_gemm(device const uchar *w8 [[buffer(0)]],
                         device const uchar *s8 [[buffer(1)]],
                         device const uchar *x8 [[buffer(2)]],
                         device uchar *y8 [[buffer(3)]],
                         constant uint &O [[buffer(4)]],
                         constant uint &K [[buffer(5)]],
                         constant uint &nkb [[buffer(6)]],
                         constant ulong &woff [[buffer(7)]],
                         constant ulong &soff [[buffer(8)]],
                         constant ulong &xoff [[buffer(9)]],
                         constant ulong &yoff [[buffer(10)]],
                         uint gid [[thread_position_in_grid]]) {
    device const uchar *wc = w8 + woff;
    device const float *ws = (device const float *)(s8 + soff);
    device const ushort *x = (device const ushort *)(x8 + xoff);
    device ushort *y = (device ushort *)(y8 + yoff);
    uint o = gid % O;
    uint m = gid / O;
    ulong wb = (ulong)o * K, xb = (ulong)m * K;
    ulong sb = (ulong)(o / 128u) * nkb;
    float acc = 0.0f;
    for (uint k = 0u; k < K; k++) {
        float wv = bf16r(e4m3_deq(wc[wb + k]) * ws[sb + (k >> 7)]);
        acc = acc + wv * bf16w(x[xb + k]);
    }
    y[(ulong)m * O + o] = bf16n(acc);
}
)MSL";

/* ========================================================================
 * Context
 * ====================================================================== */

typedef struct {
    uintptr_t lo, hi;           /* page-rounded wrapped range */
    uintptr_t orig;             /* registered base pointer (unregister key) */
    id<MTLBuffer> buf;
} GMtlReg;

struct GMtlCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLComputePipelineState> pso_bf16 = nil;
    id<MTLComputePipelineState> pso_fp8blk = nil;
    uint64_t min_bytes = UINT64_MAX;    /* APUS_GMETAL_MIN_KB (bf16);
                                           default OFF (P4 real-scale) */
    uint64_t fp8_min_bytes = 0;            /* APUS_GMETAL_FP8_MIN_KB */
    int64_t  max_m = 1;                    /* APUS_GMETAL_MAX_M (P4) */
    uint64_t bytes_wrapped = 0;         /* ephemeral zero-copy weight views */
    uint64_t bytes_pinned = 0;          /* P4: registered-region bytes */
    uint64_t dispatches = 0;
    uint64_t off_bf16 = 0;              /* hooked bf16 GEMM ops taken */
    uint64_t off_fp8 = 0;               /* hooked fp8blk ops taken */
    /* P4 registered regions (model-owned weights — see the header) */
    GMtlReg *regs = NULL;
    int regs_n = 0, regs_cap = 0;
};

static GMtlCtx *g_gctx = NULL;

/* Zero-copy wrap of an existing CPU allocation (the V4 backend's walk):
 * verify the page-rounded range [lo, hi) is covered by contiguous mapped
 * vm_regions with the needed protection, then wrap it (no deallocator —
 * the engine owns the memory). The tensor starts *off bytes into the
 * returned buffer; the rounded range goes to lo_out/hi_out. nil when
 * not possible (caller falls back to CPU / ephemeral). */
static id<MTLBuffer> gwrap_create(GMtlCtx *c, const void *ptr, size_t len,
                                  int need_write, size_t *off,
                                  uintptr_t *lo_out, uintptr_t *hi_out) {
    uintptr_t lo = (uintptr_t)ptr & ~(uintptr_t)4095;
    uintptr_t hi = ((uintptr_t)ptr + len + 4095) & ~(uintptr_t)4095;
    if (hi <= lo) return nil;
    vm_prot_t want = VM_PROT_READ | (need_write ? VM_PROT_WRITE : 0);
    uintptr_t cur = lo;
    while (cur < hi) {
        vm_address_t q = (vm_address_t)cur;
        vm_size_t rsize = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t obj = MACH_PORT_NULL;
        if (vm_region_64(mach_task_self(), &q, &rsize,
                         VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info,
                         &count, &obj) != KERN_SUCCESS)
            return nil;
        if ((uintptr_t)q > cur || (uintptr_t)q + rsize <= cur)
            return nil;   /* gap / unexpected layout */
        if ((info.protection & want) != want) return nil;
        cur = (uintptr_t)q + rsize;
    }
    *off = (uintptr_t)ptr - lo;
    *lo_out = lo;
    *hi_out = hi;
    return [c->dev newBufferWithBytesNoCopy:(void *)lo
                                     length:(NSUInteger)(hi - lo)
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

/* Ephemeral wrap (the M7b policy): created per op, released after the
 * synchronous dispatch. */
static id<MTLBuffer> gwrap(GMtlCtx *c, const void *ptr, size_t len,
                           int need_write, size_t *off) {
    uintptr_t lo, hi;
    return gwrap_create(c, ptr, len, need_write, off, &lo, &hi);
}

/* P4: persistent-wrap lookup for a READ range (weights only). Returns the
 * registered MTLBuffer (NOT retained — the registration owns it) with
 * *off = byte offset of ptr into it, or nil. */
static id<MTLBuffer> greg_lookup(GMtlCtx *c, const void *ptr, size_t len,
                                 size_t *off) {
    uintptr_t p = (uintptr_t)ptr;
    for (int i = 0; i < c->regs_n; i++) {
        GMtlReg *r = &c->regs[i];
        if (p >= r->lo && p + len <= r->hi) {
            *off = (size_t)(p - r->lo);
            return r->buf;
        }
    }
    return nil;
}

/* One synchronous dispatch: encode with `encode`, commit, wait. Returns 0
 * on success, 1 on GPU error (fail-soft: caller falls back to CPU). */
typedef void (^GEncodeFn)(id<MTLComputeCommandEncoder> enc);
static int grun_op(GMtlCtx *c, GEncodeFn encode) {
    id<MTLCommandBuffer> cb = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    c->dispatches++;
    return [cb status] == MTLCommandBufferStatusError ? 1 : 0;
}

#define APUS_GMTL_MAXGRID (1u << 28)   /* threads per dispatch chunk */

/* ========================================================================
 * Ops (all return 0 = done, 1 = unsupported -> CPU fallback)
 * ====================================================================== */

static int gmtl_bf16(GMtlCtx *c, const uint16_t *w, const uint16_t *x,
                     uint16_t *y, size_t M, size_t O, size_t K) {
    if (!c || !M || !O || !K || (uint64_t)M * O >= (1ull << 31)) return 1;
    size_t woff = 0, xoff = 0, yoff = 0;
    /* P4: a registered (model-owned) weight range reuses its persistent
     * wrap; anything else takes the ephemeral per-op policy. */
    id<MTLBuffer> wb = greg_lookup(c, w, O * K * sizeof(uint16_t), &woff);
    int wb_reg = wb != nil;
    if (!wb) wb = gwrap(c, w, O * K * sizeof(uint16_t), 0, &woff);
    id<MTLBuffer> xb = gwrap(c, x, M * K * sizeof(uint16_t), 0, &xoff);
    id<MTLBuffer> yb = gwrap(c, y, M * O * sizeof(uint16_t), 1, &yoff);
    if (!wb || !xb || !yb) {
        if (wb && !wb_reg) [wb release];
        if (xb) [xb release];
        if (yb) [yb release];
        return 1;
    }
    uint Ou = (uint)O, Ku = (uint)K;
    size_t mchunk = O < APUS_GMTL_MAXGRID ? APUS_GMTL_MAXGRID / O : 1;
    int rc = 0;
    for (size_t m0 = 0; m0 < M && !rc; m0 += mchunk) {
        uint mc = (uint)(M - m0 < mchunk ? M - m0 : mchunk);
        uint64_t xo = xoff + (uint64_t)m0 * K * sizeof(uint16_t);
        uint64_t yo = yoff + (uint64_t)m0 * O * sizeof(uint16_t);
        uint64_t wo = woff;
        rc = grun_op(c, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:c->pso_bf16];
            [enc setBuffer:wb offset:0 atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:yb offset:0 atIndex:2];
            [enc setBytes:&Ou length:4 atIndex:3];
            [enc setBytes:&Ku length:4 atIndex:4];
            [enc setBytes:&wo length:8 atIndex:5];
            [enc setBytes:&xo length:8 atIndex:6];
            [enc setBytes:&yo length:8 atIndex:7];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)mc * Ou, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
    }
    if (!wb_reg) [wb release];
    [xb release];
    [yb release];
    if (!rc && !wb_reg) c->bytes_wrapped += O * K * sizeof(uint16_t);
    return rc;
}

static int gmtl_fp8blk(GMtlCtx *c, const uint8_t *codes, const float *ws,
                       const uint16_t *x, uint16_t *y,
                       size_t M, size_t O, size_t K) {
    if (!c || !M || !O || !K || (uint64_t)M * O >= (1ull << 31)) return 1;
    size_t nkb = (K + 127) / 128;
    size_t nsb = ((O + 127) / 128) * nkb * sizeof(float);
    size_t woff = 0, soff = 0, xoff = 0, yoff = 0;
    id<MTLBuffer> cb = greg_lookup(c, codes, O * K, &woff);
    int cb_reg = cb != nil;
    if (!cb) cb = gwrap(c, codes, O * K, 0, &woff);
    id<MTLBuffer> sb = greg_lookup(c, ws, nsb, &soff);
    int sb_reg = sb != nil;
    if (!sb) sb = gwrap(c, ws, nsb, 0, &soff);
    id<MTLBuffer> xb = gwrap(c, x, M * K * sizeof(uint16_t), 0, &xoff);
    id<MTLBuffer> yb = gwrap(c, y, M * O * sizeof(uint16_t), 1, &yoff);
    if (!cb || !sb || !xb || !yb) {
        if (cb && !cb_reg) [cb release];
        if (sb && !sb_reg) [sb release];
        if (xb) [xb release];
        if (yb) [yb release];
        return 1;
    }
    uint Ou = (uint)O, Ku = (uint)K, nkb32 = (uint)nkb;
    size_t mchunk = O < APUS_GMTL_MAXGRID ? APUS_GMTL_MAXGRID / O : 1;
    int rc = 0;
    for (size_t m0 = 0; m0 < M && !rc; m0 += mchunk) {
        uint mc = (uint)(M - m0 < mchunk ? M - m0 : mchunk);
        uint64_t xo = xoff + (uint64_t)m0 * K * sizeof(uint16_t);
        uint64_t yo = yoff + (uint64_t)m0 * O * sizeof(uint16_t);
        uint64_t wo = woff, so = soff;
        rc = grun_op(c, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:c->pso_fp8blk];
            [enc setBuffer:cb offset:0 atIndex:0];
            [enc setBuffer:sb offset:0 atIndex:1];
            [enc setBuffer:xb offset:0 atIndex:2];
            [enc setBuffer:yb offset:0 atIndex:3];
            [enc setBytes:&Ou length:4 atIndex:4];
            [enc setBytes:&Ku length:4 atIndex:5];
            [enc setBytes:&nkb32 length:4 atIndex:6];
            [enc setBytes:&wo length:8 atIndex:7];
            [enc setBytes:&so length:8 atIndex:8];
            [enc setBytes:&xo length:8 atIndex:9];
            [enc setBytes:&yo length:8 atIndex:10];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)mc * Ou, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
    }
    if (!cb_reg) [cb release];
    if (!sb_reg) [sb release];
    [xb release];
    [yb release];
    if (!rc && !cb_reg) c->bytes_wrapped += O * K + nsb;
    return rc;
}

/* ========================================================================
 * Public C API (strong definitions; weak stubs live in the bf16.h TU)
 * ====================================================================== */

static int gmtl_init(char *err, size_t errcap) {
    if (g_gctx) return 0;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            snprintf(err, errcap, "no Metal device");
            return -1;
        }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        /* FP32 IEEE semantics: no fast-math contraction/reassociation —
         * the GLM CPU kernels forbid FMA (two roundings per element), so
         * the shaders must not contract either. (mathMode supersedes
         * fastMathEnabled on macOS 15+.) */
        if ([opts respondsToSelector:@selector(setMathMode:)])
            opts.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *nserr = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kGShaderSrc
                                               options:opts
                                                 error:&nserr];
        if (!lib) {
            snprintf(err, errcap, "shader compile: %s",
                     nserr ? [[nserr localizedDescription] UTF8String]
                           : "unknown");
            return -1;
        }
        GMtlCtx *c = new GMtlCtx();
        c->dev = dev;
        c->q = [dev newCommandQueue];
        if (!c->q) { snprintf(err, errcap, "no command queue"); delete c; return -1; }
        struct { const char *name; id<MTLComputePipelineState> *slot; } t[] = {
            { "gbf16_gemm",  &c->pso_bf16 },
            { "gfp8blk_gemm", &c->pso_fp8blk },
        };
        for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:t[i].name]];
            *t[i].slot = fn ? [dev newComputePipelineStateWithFunction:fn
                                                                 error:&nserr]
                            : nil;
            if (!*t[i].slot) {
                snprintf(err, errcap, "pipeline %s failed", t[i].name);
                delete c;
                return -1;
            }
        }
        /* P4 real-scale floor policy (the header has the numbers): the
         * bf16 GEMV offload is a NET LOSS at every real decode shape on
         * the 32 GB dev machine (bandwidth contention with the expert
         * stream + compressed-page stalls; fixture-scale wins do not
         * transfer), so the bf16 floor defaults to OFF (keep the
         * initializer); setting APUS_GMETAL_MIN_KB re-enables (0 =
         * offload all — the test setting; 32768 = the M7b fixture-tuned
         * split, worth revisiting on bigger-RAM hosts). */
        const char *e = getenv("APUS_GMETAL_MIN_KB");
        if (e && *e) {
            long kb = atol(e);
            if (kb < 0) kb = 0;
            c->min_bytes = (uint64_t)kb << 10;
        }
        long fkb = 0;
        e = getenv("APUS_GMETAL_FP8_MIN_KB");
        if (e && *e) fkb = atol(e);
        if (fkb < 0) fkb = 0;
        c->fp8_min_bytes = (uint64_t)fkb << 10;
        /* P4 M-gate (the header has the rationale): default decode-only
         * offload; <= 0 = unlimited (tests). */
        long mm = 1;
        e = getenv("APUS_GMETAL_MAX_M");
        if (e && *e) mm = atol(e);
        c->max_m = mm;
        g_gctx = c;
    }
    return 0;
}

/* --- hook trampolines (the APUS_GMETAL_MIN_KB floor lives here) --- */
static int hook_bf16_gemm(const uint16_t *w, const uint16_t *x,
                          uint16_t *y, size_t M, size_t O, size_t K) {
    GMtlCtx *c = g_gctx;
    if (!c || O * K * sizeof(uint16_t) < c->min_bytes) return 1;
    if (c->max_m > 0 && (int64_t)M > c->max_m) return 1;   /* P4 M-gate */
    int rc = gmtl_bf16(c, w, x, y, M, O, K);
    if (!rc) c->off_bf16++;
    return rc;
}
static int hook_fp8blk_linear(const uint8_t *codes, const float *scales,
                              const uint16_t *x, uint16_t *y,
                              size_t M, size_t O, size_t K) {
    GMtlCtx *c = g_gctx;
    if (!c || O * K < c->fp8_min_bytes) return 1;
    if (c->max_m > 0 && (int64_t)M > c->max_m) return 1;   /* P4 M-gate */
    int rc = gmtl_fp8blk(c, codes, scales, x, y, M, O, K);
    if (!rc) c->off_fp8++;
    return rc;
}

int apus_gmetal_enable(char *err, size_t errcap) {
    char e2[256];
    if (!err) { err = e2; errcap = sizeof e2; }
    if (gmtl_init(err, errcap)) return -1;
    apus_gmetal_hooks.bf16_gemm = hook_bf16_gemm;
    apus_gmetal_hooks.fp8blk_linear = hook_fp8blk_linear;
    return 0;
}

int apus_gmetal_register_region(const void *ptr, size_t len) {
    GMtlCtx *c = g_gctx;
    if (!c || !ptr || !len) return 1;
    uintptr_t orig = (uintptr_t)ptr;
    for (int i = 0; i < c->regs_n; i++)
        if (c->regs[i].orig == orig) return 0;      /* idempotent */
    size_t off = 0;
    uintptr_t lo = 0, hi = 0;
    id<MTLBuffer> b = gwrap_create(c, ptr, len, 0, &off, &lo, &hi);
    if (!b) return 1;    /* fail-soft: ops on this range stay ephemeral */
    if (c->regs_n == c->regs_cap) {
        int ncap = c->regs_cap ? 2 * c->regs_cap : 64;
        GMtlReg *nr = (GMtlReg *)realloc(c->regs,
                                         (size_t)ncap * sizeof *nr);
        if (!nr) { [b release]; return 1; }
        c->regs = nr;
        c->regs_cap = ncap;
    }
    c->regs[c->regs_n].lo = lo;
    c->regs[c->regs_n].hi = hi;
    c->regs[c->regs_n].orig = orig;
    c->regs[c->regs_n].buf = b;
    c->regs_n++;
    c->bytes_pinned += (uint64_t)(hi - lo);
    return 0;
}

void apus_gmetal_unregister_region(const void *ptr) {
    GMtlCtx *c = g_gctx;
    if (!c || !ptr) return;
    uintptr_t orig = (uintptr_t)ptr;
    for (int i = 0; i < c->regs_n; i++)
        if (c->regs[i].orig == orig) {
            [c->regs[i].buf release];
            c->bytes_pinned -= c->regs[i].hi - c->regs[i].lo;
            c->regs[i] = c->regs[--c->regs_n];
            return;
        }
}

void apus_gmetal_disable(void) {
    memset(&apus_gmetal_hooks, 0, sizeof apus_gmetal_hooks);
    GMtlCtx *c = g_gctx;
    g_gctx = NULL;
    if (c) {
        for (int i = 0; i < c->regs_n; i++)
            [c->regs[i].buf release];    /* registered wraps */
        free(c->regs);
        delete c;        /* ARC off: remaining ObjC members leak harmlessly
                            at exit; disable is test/teardown only */
    }
}

int apus_gmetal_is_enabled(void) {
    return g_gctx && apus_gmetal_hooks.bf16_gemm;
}

uint64_t apus_gmetal_bytes_wrapped(void) { return g_gctx ? g_gctx->bytes_wrapped : 0; }
uint64_t apus_gmetal_bytes_pinned(void) { return g_gctx ? g_gctx->bytes_pinned : 0; }
uint64_t apus_gmetal_dispatches(void) { return g_gctx ? g_gctx->dispatches : 0; }
uint64_t apus_gmetal_offload_bf16(void) { return g_gctx ? g_gctx->off_bf16 : 0; }
uint64_t apus_gmetal_offload_fp8blk(void) { return g_gctx ? g_gctx->off_fp8 : 0; }

/* --- direct entry points (kernel-level tests; lazy init, no floor) --- */

int apus_gmetal_bf16_gemm(const uint16_t *w, const uint16_t *x,
                          uint16_t *y, size_t M, size_t O, size_t K) {
    char e[128];
    if (gmtl_init(e, sizeof e)) return 1;
    return gmtl_bf16(g_gctx, w, x, y, M, O, K);
}

int apus_gmetal_fp8blk_linear(const uint8_t *codes, const float *scales,
                              const uint16_t *x, uint16_t *y,
                              size_t M, size_t O, size_t K) {
    char e[128];
    if (gmtl_init(e, sizeof e)) return 1;
    return gmtl_fp8blk(g_gctx, codes, scales, x, y, M, O, K);
}
