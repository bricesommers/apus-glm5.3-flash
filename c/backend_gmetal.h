/*
 * c/backend_gmetal.h — optional Metal backend for the GLM-5.3-Flash dense
 * compute (M7b): BF16 GEMV/GEMM and the fused FP8-block-dequant + BF16
 * GEMM composition as Metal shaders over zero-copy unified-memory buffers.
 * Pure C11 interface (the implementation lives in c/backend_gmetal.mm,
 * Objective-C++, only in the `metal=1` build and the tests/m7b binaries).
 *
 * This is the GLM twin of the V4 c/backend_metal.h (which stays untouched
 * for the retained engine): the g-prefixed hook table lives in the
 * APUS_BF16_IMPLEMENTATION TU (c/bf16.h is linked into every GLM engine
 * binary), all-NULL by default = the pinned CPU kernels.
 * apus_gmetal_enable() (strong definition in c/backend_gmetal.mm; weak
 * stub in the bf16.h TU so the plain CPU binary links and behaves exactly
 * as before) initializes Metal and fills the table. Call sites
 * (c/bf16.h apus_bf16_gemv_mt/gemm_mt, c/gdsa.h apus_gdsa_fp8_linear) try
 * the hook first and fall back to the CPU kernel when the pointer is NULL
 * or the call returns nonzero — per-op fail-soft, never a crash.
 *
 * NUMERICS CONTRACT — BITWISE, not a tolerance class. The GLM CPU kernels
 * accumulate SEQUENTIALLY over k with two IEEE fp32 roundings per element
 * (a product rounding, then an add rounding; NO FMA — c/bf16.h header),
 * so a GPU thread that walks k in the same ascending order with plain
 * mul+add reproduces the CPU bits exactly (unlike the V4 fp8 path, whose
 * NEON-canonical 4-lane order the V4 shaders had to mirror). The shaders
 * are compiled with fast-math DISABLED (no contraction, no reassociation):
 *   - gbf16_gemm: inline exact bf16->f32 widen (bits << 16), sequential
 *     acc += wv*xv, RNE bf16 narrow of the output (the c/num.h
 *     apus_bf16_bits bit trick, NaN passing through as the high 16 bits).
 *     BITWISE == apus_bf16_gemv_mt / apus_bf16_gemm_mt at every shape
 *     (asserted in tests/m7b/test_gkernels.c).
 *   - gfp8blk_gemm: the m3g composition fused — per element, EXACT E4M3
 *     decode (NaN codes decode as +-480 like apus_e4m3_dequant_f32), ONE
 *     fp32 multiply by the F32 weight_scale_inv of its 128x128 block
 *     (ceil shapes supported), RNE bf16 round (kept in an fp32 container,
 *     which is bit-identical to narrow+widen), then the same sequential
 *     mul+add accumulation as gbf16_gemm. BITWISE ==
 *     apus_fp8blk_dequant + apus_bf16_gemm_mt (asserted).
 * The one measured edge: intermediates in the fp32 SUBNORMAL range
 * (|product| < 2^-126) are flushed to zero on the GPU while the CPU
 * accumulates them — observable only when an output's ENTIRE sum is
 * subnormal-magnitude (|delta| <= one bf16-subnormal ulp of output,
 * ~1e-39; measured in tests/m7b/test_gkernels.c). Unreachable from
 * normative data: a subnormal product needs a bf16-subnormal factor
 * (< 2^-126), which no gated fixture or real-scale forward produces;
 * and a 1e-39 logit delta cannot flip an argmax. Everything else —
 * including bf16-subnormal inputs whose products stay fp32-normal —
 * is bitwise.
 *
 * BUFFERS — zero-copy only (unified memory — no upload copies ever;
 * bytes_uploaded stays 0), two persistence classes:
 *   - REGISTERED (P4): c/gmodel.h registers every MODEL-OWNED dense weight
 *     region at open (apus_gmetal_register_region — shard-view tensors,
 *     owned dequant buffers; never freed until apus_gmodel_close
 *     unregisters them). The backend wraps each registered range ONCE
 *     (page-rounded vm_region walk + newBufferWithBytesNoCopy at
 *     registration) and reuses that MTLBuffer for every later op whose
 *     weight range it covers. At real scale this removes the per-op
 *     16k-page wrap/teardown churn the M7b ephemeral policy paid on every
 *     dispatch (~230 ops/token x 64 MiB KDA projections — the P2 4x
 *     --metal loss). Numerics unchanged: same shaders, same dispatch.
 *   - EPHEMERAL (M7b policy, unchanged): everything NOT registered —
 *     gcache expert payloads, activations, outputs, engine scratch — is
 *     wrapped per op and released before returning. Nothing is held across
 *     ops for those, so nothing is held across the M6 layer_end/RSS-guard
 *     boundary either — the tiered gcache may recycle or free expert
 *     payloads at any layer boundary without invalidating any GPU state
 *     (the M6 Metal-tier invariant by construction: only memory the MODEL
 *     owns for its whole lifetime can be registered, and gmodel
 *     unregisters before freeing).
 *
 * APUS_GMETAL_MIN_KB (default OFF — P4 real-scale change): BF16 weights
 * smaller than this stay on the CPU. Fixture-scale (tests/m7b README) the
 * GPU wins the big GEMV shapes (dense MLP x1.59, KDA projections) while
 * the CPU NEON-mt kernels win the small ones — but at REAL scale on the
 * 32 GB dev machine every bf16 decode shape measures a net LOSS
 * (bandwidth contention with the expert slab stream + compressed-page
 * stalls; P4 probes at 2 GiB tiered, 8-token steady state: CPU 11.79
 * s/tok, floor 32768 12.20, floor 65664 (dense MLP + head only) 11.59,
 * floor OFF 11.08), so the floor defaults to "everything stays CPU".
 * Set it explicitly (0 = offload all, the test setting; 32768 = the
 * fixture-tuned split) to re-enable — worth revisiting on hosts where
 * the GPU and the expert stream don't share a saturated memory
 * controller. APUS_GMETAL_FP8_MIN_KB (default 0): the fused FP8-block
 * path skips the CPU's per-call dequant materialization entirely
 * (measured x2.7-3.8 at the DSA shapes, and still a real-scale win), so
 * it offloads at every size. Both read once at apus_gmetal_enable.
 * APUS_GMETAL_MAX_M (default 1; P4, both hooks): activation-row count
 * above this stays on the CPU. The one-thread-per-(m,o) shaders re-read
 * the weight matrix per m-row (no m-blocking), so GPU weight traffic
 * scales xM while the CPU NEON-mt kernels share each W chunk across 4
 * rows: at M=1 (decode) the GPU wins on raw bandwidth; at prefill M the
 * traffic multiplier erases it (measured at real scale, P4: metal
 * prefill 108.7 s vs CPU 78.0 s before the gate). <= 0 = unlimited (the
 * test setting). Pure dispatch policy, like the size floors — both sides
 * are the bitwise-pinned kernels.
 *
 * Threading: the hooks are compute-thread only (the engine's forward is
 * single-threaded; the gcache I/O pool never calls GEMMs). Synchronous
 * execution, one command buffer per op, no locks.
 */
#ifndef APUS_BACKEND_GMETAL_H
#define APUS_BACKEND_GMETAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend hook table. All-NULL = CPU kernels (default). A hook returns 0
 * when it produced the result, nonzero = unsupported (caller runs the CPU
 * kernel). Shapes match the CPU helpers they replace:
 *  - bf16_gemm:     c/bf16.h apus_bf16_gemv_mt (M=1) / apus_bf16_gemm_mt —
 *                   w [O,K] BF16 codes, x [M,K] BF16 codes, y [M,O] BF16
 *                   codes out (sequential-k fp32 accumulate, RNE narrow).
 *  - fp8blk_linear: c/gdsa.h apus_gdsa_fp8_linear — codes [O,K] E4M3,
 *                   scales [ceil(O/128)*ceil(K/128)] F32, x/y as above
 *                   (the dequant+GEMM composition, fused). */
typedef struct {
    int (*bf16_gemm)(const uint16_t *w, const uint16_t *x, uint16_t *y,
                     size_t M, size_t O, size_t K);
    int (*fp8blk_linear)(const uint8_t *codes, const float *scales,
                         const uint16_t *x, uint16_t *y,
                         size_t M, size_t O, size_t K);
} ApusGmetalHooks;

extern ApusGmetalHooks apus_gmetal_hooks;

/* Enable the GLM Metal backend: initialize the device/queue/pipelines and
 * fill apus_gmetal_hooks. Returns 0 on success; nonzero (err filled) when
 * Metal is unavailable or the backend was not compiled in — the hooks
 * stay NULL and the engine runs the CPU kernels (fail-soft). Idempotent. */
int  apus_gmetal_enable(char *err, size_t errcap);
/* Detach the hooks (CPU kernels again) and release backend state. */
void apus_gmetal_disable(void);
int  apus_gmetal_is_enabled(void);

/* P4 persistent wraps: register a MODEL-OWNED weight region (shard-view
 * tensor or owned dequant buffer, alive until the matching unregister).
 * Its zero-copy wrap is created once and reused by every later hooked op
 * whose weight range it covers; unregistered pointers keep the ephemeral
 * per-op policy. Returns 0 on success, nonzero (fail-soft: the ops fall
 * back to ephemeral wraps) when the backend is disabled or the range
 * cannot be wrapped. Idempotent per pointer. Registration is
 * compute-thread only (gmodel open/close), no locks. */
int  apus_gmetal_register_region(const void *ptr, size_t len);
void apus_gmetal_unregister_region(const void *ptr);

/* Instrumentation: resident weight bytes wrapped zero-copy, GPU
 * dispatches submitted, ops offloaded per kernel family (the model gate
 * asserts these are nonzero — proof the GPU path actually ran). */
uint64_t apus_gmetal_bytes_wrapped(void);
uint64_t apus_gmetal_bytes_pinned(void);    /* P4: registered-region bytes */
uint64_t apus_gmetal_dispatches(void);
uint64_t apus_gmetal_offload_bf16(void);
uint64_t apus_gmetal_offload_fp8blk(void);

/* Direct entry points (kernel-level tests; same semantics as the hooks,
 * no hook-table indirection, no APUS_GMETAL_MIN_KB floor, Metal
 * initialized lazily). Return 0 on success, 1 when Metal/unsupported. */
int apus_gmetal_bf16_gemm(const uint16_t *w, const uint16_t *x,
                          uint16_t *y, size_t M, size_t O, size_t K);
int apus_gmetal_fp8blk_linear(const uint8_t *codes, const float *scales,
                              const uint16_t *x, uint16_t *y,
                              size_t M, size_t O, size_t K);

#ifdef __cplusplus
}
#endif

#endif /* APUS_BACKEND_GMETAL_H */
