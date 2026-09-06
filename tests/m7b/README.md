# tests/m7b — GLM Metal backend: dense compute on the Apple GPU (optional)

Milestone M7b: the **optional** GLM Metal backend
(`c/backend_gmetal.mm` + `c/backend_gmetal.h`) offloads the
GLM-5.3-Flash matmuls to the Apple GPU so the CPU is freer for
expert streaming. FP32 shader math only, fast-math disabled, zero-copy
unified-memory buffers, per-op fail-soft to the pinned CPU kernels.
(The V4 backend + its test_kernels/bench_metal legs were removed
2026-09-05 with the V4 engine cleanup — the full V4 Metal stack remains
in `../Apus`.)

```
make metal=1 apus    # builds bin/apus_metal (== make bin/apus_metal) — the GLM backend
make test-m7b        # GLM kernel/model gates + m7a server suite on the metal binary
make ubsan-m7b       # same, UBSan on the C side
make bench-m7b       # CPU-vs-Metal microbenches (informational, bench_gmetal)
bin/apus_metal run --model DIR --metal ...   # or APUS_METAL=1
```

**CPU stays the default and is untouched behaviorally**: `bin/apus` never
links the backend; every hook defaults to NULL; with no `--metal` flag
the `bin/apus_metal` binary runs the identical CPU kernels. All prior
suites (m2…m7a) green, including under UBSan.

## GLM backend (c/backend_gmetal.mm) — what is offloaded

Two hook families, filled by `apus_gmetal_enable()` into
`ApusGmetalHooks apus_gmetal_hooks` (defined in the `APUS_BF16_IMPLEMENTATION`
TU — c/bf16.h is linked into every GLM engine binary; **weak stubs** there
let the plain CPU binary link):

| hook | call site | semantics |
|---|---|---|
| `bf16_gemm` | `c/bf16.h` `apus_bf16_gemv_mt` / `apus_bf16_gemm_mt` — catches **every** GLM BF16 matmul: KDA projections, DSA BF16 projections (kv_b, indexer), router gate, dense-MLP/shared/routed-expert GEMVs, DSA attention-score GEMMs, the LM head | BF16 codes in, sequential-k FP32 accumulate (two roundings per element, NO FMA), RNE BF16 out |
| `fp8blk_linear` | `c/gdsa.h` `apus_gdsa_fp8_linear` (q_a/q_b/kv_a/o_proj of every DSA layer) | the m3g composition fused: EXACT E4M3 decode × F32 `weight_scale_inv` (128×128 blocks, ceil shapes) → RNE BF16 → the same sequential accumulate — **skips materializing the BF16 weight** |

**Not offloaded** (CPU, this milestone): the KDA conv + chunked/recurrent
delta-rule state math (FP32 state, ordering-sensitive), the Lightning
indexer pool rebuild/scoring/top-k (selection-critical), mHC/Sinkhorn,
RMSNorm, router sigmoid/topk/norm, SwiGLU + clamps, the MoE accumulation
order, sampling, and the gcache dequant-on-fill (I/O-worker CPU — its
BF16 payloads then flow through the `bf16_gemm` hook like eager ones).

Offload floors (perf heuristics, measured on the dev M1 Pro; numerics are
size-independent): `APUS_GMETAL_MIN_KB` (BF16, default **32768** = 32 MiB —
the NEON-mt CPU kernels win the small GEMV shapes: expert 16 MiB x0.85,
router 2.25 MiB x0.69; the GPU wins the big ones: dense MLP x1.59, head
parity at the DRAM bound) and `APUS_GMETAL_FP8_MIN_KB` (default **0** —
the fused DSA path skips the CPU's per-call dequant materialization and
streams half the bytes: measured **x2.7–4.0** at q_b/o_proj, worth it at
every size). `APUS_GMETAL_MIN_KB=0` offloads everything (the test
setting).

## Numerics policy — BITWISE, no tolerance tier

Unlike the V4 FP8 path (whose NEON-canonical 4-lane order the V4 shaders
had to mirror to reach bitwise), the GLM CPU kernels
(`c/bf16.h`) accumulate **sequentially over k with two IEEE roundings per
element** (product, then add; `-ffp-contract=off`, no FMA anywhere). A
GPU thread walking k in the same order with plain mul+add — shaders
compiled with `MTLMathModeSafe`, no contraction/reassociation — reproduces
the CPU bits **exactly**. The fused FP8 shader likewise reproduces the
dequant (exact E4M3 decode, one FP32 scale multiply, RNE narrow) element
for element before accumulating. Asserted, not assumed:

- **test_gkernels (75 checks, 0 failures)**: GPU == CPU **bitwise** on
  every output of the battery — the real GLM shapes (experts
  2048×4096/4096×2048, dense 12288×4096, DSA q_a/q_b/kv_a/o_proj, router
  288×4096, head slice), odd/partial shapes (K=1/7/31/100/4095/4097,
  O∈{1,3,127,129}, M∈{1,2,3,5,9}), ceil-shaped partial 128-blocks,
  all-256-E4M3-code coverage (subnormals; NaN codes → ±480), F32 scales
  2^-24…2^16, zero/±inf/NaN-parity rows, and a scalar-anchor cross-check.
- **test_gmodel (9 checks, 0 failures)**: on the m5g container, the FULL
  logit stream (70-token prefill + 12 greedy decodes) is
  **memcmp-identical CPU vs Metal**, eager AND 1-slot tiered (digest
  `9ed892c248e9d606` on the dev host, identical at APUS_THREADS=1/4/8);
  Metal run twice identical (GPU determinism); eager-CPU == tiered-CPU
  re-asserted. No teacher-forcing, no tolerance tier — selections are
  bitwise because the logits are.

**The one measured divergence class (hard gate 1 documentation)**:
*fp32-subnormal intermediates*. The GPU flushes subnormal products/
partials to zero; the CPU accumulates them. Observable only when an
output's ENTIRE sum is subnormal-magnitude: pure bf16-subnormal weight
rows give |Δ| = 1.7e-39 (cpu bf16 `0x0013` vs gpu `0x0000`) — bounded by
one bf16-subnormal ulp of output. Unreachable from normative data: a
subnormal product needs a bf16-subnormal factor (< 2^-126), which no
gated fixture or real-scale forward produces, and a 1e-39 logit delta
cannot flip an argmax (the m7a margins are ≥ 1824). bf16-subnormal inputs
whose products stay fp32-normal are bitwise (gated).

## Buffer management — zero-copy, two persistence classes (P4 update)

Both classes use `newBufferWithBytesNoCopy` over the page-rounded
`vm_region` (validated walk, unified memory — **no upload copies ever**),
synchronous dispatch (commit + `waitUntilCompleted`):

- **EPHEMERAL (the M7b policy, unchanged)**: gcache expert payloads,
  activations, outputs, engine scratch — wrapped per op, released before
  returning. Nothing is held across ops for these, so nothing is held
  across the M6 `layer_end`/RSS-guard boundary either — the tiered gcache
  may recycle or free expert payloads at any layer boundary without
  invalidating any GPU state (**the M6 Metal-tier invariant holds by
  construction**, not by an invalidation protocol).
- **REGISTERED (P4)**: `c/gmodel.h` registers every MODEL-OWNED dense
  weight region at open (`apus_gmetal_register_region` — shard-view
  tensors and owned dequant buffers; never the expert payloads, which are
  gcache-owned and recycled). The wrap is created ONCE and reused by every
  later op whose weight range it covers; `apus_gmodel_close` unregisters
  before freeing. Only memory the model owns for its whole lifetime can
  be registered, so the invariant above is preserved. Rationale: at real
  scale the M7b ephemeral policy wrapped/unwrapped ~230 dense-weight
  ranges per token (64 MiB KDA projections — a 16k-page vm_region walk +
  GPU page-table churn per op), the dominant term in the P2 4x `--metal`
  loss. Numerics unchanged: same shaders, same dispatch order (gated by
  leg E of test_gmodel: open with Metal enabled, stream memcmp-identical,
  wraps released at close). RSS cost is zero beyond the engine's own
  pages (zero-copy) either way.

`APUS_GMETAL_MIN_KB` floor: weights below it return "unsupported" from
the hook and the caller runs the pinned CPU kernel (per-op fail-soft —
bitwise by construction, asserted in test_gkernels with a ~100 GB floor).

## Measured results (MacBook Pro M1 Pro, 32 GB)

### GLM kernel battery (test_gkernels, 75 checks, 0 failures)

- bf16 GEMV/GEMM: **100% bitwise** vs `apus_bf16_gemv_mt`/`gemm_mt` and
  the scalar anchor across the whole battery (see above).
- fused fp8blk linear: **100% bitwise** vs
  `apus_fp8blk_dequant` + `apus_bf16_gemm_mt`, incl. partial blocks and
  the full E4M3 code space.
- denormal-flush class: |Δ| = 1.7e-39 on pure-subnormal rows (measured,
  bounded, documented above); fp32-subnormal probe (tiny×tiny → exact 0
  both sides) bitwise.
- zero-copy engaged everywhere: 2.8 GB wrapped across the battery, 0
  uploaded (there is no upload path).

### GLM model gate (test_gmodel, 12 checks, 0 failures)

- eager CPU vs Metal: stream digest `9ed892c248e9d606` both sides;
  3,278 bf16 + 52 fp8blk ops offloaded over the run (proof the GPU path
  ran — the fixture's tiny shapes go through the hook with
  `APUS_GMETAL_MIN_KB=0`).
- tiered 1-slot CPU vs Metal: identical digest (eviction churn + payload
  recycling invisible to the backend by construction).
- Metal rerun: identical. APUS_THREADS=1/4/8: identical output.
- P4 persistent wraps (leg E): opening the model WITH Metal enabled
  registers the model-owned dense weights (5.2 MiB pinned at fixture
  scale, `apus_gmetal_bytes_pinned`), the stream stays memcmp-identical,
  and closing the model with the backend still enabled releases every
  registered wrap (`bytes_pinned` back to 0).

### Server-level

The full **m7a server suite passes against the Metal binary**
(`APUS_BIN=bin/apus_metal APUS_METAL=1 APUS_GMETAL_MIN_KB=0`), O2 and
UBSan — the GLM parrot streams stay bitwise == the oracle through the
offload.

### Performance (informational, `make bench-m7b` → bench_gmetal)

Decode GEMV (M=1) / small-prefill GEMM (M=32), effective weight-streaming
bandwidth:

| op | shape | CPU | Metal | speedup |
|---|---|---|---|---|
| bf16 expert gate/up | 2048×4096 M1 | 3.43 ms (4.9 GB/s) | 4.02 ms (4.2 GB/s) | x0.85 |
| bf16 expert down | 4096×2048 M1 | 2.96 ms (5.7 GB/s) | 3.08 ms (5.4 GB/s) | x0.96 |
| bf16 dense MLP | 12288×4096 M1 | 19.9 ms (5.1 GB/s) | 12.5 ms (8.0 GB/s) | x1.59 |
| bf16 router | 288×4096 M1 | 0.97 ms (2.4 GB/s) | 1.41 ms (1.7 GB/s) | x0.69 |
| bf16 head slice | 40960×4096 M1 | 33.5 ms (10.0 GB/s) | 33.2 ms (10.1 GB/s) | x1.01 |
| bf16 expert gate/up | 2048×4096 M32 | 33.2 ms | 30.9 ms | x1.07 |
| bf16 head slice | 40960×4096 M32 | 697.8 ms | 634.9 ms | x1.10 |
| fused fp8 dsa q_b | 16384×1536 M1 | 16.7 ms (dequant+gemm) | 6.19 ms | x2.70 |
| fused fp8 dsa o_proj | 4096×16384 M1 | 43.7 ms | 11.5 ms | x3.79 |
| fused fp8 dsa o_proj | 4096×16384 M32 | 488.7 ms | 478.7 ms | x1.02 |

m5g mini-model greedy decode: **CPU 173 tok/s, Metal 24 tok/s** (x7.3
*slower* — ~50 µs dispatch+wrap overhead × ~40 hooked ops/token dominates
at 256-dim toy shapes; the same finding as the V4 bench and the reason
for the offload floors).

Reading: the CPU NEON-mt kernels win the small decode GEMVs (the default
32 MiB floor keeps those on the CPU); the GPU wins the big dense matmuls
and — the real prize — the fused DSA FP8 linears (no dequant
materialization, half the bytes). At the real decode profile the tier is
NVMe-bound (~7.9 GiB/token cold), so the win is architectural: dense
compute off the CPU, freeing it for expert resolve/disk I/O per
ARCHITECTURE §7. The naive one-thread-per-output shaders have the same
documented headroom as the V4 ones (vectorized loads, simdgroup
reductions, fused dispatch); numerics correctness came first.

P4 real-scale correction (the fixture numbers above are unchanged, but
they do NOT transfer to the 306 GiB container on 32 GB — measured with
persistent wraps engaged, 2 GiB tiered, 8-token steady state): every
bf16 GEMV offload shape is a net loss against the expert stream's memory
bandwidth (CPU 11.79 s/tok; floor 32768 12.20; floor 65664 11.59; floor
OFF 11.08), and prefill GEMMs lose to the shaders' missing m-blocking
(108.7 s vs 78.0 s at M=26). Shipped defaults are therefore
`APUS_GMETAL_MIN_KB` OFF, `APUS_GMETAL_FP8_MIN_KB=0` (the fused path IS
a real-scale win), `APUS_GMETAL_MAX_M=1`. See docs/locality/README.md P4
and docs/STATUS.md P4.

## V4 backend (removed)

The V4 backend (`c/backend_metal.mm`) and its test_kernels/bench_metal
legs were removed from this repo on 2026-09-05 with the V4 engine
cleanup (user decision — the repo is GLM-only). Its numerics contract,
buffer model, and measured numbers are the base project's; see git
history (the pre-removal README) or `../Apus` for the full record. The
V4 model-level story ended with the V4 fixtures (tests/m4c/m5/m6a/m6b/
m9c/m9d removed at this milestone — their GLM successors are
m4g/m4h/m5g/m6g/m7a).

## Fail-soft behavior (all covered by tests)

- No Metal device / shader compile failure → `apus_gmetal_enable`
  returns nonzero, the hooks stay NULL, the engine
  runs CPU (the CLI prints the reason). The CPU binary with
  `APUS_METAL=1` prints "not compiled in" and continues.
- Below the `APUS_GMETAL_*_MIN_KB` floors → the hook declines → CPU
  kernel, bitwise (asserted).
- Wrap failure (unmappable range) / GPU command-buffer error → the op
  returns 1 → CPU fallback for that op.
- `apus_gmetal_disable()` clears the hooks and releases the backend.

## Remaining gaps / future work

- ~~Registration-based stable-weight cache~~ — DONE in P4 (the REGISTERED
  persistence class above). Real-scale result: it removed the 4× --metal
  loss, but the bf16 GEMV offload is still bandwidth-bound against the
  expert stream, so `APUS_GMETAL_MIN_KB` defaults OFF; the fused FP8 path
  (floor 0) and the M-gate (`APUS_GMETAL_MAX_M=1`, prefill stays CPU) are
  the shipped policy.
- m-blocked / tiled GEMM shaders (threadgroup-shared W tiles) would lift
  the ×M prefill weight-traffic multiplier WITHOUT touching the
  per-output sequential-k order (bitwise-preserving by construction, but
  needs its own gate run) — that is what would unlock GPU prefill.
- KDA recurrent step + indexer pool rebuild on GPU (left on CPU for
  numerics safety this milestone — both are ordering-sensitive; each
  would need its own bitwise design like the GEMM shaders).
- Shader perf: vectorized loads / simdgroup reductions / fused per-layer
  command buffers (the V4 README's list carries over).
- MTP linears (M8+) would flow through the same hooks.
- **Real-model smoke is weight-gated** (hard gate 8): everything here is
  verified on the synthetic fixtures; the real-container run is the last
  open M7b item.
