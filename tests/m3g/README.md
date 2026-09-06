# tests/m3g — GLM kernel hard gate (c/fp8blk.h + c/bf16.h)

Milestone M3 (GLM-5.3-Flash adapter): the kernel layer behind the
normative compute path — FP8-E4M3 + F32 `weight_scale_inv` 128×128-block
dequant → BF16 (`c/fp8blk.h`), and BF16 GEMV/GEMM with FP32 sequential
accumulation (`c/bf16.h`). Scalar anchor first; NEON and AVX2 paths are
**bitwise identical** to it; the threaded (mt) path is bitwise at every
`APUS_THREADS`. (Directory name: the inherited V4 fp4/fp8 kernel battery
already owns `tests/m3`.)

Run the gate:

    make test-m3g        # goldens + bitwise battery + APUS_THREADS=1/4/8 diff
    make ubsan-m3g       # UBSan variant (ASan is broken on the dev Mac)

## Numerics contract (what is pinned, and where it comes from)

The semantics are the M0 oracle's (`tools/oracle.py`, pinned in
`tests/m0/README.md`); `gen_golden.py` builds the fixtures with the
oracle's own codec (`E4M3_TABLE`, `bf16_round`, `_mm`), so goldens ARE the
normative semantics, not a re-derivation:

- **Dequant** (oracle `fp8_dequant`, f32 mode + `bf16_round`):
  `out[o,k] = bf16_rne(E4M3(w[o,k]) * ws[o/128, k/128])` — ONE IEEE fp32
  multiply per element, then round-to-nearest-even to BF16. Scales are
  plain F32 (NOT the V4/base UE8M0 pow2 format). `O`/`K` need not be
  multiples of 128 — edge blocks are partial, and the scale grid is
  ceil-shaped (the real checkpoint ships such shapes, tests/m1).
- **BF16 GEMV/GEMM** (oracle `_mm`, f32 mode + `bf16_round`):
  `y[m,o] = bf16_rne(Σ_k f32(x[m,k]) * f32(w[o,k]))` — fp32 accumulator,
  strictly ascending k, mul+add as two separate IEEE roundings, NO FMA
  (`-ffp-contract=off` pinned; the UBSan target re-adds the flag
  explicitly because sanitizer flags override CFLAGS). The oracle's `_mm`
  uses exactly this order, so the C scalar anchor is **bitwise** equal to
  the oracle f32-mode matmul — the gate is bitwise, not a tolerance
  class. There are NO reorder classes in this battery.
- **Composition** (oracle `fp8_linear`): dequant → BF16 → BF16 GEMM —
  the FP8 dense linear the M4 sublayers will call.

## What is tested (test_m3g.c)

1. **BF16 widen/narrow exhaustive** — all 65,536 codes: widen is exactly
   `code << 16`; `narrow(widen(code)) == code` (NaN payloads pass through
   as the high 16 bits).
2. **E4M3 decode exhaustive** — all 256 codes bitwise vs the `ldexpf`
   formula. NaN codes 0x7F/0xFF decode as ±480 (e=15, m=7 as a normal) —
   documented out-of-contract behavior: the converter refuses them
   (tests/m1), and the SIMD expansions decode them identically, so all
   paths stay bitwise even then.
3. **Dequant exhaustive** — all 256 codes × scale corners (1, 0.5, −2, 0,
   2⁻¹²⁶, 2⁻¹²⁷ subnormal, 2⁴⁰, 448, 2⁻⁹, 3.14e-30): scalar vs NEON vs
   dispatch vs mt, bitwise.
4. **Golden dequant** — 9 oracle cases, shapes (128,128), (256,384),
   (200,136), (1,128), (384,128), (160,200), (128,127), (300,264),
   (5,129) — non-multiples of the 128 grid on both dims; scalar / NEON /
   dispatch / mt bitwise vs the oracle bytes + FNV-1a digests.
5. **Golden GEMV/GEMM** — 13 oracle cases sweeping the SIMD tails
   (K % 32 ≠ 0 chunk tails, O % 8 / O % 4 row-group tails, M % 4 groups):
   (1,1,1)…(7,200,160), (2,33,31), (1,256,128); scalar / NEON / AVX2 /
   mt bitwise vs the oracle; M-independence (GEMM row ≡ GEMV).
6. **Golden fp8_linear composition** — 4 cases (incl. (5,200,264),
   (3,160,200) ceil shapes): `apus_fp8blk_dequant` → `apus_bf16_gemm_mt`
   bitwise vs the oracle's `fp8_linear` f32 mode.
7. **In-test shape sweep** — 18 random shapes, scalar == NEON == AVX2 ==
   mt bitwise.
8. **Thread-count independence** — `make test-m3g` diffs full output
   (incl. the FNV-1a mt output digest) across APUS_THREADS=1/4/8.

Result on the dev machine (M1 Pro, clang): **197 checks, 0 failures**;
digests identical at APUS_THREADS=1/4/8; UBSan clean; `leaks` clean (0).
(188 pre-P5; +9 = the dequant_mt golden legs.)
The AVX2-vs-scalar bitwise gates run on x86-64 hosts (CI linux + windows
jobs; off-x86 the AVX2 sections compile out and the mt path falls back to
the scalar anchor).

## Reorder classes

None. Every path in `c/bf16.h` / `c/fp8blk.h` reproduces the scalar
anchor's rounding sequence exactly (staged single-rounded products,
strictly ascending adds; ILP only from interleaving independent output
chains). The donor repo's M9b ILP-reorder kernels and BLAS dispatch were
deliberately NOT ported — they are perf classes, to be reconsidered at
the perf milestones with their own gates.

P5 added `apus_fp8blk_dequant_mt` (c/fp8blk.h): dequant is ELEMENTWISE
(one independent multiply + RNE narrow per output element — there is no
accumulation order to reorder), so partitioning output rows over the
c/pool.h lanes is bitwise at every APUS_THREADS by construction, NOT a
reorder class. The plain `apus_fp8blk_dequant` dispatch stays
single-threaded: the gcache I/O workers call it from their own threads
and the pool is one-job-at-a-time. Compute-thread call sites
(c/gdsa.h's per-token DSA FP8 projections, gmodel open-time dense
dequant) use the mt entry. The mt leg is gated above (exhaustive +
golden, T=1/4/8 diffed).

## Performance bench (informational, NOT a gate)

    make bench-m3g    # tests/m3g/bin/bench_bf16

P5 decode-GEMV effective-bandwidth bench at the real GLM decode shapes:
scalar / single-thread SIMD / mt GB/s per shape plus two rooflines
(`wadd8` = the numerics-preserving ceiling of any bitwise-safe kernel —
widen + strictly-sequential adds, 8 row chains, no multiply; `read` =
pure streaming-read ceiling), and the DSA FP8-linear legs (deq1 vs
deq_mt vs the mt GEMV). `APUS_THREADS` sets the pool lanes,
`APUS_BF16_BENCH_REPS` overrides the rep counts. P5 baseline (M1 Pro,
APUS_THREADS=8, measured under background load): mt GEMV 24-64 GB/s ≈
the per-process read roofline — the decode GEMVs are memory-bound, not
kernel-bound; deq_mt is 5-7x the single-thread dequant.

## Ambiguities pinned at M3

- **E4M3 NaN codes (0x7F/0xFF)**: the oracle's numpy table maps them to
  NaN; the C scalar codec (`apus_e4m3_dequant_f32`, inherited from the
  base) evaluates them as ±480. Pinned: keep the C behavior (converter
  refuses these codes; they never occur in lab checkpoints), SIMD paths
  match it bitwise, and this README records the divergence.
- **bf16_round NaN inputs**: the C helper passes NaN through unchanged;
  the oracle's bit-trick has no special case (can mangle NaN payloads).
  Identical for every non-NaN input; fixtures are finite, and NaN inputs
  never occur in the normative path (inf/NaN propagation inside the
  GEMM — 0×inf, overflow — is IEEE-deterministic and identical across
  paths by construction).
- **Ceil-shape dequant**: the oracle's `fp8_dequant` (`np.repeat`) only
  covers 128-multiples; `gen_golden.py` generalizes by direct block
  indexing (`scales[o//128, k//128]`) — same one-multiply-per-element
  semantics, matching the checkpoint's ceil-shaped scale tensors
  (tests/m1 accepts them).
- **M0 fixture naming note**: the m0 fixtures keep FP8 matrices at
  128-multiples; the m3g fixtures deliberately cover the ceil shapes the
  real checkpoint can ship.

## Files

- `gen_golden.py` — fixture generator (uses `tools/oracle.py`'s codec)
- `test_m3g.c` — the C gate (this battery)
- `bench_bf16.c` — P5 decode-GEMV/FP8-dequant bandwidth bench (`make
  bench-m3g`; informational, see the bench section above)
- `golden/` — generated fixtures (gitignored; regenerate via `make golden-m3g`)
