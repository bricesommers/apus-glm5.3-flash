# tests/m4g — GLM mHC (c/gmhc.h) + GLM MoE (c/gmoe.h) hard gate

Milestone M4a (GLM): the mHC residual stream (per-token pre/post/comb map
generation with the norm-before-fn unweighted RMSNorm, Sinkhorn-20,
collapse/expand, the unweighted-mean HyperHead) and the MoE sublayer
(sigmoid router with selection-only noaux_tc bias, FP8-E4M3 swiglu
experts with the swiglu_limit=10 clamps, shared expert, bf16-stepped
accumulation), gated against the M0 numpy oracle's OWN sublayer functions
(tools/oracle.py hc_pre/hc_post/router_forward/moe_forward/swiglu_mlp,
f32-faithful mode) — the goldens ARE the normative semantics.

Run the gate:

    make test-m4g        # golden regen + C gate, APUS_THREADS=1/4/8 diffed
    make ubsan-m4g       # UBSan build (ASan is broken on the dev Mac)

`tests/m4g/golden/` is gitignored; `make golden-m4g` regenerates it.

## GLM-vs-base deltas pinned here (the reason these are new headers)

`c/mhc.h` (V4) and `c/moe.h` (V4) stay untouched for the inherited engine.
The GLM differences — all bitwise-sensitive — live in c/gmhc.h / c/gmoe.h:

- **Norm-before-fn**: GLM applies the UNWEIGHTED RMSNorm (eps =
  rms_norm_eps = 1e-5, fp32, no weight, no rounding) to the flattened
  streams BEFORE the fn matmul (glm5:278-279); V4 multiplied by rsqrt
  AFTER it (norm_eps 1e-6). Mathematically equal, rounding different.
- **HyperHead is an unweighted MEAN** over the stream axis (glm5:1494),
  one bf16 rounding — no fn/scale/base/sigmoid/Sinkhorn (unlike V4's
  hc_head).
- **Router**: sigmoid scores (not sqrtsoftplus), fp32 logits (no
  rounding), `e_score_correction_bias` added for top-k SELECTION only,
  weights gathered from the UNBIASED scores, `w/(Σ+1e-20)`
  (norm_topk_prob), ×2.5, group-limiting a literal no-op at n_group=1.
- **Expert accumulation**: per-expert contribution bf16-rounded,
  accumulated in ASCENDING EXPERT INDEX order with ONE bf16 rounding PER
  ADD; shared expert added LAST (one more rounding). V4 accumulated fp32
  with a single final rounding.
- **SwiGLU clamps** (glm5:102-103, swiglu_limit=10), exact order:
  `g = min(g, 10)` (gate clamped ABOVE only), `u = clip(u, -10, 10)` (up
  clamped BOTH sides) — applied to the bf16 GEMM outputs BEFORE the
  silu; then `h = bf16(g·sigmoid(g))`, `h = bf16(h·u)` (TWO bf16
  roundings; the oracle's `_B(u)` is a no-op since u is bf16-valued and
  ±10 is bf16-exact).

## Numerics replication strategy (how the C is bitwise vs numpy f32)

Everything except `exp` is IEEE-exact given the right ORDER, and the C
replicates the oracle's orders exactly:

- **Dots/matmuls**: the oracle's `_mm` convention — acc = +0.0f,
  strictly ascending k, mul+add two roundings, no FMA
  (-ffp-contract=off). Router logits (fp32, no rounding), the mHC fn
  matmul, and the expand `comb^T@residual` matmul. The expert GEMMs are
  m3g's `apus_fp8blk_dequant` + `apus_bf16_gemv_mt` — already gated
  bitwise == `_mm`, at every APUS_THREADS.
- **Axis sums**: numpy's pairwise summation order is replicated
  (`apus_gmhc_pw_sum`: n < 8 sequential from a[0]; n ≤ 128 the
  8-accumulator tree + sequential remainder; larger n recursive halving
  to multiples of 8 — verified bitwise against numpy 2.5.2 at every
  fixture size, 1–2048, plus the real-model 16384). Used for the
  RMSNorm mean (n = 1024/512), the sinkhorn sums (n = 4), and the
  router weight normalization (topk = 3 sequential, topk = 8 TREE —
  rtr3/moe1 exercise the tree). Note the init asymmetry: numpy axis
  sums start from a[0], `_mm` starts from +0.0f (matters only for
  signed zeros; replicated deliberately).
- **Sigmoid**: numpy's numerically-stable form (oracle.py sigmoid):
  `e = expf(-|x|); x ≥ 0 ? 1/(1+e) : e/(1+e)` — NOT the naive
  1/(1+expf(-x)) for x < 0 (different rounding).
- **bf16 roundings** at exactly the oracle's `_B` points (collapse out,
  expand post/comb casts + t1 + matmul + final add, head mean, every
  expert linear out, silu h and h·u, per-expert contribution, per-add
  accumulation, shared add).
- **exp**: `expf` vs numpy float32 exp is a HOST property (see below).

## Gate tiers (the host-transcendental class)

The only non-IEEE-exact op is `exp` (sigmoid, sinkhorn softmax, silu).
numpy ≥ 2.x computes float32 exp with SIMD kernels (X86_V3 on x86-64);
the C side calls libm `expf`. Measured (500k-sample probes):

- **macOS arm64**: numpy's f32 exp ≡ Apple libm expf BITWISE (0/500007
  mismatches). No pinning needed.
- **Linux x86_64**: numpy's X86_V3 exp differs (~40 % of values, ±1
  ulp), but pinning `NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX
  AVX512_SPR"` drops numpy to its baseline kernel (scalar libm calls),
  which IS bitwise identical to gcc expf (0/500000). The Makefile's
  `golden-m4g` recipe sets this pin; it is a no-op on arm64.
- **Windows/MinGW**: not probed here; if the runtime probe fails, the
  gate still runs in the tolerance tier.

The C test therefore probes `expf` vs the stored `np.exp` values at
startup (probe_x/probe_y, 1024 points covering [-90, 10]):

- **BITWISE tier** (probe clean): EVERY golden compared memcmp-bitwise
  — f32 intermediates (mixes, pre/post/comb, router scores/biased,
  weights), the bf16 outputs (compared widened: equal iff the codes are
  equal), and the router selections. Engaged on the dev Mac AND on
  Linux (via the pin above; verified in tools/docker).
- **TOLERANCE tier** (probe differs): f32 goldens rel err ≤ 1e-5 (the
  documented host-transcendental class — goldens store the PRE-ROUND
  f32 values, so no bf16 code-boundary mush), router selections STILL
  bitwise (margin-protected, below).

Either way the full output (incl. the FNV-1a digest of the C outputs)
is diffed across APUS_THREADS=1/4/8: the expert GEMVs run on the
bitwise-at-any-thread-count m3g mt kernels and everything else is
single-threaded, so the digest is thread-count invariant. It is also
identical between the -O2 and UBSan -O1 builds on the same host
(7f24a242b80ea050 on the dev Mac). Golden BYTES are host-pinned (expf
varies across libms — the same caveat as m0's f32 goldens), so the
digest differs between macOS and Linux; each host regenerates its own
fixtures.

## Near-tie robustness (the M0 ~2 % flip warning)

The M0 oracle documented that with random weights, router top-k
selections flip ~2 % of entries between f32 and f64 evaluation. That
cascade is a PRECISION-CLASS difference (f32 vs f64); this gate compares
C-f32 against oracle-f32 with REPLICATED orders, so on the bitwise tier
the selections are identical by construction. Residual risk lives in the
tolerance tier (host exp differences, ±few ulp on scores). Policy:

- Random fixtures are RE-DRAWN at generation until every token's
  biased-score gap across the top-k boundary exceeds **1e-4** (recorded
  as `*_margin` in the manifest; observed 1.5e-4 – 2.3e-2). A ±few-ulp
  (~1e-7) score perturbation cannot flip a 1e-4 gap, so the
  always-bitwise selection check is robust in both tiers.
- The tie-break itself (stable descending, LOWER index first — the M0
  pin, `np.argsort(-row, kind="stable")`) is pinned by `rtr1`, a
  CRAFTED EXACT TIE (bitwise-identical gate rows + equal bias →
  identical biased scores, zero flip risk): the golden asserts
  idx = [2, 5, …]. `rtr2` pins the selection-only bias (a zero-bias
  ranking differs from the biased one; the boosted expert's weight is
  its UNBIASED score).

## What is tested (17 cases, 204 checks, 0 failures)

- **mhc0–3** (maps + collapse): s=6 d=256 random; edge case (zero /
  ×100 / ×1e-3 states, ±30 sigmoid-saturating bases); d=128 second
  shape; s=1 decode shape. Goldens: mixes, pre, post, comb (all fp32),
  collapse out (bf16).
- **exp0–1** (hc_post expand): random post/comb, s=6 d=256 and s=1
  d=128; post/comb bf16-cast order, comb^T indexing, the three bf16
  roundings.
- **head0–1** (HyperHead mean): s=6 d=256, s=1 d=128.
- **rtr0–3** (router): random E=8 topk=3 (margin 1.5e-4); crafted exact
  tie; crafted selection-only bias; E=16 topk=8 (pairwise-tree weight
  sum).
- **xp0–2** (expert/dense MLP swiglu): random; SATURATING (x_std=8 —
  61 gate-above clamps, 55 up-above, 46 up-below); dense-MLP shape
  inter=256. Goldens: gate/up GEMM outs (pre-clamp), h, out.
- **moe0–1** (full forward): tiny-config E=8 topk=3 s=6; E=16 topk=8
  s=4. Goldens: scores, biased, idx, weights, routed, shared, out.

UBSan clean; `leaks` clean (0). Scalar + m3g-kernel dispatch only — the
mHC maps/Sinkhorn are scalar (SIMD is M7 perf work; it must preserve
these orders bitwise).

## Files

- `c/gmhc.h` — GLM mHC (`APUS_GMHC_IMPLEMENTATION`): sigmoid,
  pairwise-sum replica, unweighted RMSNorm, Sinkhorn-20 (step-exposed),
  maps, collapse, expand, head. Self-contained (c/num.h).
- `c/gmoe.h` — GLM MoE (`APUS_GMOE_IMPLEMENTATION`): router
  (`apus_gmoe_router`, `apus_gmoe_topk_stable`), `apus_gmoe_expert`
  (shared by routed/shared/dense), `apus_gmoe_forward`. Needs c/bf16.h
  + c/gmhc.h implementations linked.
- `tests/m4g/gen_golden.py` — oracle-driven fixture generator (imports
  tools/oracle.py; margin re-draws; exp probe).
- `tests/m4g/test_m4g.c` — the gate (probe → tier, per-case bitwise /
  tolerance asserts, thread-invariance digest).
- `tests/m4g/golden/` — generated fixtures (gitignored).
