# tests/m4h — GLM KDA (c/gkda.h) + DSA/Lightning-indexer (c/gdsa.h) hard gate

Milestone M4b (GLM): the KDA linear-attention sublayer in BOTH orderings
of the M0 KDA ORDERING CONTRACT (docs/ARCHITECTURE.md §4) and the DSA
(MLA pure-NoPE) + Lightning-Indexer sublayer, gated against the M0 numpy
oracle's OWN sublayer functions (tools/oracle.py kda_forward /
dsa_forward / indexer_forward / dsa_attention, f32-faithful mode) — the
goldens ARE the normative semantics.

Run the gate:

    make test-m4h        # golden regen + C gate, APUS_THREADS=1/4/8 diffed
    make ubsan-m4h       # UBSan build (ASan is broken on the dev Mac)

`tests/m4h/golden/` is gitignored; `make golden-m4h` regenerates it.

## The two KDA orderings (the pinned contract)

HF prefill runs KDA through the CHUNKED kernel
(`chunk_kimi_delta_attention`, glm5:483-579, chunk 64) and single-token
decode through the RECURRENT kernel (`recurrent_kimi_delta_attention`,
glm5:428-479). Same math, different fp32 summation orderings — NOT
bit-identical to each other, and there is no cross-phase bitwise
expectation. The C engine therefore carries TWO separately verified
orderings and is gated bitwise per phase:

- `apus_gkda_chunk` — chunked prefill (any s ≥ 1), including
  state-carrying continuations (chunked prefill on top of a cached state,
  the §4.4 case): cases k0 (s=68, chunk+4-tail), k1 (s=64, exact
  boundary), k2 (s=1), k3 (s=130, 2 full chunks + 2), k4 (68 then 70 on
  the carried state).
- `apus_gkda_recurrent` — the recurrent step: case k5 (68-prefill + 6
  decode steps, multi-step chain through the carried fp32 state) and k6
  (4 decode steps from a zero state).

Neither ordering may "simplify" into the other; both are pinned here.

## KDA replication details (how the C is bitwise vs numpy f32)

- **Chunked scan order**: per head, per 64-token chunk, zero
  right-padding (the pad is benign: g is cumsum'ed WITHIN each chunk, so
  padded rows inherit the last real row's cumulative decay — the C pads
  gc with ZEROS and lets the cumsum carry it; padding gc with the last
  row's raw g instead was a caught bug — it double-counts that decay in
  `g_last`). `decay[i,j,d] = expf(gc[i,d]-gc[j,d])` is computed only for
  j ≤ i — the reference computes the overflowing j > i entries and
  REPLACES them via masked_fill, so the masked entries are exactly +0.0f
  either way. The masked attn is `-sum_d (k_beta*k)*decay` (j < i); the
  UT-transform forward substitution runs rows in ascending i with each
  `(row[...,None]*sub).sum(axis=-2)` a STRIDED sum (plain sequential from
  +0.0f — see the numpy reduce-order pin below), and the `+= I` comes
  AFTER the whole solve (the diagonal is still +0.0f while later rows
  read it — adding it inside the loop was a second caught bug). Then
  value/kcd matmuls, inter = (q·exp(g))@S, intra (j ≤ i, diagonal KEPT),
  v_new = value − kcd@S, out = inter + intra@v_new, and the state update
  S·exp(g_last) + (k·exp(g_last−g))ᵀ@v_new. All matmul contractions are
  the oracle `_mm` order (+0.0f init, ascending k, mul+add two roundings,
  no FMA — `-ffp-contract=off` pinned).
- **Recurrent step order**: decay S·exp(g) → kv_mem = (S·k).sum(axis=-2)
  → delta = (v−kv_mem)·beta → S += k⊗delta → o = (S·q).sum(axis=-2).
  The axis=-2 sums are STRIDED (plain sequential from +0.0f). The rank-1
  update multiplies as k·((v−kv_mem)·beta) — the reference's delta-first
  order (the Ling donor's (beta·k)·u rounds differently).
- **Conv state handling**: the fused causal conv (k=4) keeps the last 3
  PRE-conv inputs per channel as BF16 codes, channel-major
  `state[c*3+j]`, j=0 oldest (the oracle's [k-1, C] token-major layout
  carries the same bf16 values — an engine layout choice, the test
  compares transposed). Prefill is a strict loop of the decode step's
  per-token body (bitwise == the oracle's windowed form token for token);
  the body reads the input into a register FIRST so x may alias out (an
  aliasing bug here corrupted the state with post-conv values — caught by
  the state goldens).
- **l2norm** (eps 1e-6) is applied AFTER the fp32 cast and its output
  stays FP32 (never rounded to bf16); the q-scale D^-0.5 multiplies after
  (q only). Literal division by the root.
- **Forget gate**: fg = bf16(f_b@bf16(f_a@x)), then fp32 end-to-end:
  g = −5·sigmoid_stable(expf(A_log)·(fg+dt_bias)), never rounded.
- **Beta**: bf16(sigmoid_stable(bf16(b_proj@x))) — rounded after the
  sigmoid (the donor kept it fp32).
- **o_norm** (glm5:339-359): strict fp32, eps = rms_norm_eps = **1e-5**
  (glm5:602,623 — NOT the Ling donor's 1e-6), pairwise mean variance,
  (core·rsqrt)·w·sigmoid_stable(gate), ONE bf16 rounding.
- All sigmoids/silus use numpy's numerically-stable form
  (apus_gmhc_sigmoid: e = expf(−|x|), branch on sign) — the naive
  1/(1+expf(−x)) rounds differently for x < 0.

## DSA + Lightning indexer replication details

- **MLA pure-NoPE** (no RoPE anywhere): q_resid = weighted RMSNorm(fp8
  q_a), q = fp8 q_b, k_pass = weighted RMSNorm(fp8 kv_a), k|v = bf16 kv_b
  split, attention, out = fp8 o_proj. All FP8 linears are m3g
  dequant+bf16 GEMM (bitwise == oracle `_mm`). The weighted RMSNorm: fp32
  internal (numpy pairwise mean), bf16 round, THEN ×weight, round again.
  The expanded kv cache stores BF16 codes, head-major `[H][cap·dim]`
  (engine layout; the oracle's [pos][H][dim] carries the same values).
- **Indexer pool rebuild**: the k/gate caches are appended with this
  call's tokens and ALL k-pools are REBUILT from the FULL cache every
  forward, decode included (glm5:811). Only FULL pools (all kpool=4
  tokens valid) are scored; pool key = softmax(gate+ape) weighted average
  of member keys with the probs CAST TO BF16 before the average and the
  average rounded again (glm5:965-968). The kpool-axis sums are STRIDED
  (plain sequential from +0.0f). Scores are fp32 `_mm` dots (NO bf16
  rounding), relu AFTER the ×head_dim^-0.5 scale, weights ×n_heads^-0.5,
  head-sum from +0.0f.
- **Candidate validity**: the pool's LAST token must be causally visible;
  masked scores are finfo.min. top-(topk/kpool) pools, stable descending,
  ties to the LOWER pool index (the M0 pin); invalid selections → −1;
  each selected pool expands to its 4 raw token indices. The CURRENT
  INCOMPLETE pool is always appended as up to 3 raw tail indices
  IMMEDIATELY after the selected pools (tail_count = (q_pos+1) % kpool,
  tail_count = 0 → all −1), and the row is −1-padded to width
  index_topk+kpool−1 AFTER the tail (placing the tail at the fixed end of
  the row instead was a caught bug — it only coincides with the reference
  layout when select_k == topk/kpool).
- **Sparse attention**: over the FULL cached kv with the additive-mask
  semantics (vis boolean dedup, everything not selected gets finfo.min —
  NOT a gather): bf16(q@kᵀ) round, ×scale round again, + mask, fp32
  softmax (contiguous pairwise sum) → bf16, P·V ascending kv positions
  from +0.0f → bf16.

## numpy reduce-order pin (verified bitwise vs numpy 2.5.2)

numpy's summation order depends on the reduction AXIS LAYOUT:

- **Contiguous last-axis sums** (`.sum(-1)`, means, softmax/l2norm/RMSNorm
  denominators): numpy pairwise summation — replicated by
  `apus_gmhc_pw_sum` (the m4g replica).
- **Strided-axis sums** (`.sum(axis=-2)` of a row-major array, middle-axis
  sums like the kpool axis): PLAIN SEQUENTIAL from +0.0f. Verified across
  n = 4…1536 and shapes [n,128], [17,4,128], [2,128,128], [2,2,i,i].
- Cumsum: sequential. Matmul-style contractions: the oracle `_mm` order.

## Gate tiers (the host-transcendental class)

Same two-tier design as m4g. The only non-IEEE-exact op is `exp`
(conv/forget/beta/o_norm sigmoids, KDA decays, indexer pool softmax,
attention softmax). The C test probes 1024 stored np.exp values at
startup (macOS arm64: 0 diffs; Linux x86_64: 0 diffs with numpy pinned to
its baseline exp kernel via the Makefile's NPY_DISABLE_CPU_FEATURES
recipe):

- **BITWISE tier** (probe clean): EVERY golden memcmp-compared — f32
  goldens, bf16 outputs (widened: equal iff the codes are equal), the
  indexer top-k selections, and both states (KDA conv/rec state, DSA
  caches). Engaged on the dev Mac AND on Linux (via the pin).
- **TOLERANCE tier**: f32 goldens rel ≤ 1e-5 or abs ≤ 1e-6; bf16-code
  goldens rel ≤ 1e-2 or abs ≤ 1e-6 (the ±1-code boundary-flip class: a
  1-ulp host-exp difference can flip a bf16 rounding); indexer pool
  scores rel ≤ 1e-5 or abs ≤ 5e-3 (the pool-prob bf16-flip class —
  measured: a 1-ulp exp perturbation essentially never moves a pool score
  by more than ~1e-3 on these fixture magnitudes); indexer selections
  STILL bitwise EXCEPT the manifest's fragile queries (below).

Either way the full output (incl. the FNV-1a digest of the C outputs) is
diffed across APUS_THREADS=1/4/8: the gemms run on the
bitwise-at-any-thread-count m3g mt kernels and everything else is
single-threaded, so the digest is thread-count invariant. It is also
identical between the -O2 and UBSan -O1 builds on the same host
(e1bc2a34dc855c6d on the dev Mac). Golden BYTES are host-pinned (same
caveat as m0/m4g f32 goldens); each host regenerates its own fixtures.

## Fragile queries (indexer near-ties)

Indexer score ties are STRUCTURAL with random weights — no redraw can
escape them: relu-clipped pools tie at exactly 0.0, and ~25 % of queries
have all-negative scores (the per-head weights w are signed projections),
so the zero block tops the ranking. (Masked pools tie at exactly
finfo.min, which is deterministic and needs no margin.) The generator
records per-call FRAGILE query lists in the manifest: a query is fragile
when any relevant consecutive gap in its sorted masked pool scores (the
pairs deciding the emitted pool set/order) is below SEL_MARGIN = 0.02
(~20× the measured perturbation class above). The BITWISE tier compares
everything (exact ties are deterministic: stable descending, lower index
first, in BOTH numpy's argsort and the C top-k — m4g's
apus_gmoe_topk_stable). The TOLERANCE tier skips fragile queries'
selections/scores/attention rows and compares the rest. Case d4 pins the
tie-break itself: every token row identical → bitwise-identical pool
scores (zero flip risk) → the golden asserts the lower-index-first order.

## What is tested (12 cases, 306 checks, 0 failures)

- **k0–k6** (KDA): chunked prefill s=68/64/1/130, continuation 68+70,
  prefill+6-step decode chain, 4-step decode from zero. Goldens per call:
  post-conv mixed, g, beta, core, gate, onorm, out (all bitwise); final
  conv_state + rec_state (bitwise).
- **d0** (DSA): prefill s=68 + 4 decode steps — 17→18 pools, pool rebuild
  every step, tail_count cycling 0–3, selection churn.
- **d1**: prefill s=3 — kv_len < pool width (n_full = 0: no pools scored,
  tail-only rows, −1 padding).
- **d2**: prefill s=4 + 5 decode steps — tail_count = 0 at n ≡ 0 (mod 4),
  cache crossing pool boundaries.
- **d3**: prefill s=8 — select_k == n_full (all pools selected).
- **d4**: crafted EXACT tie (identical token rows) — the stable
  descending / lower-index-first tie-break.
- Per call: q_resid, q, k_new/v_new, idx k/gate rows, pool scores, topk
  selections (i32, always bitwise outside fragile skips), attention
  probs, attention out, sublayer out; final caches (k/v/idx k/idx gate).

UBSan clean; `leaks` clean (0). Scalar + m3g-kernel dispatch only (SIMD
for the sublayer bodies is M7 perf work; it must preserve these orders
bitwise).

## Donor divergences (../Apus-Ling-3.0-Flash-bf16 c/kda.h vs the GLM oracle)

The seam bugs Phase A warned about — each is bitwise-visible and fixed in
c/gkda.h (the donor stays untouched):

1. l2norm output rounded to bf16 (oracle: fp32, never rounded), and its
   sum-of-squares sequential (oracle: numpy contiguous pairwise).
2. Naive sigmoid 1/(1+expf(−x)) everywhere (oracle: numpy-stable form;
   differs for x < 0) — conv silu, forget gate, beta, o_norm.
3. o_norm eps 1e-6 (oracle: rms_norm_eps = 1e-5, glm5:602,623) and a
   sequential variance sum (oracle: pairwise mean).
4. Beta kept fp32 (oracle: bf16-rounded after the sigmoid).
5. Rank-1 update as (beta·k)·u (oracle: k·((v−kv_mem)·beta)).
6. No chunked prefill at all (GLM pins the chunked path for prefill).
7. (Non-numeric) conv state layout channel-major vs the oracle's
   token-major — an engine layout choice, values identical.

vs the inherited V4 c/attn.h (for the record — fresh port, not a tweak):
GLM has NO Hadamard rotation, NO FP4/score quantization, NO RoPE, NO
attention sink; the indexer rebuilds kpool-4 pools from the full cache
every forward with an always-selected tail (width topk+kpool−1).

## Files

- `c/gkda.h` — GLM KDA (`APUS_GKDA_IMPLEMENTATION`): conv
  (prefill/decode), chunked core, recurrent core, forget gate, beta,
  o_norm, composed `apus_gkda_forward`. Needs c/bf16.h + c/gmhc.h
  implementations linked.
- `c/gdsa.h` — GLM DSA (`APUS_GDSA_IMPLEMENTATION`): weighted RMSNorm,
  LayerNorm, Lightning indexer (pool rebuild, scores, top-k, tail),
  sparse attention, composed `apus_gdsa_forward` (FP8 via c/fp8blk.h).
  Needs c/bf16.h + c/fp8blk.h + c/gmhc.h + c/gmoe.h implementations.
- `tests/m4h/gen_golden.py` — oracle-driven fixture generator (imports
  tools/oracle.py; fragile-query lists; exp probe).
- `tests/m4h/test_m4h.c` — the gate (probe → tier, per-case bitwise /
  tolerance asserts, thread-invariance digest).
- `tests/m4h/golden/` — generated fixtures (gitignored).
