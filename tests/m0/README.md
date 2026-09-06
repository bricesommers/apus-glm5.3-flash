# tests/m0 — oracle (tools/oracle.py) hard gate

Milestone M0 (oracle half): a numpy-only port of the HuggingFace reference
implementation of **glm5_next** — `reference/inference/modeling_glm5_next.py`
(auto-generated from `modular_glm5_next.py`), the ONLY normative source.
Text model only (vision tower excluded); 45 main layers; the MTP layer
(layers.45, classic NextN) is DEFERRED to M8+ — `model_forward` returns the
final mHC stream `h` next to the logits so the MTP glue can be added later.

Run the gate:

    .venv/bin/python tests/m0/check_oracle.py

It regenerates `tests/m0/fixtures/` (gitignored) via
`.venv/bin/python tools/oracle.py` (CLI: `--out`, `--no-clean`, `--seed`)
and asserts the digests bitwise. Runtime ~8 s.

## Fixture config (tiny, synthetic)

`oracle.make_tiny_config()`: real glm5_next values with small dims —
hidden 256, 5 layers (KDA 0,1,2,4; DSA 3; dense MLP 0–2; MoE 3,4),
heads 2×128 everywhere, q_lora/kv_lora 128, index_topk 8 (select_k = 2
pools + 3-wide tail), 8 routed experts top-3 + 1 shared, hc_mult 4,
Sinkhorn-20, vocab 384. **Every FP8 matrix stays a multiple of 128 in both
dims** — the real 128×128 block grid is NOT shrunk.

Layer-pattern override rule (used whenever `num_hidden_layers` or
`first_k_dense_replace` is overridden): DSA at layers `i % 4 == 3`, KDA
elsewhere; dense below `first_k_dense_replace`, sparse after. Verified to
reproduce the real config's explicit 45-layer lists EXACTLY.

Fixture weight naming mirrors the real checkpoint
(`reference/model.safetensors.index.json`) minus the
`model.language_model.` prefix: `layers.3.self_attn.q_a_proj.weight` +
`.weight_scale_inv` (F32), `layers.0.self_attn.q_conv1d.weight` (BF16),
`layers.3.mlp.experts.{e}.{gate,up,down}_proj.*` (FP8),
`layers.{i}.hc_{attn,ffn}_{fn BF16, base F32, scale F32}`, top-level
`embed_tokens.weight` / `norm.weight` / `lm_head.weight` (BF16, untied).
KDA smalls (q/k/v_proj, conv1d, f/g low-rank pairs, b_proj, o_proj,
o_norm), `kv_b_proj`, the whole indexer, and the router gate are BF16;
`A_log`/`dt_bias`/`e_score_correction_bias` are F32 — exactly as in the
checkpoint anatomy (docs/STATUS.md Phase A).

## Precision semantics (normative — the C side must replicate these)

Two modes, inherited from the base oracle: **f64** (all arithmetic in
float64, the numerical truth) and **f32** (dtype-faithful: bf16 rounding
at every point the reference casts to bf16, fp32 where the reference is
fp32). FP8 weight storage is ALGORITHMIC (applied in both modes):

- **FP8 path**: `W = E4M3(code) * weight_scale_inv` per 128×128 block
  (**plain F32 scales**, one F32 multiply — NOT the base's UE8M0 pow2
  path) → round to BF16 → BF16 matmul with FP32 accumulate → BF16 out.
- **bf16_round**: round-to-nearest-even via the f32 bit trick
  (`(u >> 16 & 1) + 0x7FFF`, mask low 16 bits). Identical to the base's
  `apus_bf16_round`. All "bf16 matmuls" accumulate fp32 in the
  deterministic sequential-`_mm` order (M12b convention: the tolerance
  budget vs the C kernels covers only summation-ORDER differences).
- **Weighted RMSNorm** (glm5:65-83): fp32 internal, cast to bf16, THEN
  multiply by the weight — TWO bf16 roundings, in that order.
- **mHC**: norm-before-fn is the UNWEIGHTED RMSNorm (eps = rms_norm_eps =
  1e-5) applied to the flattened streams in fp32 BEFORE the fn matmul
  (glm5:278-279) — the base engine applied rsqrt AFTER; the order is
  bitwise-sensitive. Maps (pre/post/comb) are fp32 end-to-end. The expand
  (glm5:1317-1319) casts post AND comb to bf16 first, `comb^T @ residual`
  is a bf16 matmul, and the final add rounds to bf16. The collapse is an
  fp32 weighted sum rounded once to bf16.
- **Sinkhorn-20** (glm5:286-290): row softmax + per-element eps, col-norm
  with eps in the denominator, then 19 × (row-norm, col-norm) — the same
  recipe as the base `c/mhc.h` (column-stochastic to ~1e-6, row sums
  deviate; do not "fix").
- **hc_head** (glm5:298-302): unweighted MEAN over the stream axis — no
  fn/scale/base, no Sinkhorn (unlike the base).
- **MoE** (glm5:120-207): router in fp32 (sigmoid scores, bias for
  SELECTION only, weights from UNBIASED scores, `w/(sum+1e-20)`, ×2.5;
  group-limiting a literal no-op at n_group=1). Eager experts loop:
  per-expert contribution rounded to bf16, accumulated in ASCENDING EXPERT
  INDEX order with a bf16 rounding per add; shared expert added last (one
  more rounding). SwiGLU clamps: gate ABOVE only, up both sides, limit 10.
- **DSA attention** (glm5:1040-1062, 1219-1257): full-kv additive mask
  (finfo.min), bf16 matmul → bf16, × scale rounds, fp32 softmax → bf16,
  bf16 P·V. NO attention sink (unlike the base). Pure NoPE — no RoPE
  anywhere in the text model.
- **Indexer** (glm5:774-1025): q/k/gate bf16; pool softmax logits fp32,
  probs cast to BF16 before the weighted average (glm5:965-968); scores
  are fp32 matmuls end-to-end (explicit `.float()`), relu AFTER the
  `head_dim^-0.5` scale; weights × `n_heads^-0.5`.
- **KDA conv** (glm5:375-414): conv1d is a kept-fp32 module (checkpoint
  stores BF16, cast at load) — conv + silu computed in fp32, ONE bf16
  rounding at the output; the conv state holds bf16 values.
- **KDA forget gate** (glm5:305-335): safe-gate path only
  (`gate_lower_bound = -5.0`): `g = -5 * sigmoid(exp(A_log) *
  (bf16(f_b(f_a(x))) + dt_bias))`, fp32, NEVER bf16-rounded. The softplus
  branch is dead for this config.
- **KDA core** (chunked AND recurrent): fp32 throughout (f64 in truth
  mode); l2norm (eps 1e-6) applied AFTER the fp32 cast, BEFORE the
  `head_dim^-0.5` query scale. Recurrent state is fp32.
- **Logits**: untied BF16 lm_head, bf16 matmul → bf16-valued logits.

## The external anchor: oracle vs TRUE HuggingFace glm5_next (2026-09-05)

`tests/m0/check_vs_hf.py` (`make check-hf`, OPT-IN — needs torch +
transformers @ git main in `.venv`: `.venv/bin/pip install torch
"git+https://github.com/huggingface/transformers.git"`) runs the oracle
against the actual HF implementation (torch CPU, bf16) on a synthetic
tiny checkpoint (the m0 fixture config + generator, FP8 pairs dequantized
to BF16 with the oracle's own dequant — FP8 storage fidelity is gated
bitwise in tests/m3g; what this anchors is the COMPUTE GRAPH). Criterion:
rel logit error <= 2e-2 + argmax agreement with the margin-excused tie
class (measured cross-engine bf16 evaluation-order noise <= ~0.06 abs).
Known fragile class: indexer selection-boundary near-ties (the m4h
fragile class, cross-engine) — the gate REDRAWS the fixture seed below a
0.05 selection margin (the m4g/m5g policy). `--dump` writes both sides'
per-layer streams for a sublayer bisect on divergence.

**It caught a real one (gate 7)**: the oracle's kv_b_proj split was
head-major (`k_new = kvb[:, :H*qd]`, `v_new = kvb[:, H*qd:]`); the HF
reference (glm5:1147, `expand_kv`) splits PER HEAD interleaved
(`[..., :qd]` / `[..., qd:]`). Divergence evidence: rel 0.74 + argmax
6/16 pre-fix; post-fix rel ~0.012-0.020 with 16/16 argmax across four
seed families. The oracle fix (tools/oracle.py `dsa_forward`) landed
2026-09-05; the C engine's `c/gdsa.h` shares the layout and its gates
(m4h/m5g/m6g/m7a/m7b/m8g) go red until the C side follows (signature:
k_new/v_new diverge at every DSA case, KDA cases clean).

## KDA ordering contract (user-pinned at M0)

HF prefill runs the CHUNKED kernel (`chunk_kimi_delta_attention`,
glm5:483-579, chunk 64); single-token decode runs the RECURRENT kernel
(glm5:428-479). Same math, different fp32 orderings — NOT bit-identical to
each other. The oracle exposes them separately: `prefill()` = chunked
always, `decode_step()` = recurrent always. **The C engine is gated
bitwise per phase** (prefill vs chunked goldens, decode vs recurrent
goldens); cross-phase agreement is only mathematical.

Measured on this fixture (manifest `ordering_contract`): one-token
extension, f64 max |Δlogit| = **4.55e-15**, f32 max |Δ| = **0** (bitwise —
at this size the bf16 output quantization absorbs the fp32 ordering
difference; longer sequences will diverge in f32, which is exactly why the
per-phase gating exists). Unit check 4b runs a 68-token (2-chunk) KDA
sequence through both paths in f64: core out Δ = 7.6e-17, state Δ =
2.2e-16.

## What is gated (tests/m0/check_oracle.py)

1. **Fixture generation + reload determinism** — generate into
   `tests/m0/fixtures/`, reload through `ShardSet` (the real
   naming/format path the C loader will consume), rerun prefill (68
   tokens, 2 KDA chunks) + 6 decode steps: FNV-1a digests of f32/f64
   logits and decode-carried state must match the manifest BITWISE.
2. **Rerun determinism** — prefill twice, bitwise, both modes.
3. **Ordering contract** — extended one-shot chunked prefill vs
   prefill+recurrent-step: f64 < 1e-6 (hard), f32 reported (NOT gated).
4. **KDA unit** — chunked == recurrent over 68 tokens in f64 (< 1e-8).
5. **Indexer legality** — topk indices −1 or < kv_len; width
   index_topk + kpool − 1; every query attends to itself at fixture sizes;
   legality holds as the cache grows in decode.
6. **Sinkhorn sanity** — comb column sums ≈ 1 to ~1e-6.

## Surprises / notes for the M1+ implementers

- **Index pools are rebuilt from the FULL cache every forward**, decode
  included (glm5:811, 900-973). Pools shift as the tail grows: a token's
  pool membership is not stable across steps. Only FULL pools (all 4
  tokens valid) are scored; the incomplete pool is appended as raw tail
  indices (width kpool−1, −1-padded). Candidate validity = the pool's LAST
  token is causally visible.
- **Indexer top-k tie-breaking is unspecified in torch** (`sorted=False`
  even). Pinned here: stable descending sort, lower index first (same as
  the base oracle's ambiguity A6). The router top-k is pinned the same
  way.
- **Mask dedups**: the attention mask is built via scatter_add + `ne(0)`
  (glm5:1244-1247), so a token appearing both in a selected pool and in
  the tail is attended ONCE. The oracle computes attention over the full
  kv with an additive mask (the reference's eager semantics), NOT by
  gathering — the C engine's gather order is a summation-order
  (tolerance-class) difference.
- **Chunked-KDA decay mask overflows by design**: `exp(g_i − g_j)` is +inf
  at masked (j ≥ i) positions; the reference computes the inf products and
  then REPLACES those entries (`masked_fill`). Valid positions (j < i)
  have exponent ≤ 0, so no NaN contamination is possible — the oracle
  suppresses the numpy warnings and relies on the same structure. Tail
  padding is benign because g is cumsum'ed WITHIN each chunk (padded rows
  inherit the last real row's cumulative decay, so the state update's
  `g[..., -1]` is correct for partial chunks).
- **f32-vs-f64 divergence with random weights is dominated by selection
  flips**: router top-k flips ~2 % of entries and indexer top-k flips 4/68
  queries on this fixture → prefill logits rel diff 0.41, argmax flips
  2.9 % (decode chain: rel 0.015, 0 % flips). This is the same near-tie
  cascade the base project documented (tests/m4b/m5/m11a READMEs); C
  gates must use tolerance classes and teacher-forced comparisons, never
  free-running stream equality against the oracle.
- **E4M3 NaN codes (0x7F/0xFF)** are never produced by the fixture
  generator and (per the base project's experience) never occur in lab
  checkpoints — VERIFY at M1 from the real shards.
- **f32 goldens are only bitwise-stable on this host** (libm
  transcendentals vary across platforms — inherited caveat from m11a);
  f64 goldens are platform-stable. The digests in
  `tests/m0/fixtures/manifest.json` are regenerated locally, not checked
  in.
- The oracle is **self-contained**: safetensors IO (from tests/m1
  stutil.py) and the E4M3 codec (from tests/m3 gen_golden.py) are INLINED
  because the V4-era test dirs are pruned at M0. Do not reintroduce
  cross-test imports.
- Divergences from the inherited base oracle's conventions: FP8 scales are
  plain F32 (not UE8M0); fixture naming follows the real checkpoint (not
  the base container names); single `generate()` driver (the base's
  `--full`/`--m8` drivers return with M5/M8).

## Files

- `tools/oracle.py` — the numpy oracle + fixture generator (CLI above)
- `tests/m0/check_oracle.py` — the M0 gate (this battery)
- `tests/m0/fixtures/` — generated weights + goldens + manifest (gitignored)
- `reference/inference/modeling_glm5_next.py` — normative source (read-only)
