# tests/m5g — GLM full-model forward (c/gmodel.h) hard gate

Milestone M5: the full GLM-5.3-Flash text-model forward in C on synthetic
weights — M1 v2 container loading, config parsing, and the 45-layer mHC
stack (token embed → per-layer mHC collapse → weighted RMSNorm → KDA |
DSA → mHC expand → mHC collapse → weighted RMSNorm → dense MLP | MoE →
mHC expand → unweighted-mean HyperHead → final RMSNorm → BF16 lm_head
logits) — gated BITWISE against the M0 numpy oracle's `model_forward`
(f32-faithful mode), prefill (CHUNKED KDA) and an 8-step decode chain
(RECURRENT KDA) per the §4 KDA ORDERING CONTRACT. MTP is out (deferred
M8+). (tests/m5 belongs to the inherited V4 engine — hence m5g.)

Run the gate:

    make test-m5g        # golden regen + C gate, APUS_THREADS=1/4/8 diffed
    make ubsan-m5g       # UBSan build (ASan is broken on the dev Mac)

`tests/m5g/golden/` is gitignored; `make golden-m5g` regenerates it
(~15 s: weight synthesis → source-checkpoint rewrite → tools/convert.py
→ oracle prefill+decode from the converted container).

## What is covered

- **Config parsing** (`apus_gmodel_config_load`): the flat oracle fixture
  config AND the real `reference/config.json` (`text_config` subset +
  `linear_attn_config`). The C gate asserts the real config's facts
  (45 layers, hidden 4096, vocab 154,880, hc_mult 4, 64/128 KDA, 288
  experts top-8 ×2.5, swiglu_limit 10, index_topk 2048 kpool 4) and that
  the layer-pattern RULE (`apus_gmodel_layer_patterns`: DSA at i%4==3,
  dense below first_k_dense_replace=3) reproduces the explicit 45-layer
  `layer_types`/`mlp_layer_types` lists with 0 mismatches.
- **Container loading (M1 format v2)**: dense tensors resolved by name
  through `model.safetensors.index.json` (c/st.h ApusStSet, zero-copy
  views); `apus.index.json` verified (`format_version: 2`, `model_type:
  glm5_next`); every routed expert read through its manifest slab record
  with ONE pread (the M1 coalescing invariant) via c/st.h ApusStLazy,
  members carved by the pinned order + config-derived shapes (slab nbytes
  hard-checked), then dequantized once (m3g `apus_fp8blk_dequant`).
  Dense MLPs and shared experts are dequantized at load as well; all
  other BF16/F32 tensors stay zero-copy views into shard buffers.
- **Forward**: embed replication ×4 → 5-layer tiny-config stack (KDA
  dense 0–2, DSA MoE 3, KDA MoE 4 — every wiring combination the real
  model has except DSA+dense, which cannot occur) → HyperHead mean →
  final RMSNorm → lm_head. Per-layer block-output streams h, logits per
  call, and ALL final states (KDA conv + fp32 rec_state per KDA layer;
  KV + indexer caches per DSA layer) compared.
- **State/capacity**: `apus_gmodel_state_new(kv_cap)` owns the caches;
  the engine scratch arena (below) is grown on demand.
- **Scratch arena**: the M4a note is resolved — `apus_gmoe_forward`'s
  per-call mallocs are re-homed to `ApusGmoeScratch`
  (`apus_gmoe_scratch_init/free` + `apus_gmoe_forward2`), owned by the
  model state and reused across layers/calls. Pure allocation change:
  m4g reruns bitwise-identical (digest `7f24a242b80ea050` unchanged).
  The model's own per-call buffers live in the same state-owned arena.
- **M6 seam**: the ONLY expert access path is `apus_gmodel_expert()`
  (the MoE wiring re-fills the eg/eu/ed pointer arrays from it every
  call) — M6 slots the slab-streaming cache behind that accessor without
  touching the forward. M5 keeps every expert resident (eager dequant).

## Gate tiers (the m4g/m4h host-exp pattern, model level)

The only non-IEEE-exact op is `exp`. The C test probes 1024 stored
np.exp values at startup (macOS arm64: 0 diffs; Linux x86_64: 0 diffs
with numpy pinned to its baseline exp kernel via the Makefile's
NPY_DISABLE_CPU_FEATURES recipe):

- **BITWISE tier** (probe clean): EVERY golden memcmp-compared —
  per-layer streams h, logits (prefill + every decode step), KDA
  conv/rec states, DSA caches — for BOTH runs: run A (free-running
  model) and run B (teacher-forced selections; must equal run A exactly
  here, which also exercises the forcing path bitwise). Engaged on the
  dev Mac AND in the Linux docker.
- **TOLERANCE tier** (probe differs): ONLY run B is compared — run A
  still executes (crash/UB coverage) but is not checked, because
  free-running selections are host-exp fragile at model level (a single
  flipped indexer selection poisons the whole downstream chain; skipping
  individual queries is impossible, unlike m4h). Run B teacher-forces
  the router top-k (goldens `forced_idx_*`) and the indexer top-k
  (`forced_topk_*`) from the oracle, then compares values with the
  MEASURED classes below.

Why forcing is safe to gate with: router selections are margin-protected
(the generator re-draws the whole fixture until every biased-score top-k
boundary gap across every MoE layer/token/step exceeds 1e-4 — one
re-draw was needed at seed 20260830, final margin 3.5e-4 — the m4g
policy), and indexer near-ties are STRUCTURAL (relu-clip zero ties;
24 fragile queries in this fixture's prefill, recorded in the manifest
under `fragile_*` — m4h showed no re-draw escapes them). The BITWISE
tier compares everything (exact ties are deterministic: stable
descending, lower index first, in both numpy's argsort and the C top-k).

### Tolerance classes (measured, not estimated)

Measurement recipe (documented for re-runs; not part of the gate):
monkeypatch `np.exp` with a DETERMINISTIC 1-ulp perturbation of EVERY
float32 exp result (direction from the input bits — far harsher than any
probed host; unpinned Linux X86_V3 differs on ~40 % of values), regenerate
the fixtures into a scratch dir, run the gate against them with
`APUS_M5G_TIER=tol` (env override that forces the tolerance tier). Worst
observed element drifts (teacher-forced run vs perturbed goldens):

| quantity                        | worst abs | worst rel (near-zero elements) |
|---|---|---|
| mHC streams h (bf16 codes)      | 0.125     | huge (code flips at tiny values) |
| lm_head logits (bf16 codes)     | 0.033     | huge |
| KDA conv + DSA caches (codes)   | 0.023     | huge |
| KDA rec_state (fp32)            | 0.0012    | huge |

The gate classes carry ~2× headroom over that worst case: h rel ≤ 0.25
or abs ≤ 0.25; logits rel ≤ 0.25 or abs ≤ 0.05; bf16 state caches rel ≤
0.25 or abs ≤ 0.05; fp32 rec_state rel ≤ 0.25 or abs ≤ 5e-3. Relative
clauses are meaningless near zero — small-magnitude elements
legitimately flip several bf16 codes under compounding; the abs clauses
are the real teeth (a WRONG-math bug produces O(1) diffs in most
elements, not small-abs diffs in a few percent).

Either way the full output (incl. the FNV-1a digest of the C outputs) is
diffed across APUS_THREADS=1/4/8: the gemms run on the
bitwise-at-any-thread-count m3g mt kernels and everything else is
single-threaded. The digest is identical between the -O2 and UBSan -O1
builds on the same host (`6ab95f70522def95` on the dev Mac). Golden
BYTES are host-pinned (same caveat as m0/m4g/m4h f32 goldens); each host
regenerates its own fixtures.

## Reorder classes

None. The model level composes only the already-gated m3g/m4a/m4b
orders; per-token mHC calls are bitwise == the oracle's batched rows
(the maps are per-token independent), and per-token dense-MLP GEMVs are
bitwise == the batched oracle rows (m3g M-independence, already pinned
by m4g xp2). The KDA phase is explicit in the API
(`apus_gmodel_prefill` → chunked, `apus_gmodel_decode_step` →
recurrent), never inferred from s.

## Real-shape smoke (decision: skipped)

A real-shape (hidden 4096, 288 experts/layer) synthetic fixture was
considered and rejected: a single real-shape MoE layer's experts are
7.2 GB of FP8 (14.5 GB dequantized BF16) and the oracle's deterministic
`_mm` (a Python loop per contraction) would take hours. The tiny config
keeps every FP8 matrix a multiple of 128 in both dims (the real block
grid) and exercises every wiring combination; real-shape coverage
arrives with real weights at M7.

## What is tested (180 checks bitwise / 114 tolerance, 0 failures)

- 5 reference-config checks (facts + rule-vs-explicit layer pattern).
- Fixture-config + container open + per-layer kinds + expert accessor.
- Run A + run B: prefill logits, per-layer prefill h (5), 8 decode
  logits, per-layer decode h (5 per step), final pos, per-KDA-layer
  conv/rec states (4 layers), per-DSA-layer k/v/idx_k/idx_gate caches
  (1 layer).

UBSan clean; `leaks` clean (0). APUS_THREADS=1/4/8 output identical.

## Files

- `c/gmodel.h` — the model (`APUS_GMODEL_IMPLEMENTATION`): config
  parser, M1 v2 container loader (ApusStSet + slab ApusStLazy), state +
  scratch arena, prefill/decode forward, the `apus_gmodel_expert()` M6
  seam, `ApusGmodelForces` (test-only teacher-forcing).
- `c/gmoe.h` — ADDITIVE: `ApusGmoeScratch` + `apus_gmoe_forward2` (the
  M4a malloc re-home + forced-idx variant); `apus_gmoe_forward` kept as
  a wrapper (m4g unchanged).
- `c/gdsa.h` — ADDITIVE: `apus_gdsa_forward2` (forced-topk variant);
  `apus_gdsa_forward` kept as a wrapper (m4h unchanged).
- `tools/oracle.py` — ADDITIVE: `layer_interm` hook on
  `model_forward`/`prefill`/`decode_step` (per-layer block interms +
  `block_out_h`); all m0–m4h gates rerun green.
- `tests/m5g/gen_golden.py` — fixture generator (source checkpoint →
  tools/convert.py → oracle from the container; margin re-draws; exp
  probe).
- `tests/m5g/test_m5g.c` — the gate (probe → tier, run A + run B,
  thread-invariance digest).
- `tests/m5g/golden/` — generated fixtures (gitignored).
