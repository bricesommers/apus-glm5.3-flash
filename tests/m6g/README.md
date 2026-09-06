# tests/m6g — GLM expert tiering/cache + pilot prefetch (M6) hard gate

Milestone M6: the GLM expert tiering — streaming FP8 expert slabs from
disk through a bounded RAM cache behind the M5 accessor seam
(`apus_gmodel_expert()`), with pilot prefetch — plus the KDA/DSA state
memory management. (`tests/m6a/m6b/m6c` belong to the inherited V4
engine — hence m6g.)

Run the gate:

    make test-m6g        # golden regen (m5g + cluster) + C gate,
                         # APUS_THREADS=1/4/8 diffed
    make ubsan-m6g       # UBSan build (ASan is broken on the dev Mac)

`tests/m6g/golden/` is gitignored; `make golden-m6g` regenerates it
(oracle weight synthesis -> cluster rewrite -> tools/convert.py, ~10 s).
The m5g fixtures (`make golden-m5g`) are a prerequisite — the gate runs
its bitwise-neutrality modes on the m5g container/goldens.

## What is covered

- **`c/gcache.h` — the GLM expert cache** (NEW code; the V4 `c/cache.h`
  is untouched). Slab records from the v2 manifest `expert_slabs`; a miss
  is ONE pread of the 6-tensor slab (instrumented: `stats.preads ==
  stats.loads == apus_gcache_pread_count`, `bytes_read == preads x
  slab_bytes`); the I/O worker dequantizes gate/up/down with the m3g
  kernel into one BF16 payload `[gate | up | down]` (3*inter*dim codes —
  **48 MiB at real scale**, vs the 24.0059 MiB FP8 slab; §7 budgets are
  in payload bytes). Policy: per-layer LRU + per-forward working set with
  end-of-block promotion, generation-tagged miss overlap on a pthread
  I/O pool, demand/speculative job classes (hot-first pop), RSS guard at
  layer boundaries, payload/staging buffer recycling. Pointer contract:
  resolve views valid until the layer's `layer_end`.
  **P2 speculative yield/drop policy** (the hint-storm fix — reorder-class
  documentation): a speculative (`apus_gcache_hint`) load is pure
  optimization and is DROPPED at admission — never submitted, no slot
  state touched, `stats.spec_dropped++` — when (a) any demand-class load
  is queued or in flight (speculation yields the disk to demand), (b) the
  speculative backlog is already `APUS_GSPEC_QUEUE` deep (default
  2 × io_threads; a deeper queue is stale before it drains — the pilot
  re-hints every token), or (c) RSS is within one fill (raw slab + BF16
  payload) of the RSS-guard budget (speculation must never push the guard
  into dropping decode-live LRU payloads). Demand loads are never
  dropped. Combined with the standing promotion rule (a speculative
  working-set entry has `last == 0`, so it can only displace empty or
  never-resolved slots), speculation cannot evict demand-used payloads at
  all. Bitwise-invariance class: which speculative loads run changes ONLY
  which resolves hit vs miss — a hit and a miss dequant the same slab
  bytes with the same kernel, so output values are unchanged (gate 1 and
  gate 6). Also P2: `slots_per_layer` now derives from the SPARSE-layer
  count (layers with slab records), not `n_layers` — the dense layers
  used to take a budget share they can never use (16 GiB bought 7
  slots/layer instead of 8 on the real 45-layer/42-sparse model).
- **Tiered wiring** (`c/gmodel.h`, `apus_gmodel_open2`): the MoE branch
  resolves ONLY the routed experts — pre-pass selections (the same
  `apus_gmoe_router` `forward2` re-runs, or the forced indices when
  teacher-forcing), demand-hint the union (batch-union storm), resolve
  ascending, NULL the unselected (never dereferenced), `layer_end` after
  `forward2`. The eager path is untouched (m5g digest
  `6ab95f70522def95` unchanged).
  **P4 token-chunked prefill MoE** (`APUS_GPREFILL_MOE_CHUNK`, default 8;
  0 or >= s = unchunked; decode s==1 is always a single chunk): prefill
  runs the selection/hint/resolve/forward/`layer_end` cycle per
  token-chunk instead of once for all s tokens, bounding the live expert
  union at chunk*topk payloads per layer (the unchunked prefill held the
  whole-s union live at once — ~10 GiB of 48 MiB payloads per layer at
  s=26, the P2 41.5 GB footprint peak). Bitwise-invariance class: the MoE
  is per-token independent (each token's ascending-expert bf16-stepped
  accumulation depends only on its own x row) and every resolve returns
  the same dequantized bytes regardless of cache state, so chunk size
  changes ONLY cache timing/eviction, never values — gate 1 pins this
  end to end (tiered chunked == eager unchunked digests; m5g prefill is
  68 tokens = 9 chunks at the default), plus a real-container short-prefill
  check (chunk 0 vs 8, identical token stream — docs/STATUS.md P4).
- **`c/gpilot.h` — the GLM pilot**: dL=1 router lookahead. At layer L's
  post-attention hook (the new `ApusGmodelHooks` surface) it computes
  layer L+1's router input with L+1's own ffn-mHC + post-norm + router
  weights (the exact gated pieces: `apus_gmhc_maps/collapse`,
  `apus_gdsa_rmsnorm`, `apus_gmoe_router/topk_stable`), predicts top-N,
  and issues SPECULATIVE `apus_gcache_hint`s; prefill (s>1) unions
  per-token predictions through a seen bitset. No pilot thread/ring —
  hooks run on the compute thread; overlap comes from the gcache I/O
  pool. Numerics never touched (digest equality pilot-ON vs OFF gated).
- **`apus_gmodel_state_bytes`** — exact KDA (O(1) in context) + DSA
  (linear in kv_cap) state-cache bytes; the forward fails loudly (-1)
  past kv_cap. The long-context policy is pinned below.

## Gates (26 checks)

1. **Bitwise neutrality, host-exp independent**: digests over the full
   output stream (prefill+decode logits, per-layer `trace_h`) on the m5g
   fixture: eager == tiered big-cache == tiered **1-slot** cache ==
   tiered+pilot (`81767c3b646f62e6` on the dev Mac, identical at
   APUS_THREADS=1/4/8 and between -O2/UBSan builds). A cache hit and a
   cache miss return byte-identical BF16 expert weights — dequant-on-fill
   is the same deterministic m3g kernel on the same slab bytes. In the
   probe-BITWISE tier the eager run is also memcmp-anchored to the
   oracle logits goldens.
2. **Eviction correctness**: `slots_per_layer=1` forces eviction churn
   (41 evictions measured); the digest still equals the eager one.
3. **One pread per fill**: see above (16 fills == 16 preads on the m5g
   big-cache run; `ApusStLazy` instrumentation agrees).
4. **Prefetch measurement** (locality fixtures, SYNCHRONOUS I/O mode for
   timing-independent counters): pilot recall on clustered vs random
   containers, prefetch coverage (`demand_loads` with vs without pilot),
   prefill union hints, pilot recall accounting non-vacuous
   (pilot_k=5 < E=8 on the m5g fixture).
5. **State sizing**: `apus_gmodel_state_bytes` == hand-computed bytes
   from the fixture dims.
6. **P2 speculative yield/drop policy** (sync I/O, `rss_budget_bytes=1` —
   the queue/backlog rules are vacuous in sync mode, the RSS rule is not):
   every speculative hint is dropped at admission (`hint_loads == 0`,
   `spec_dropped > 0`), demand loads still run, the digest equals
   pilot-OFF (a resolve's bytes never depend on which speculative loads
   ran), and the RSS guard + EMPTY-slot resubmit path is exercised
   (every resolve misses; `rss_drops > 0`).

Full stdout is diffed across APUS_THREADS=1/4/8; only synchronous
(timing-independent) counters are printed. Timing-dependent counters
(`waits/wait_ns/deq_ns`, the async pilot run's cache stats) print under
`APUS_M6G_STATS=1` (reporting only). Why: an unselected pilot-hinted
load that spans a `layer_end` legitimately races the generation bump
(straggler-reset path), so async pilot-mode cache counters can wobble
run-to-run; model VALUES never do — resolves always wait for the
gen-checked payload.

## Locality fixtures (tests/m6g/golden, gen_cluster.py)

Two tiny-config **E=32** containers (2 MoE layers each) + a clustered
workload (16 prompt + 64 decode tokens in runs of one cluster):

- `cluster_container`: embeddings and every sparse layer's router gate
  rows share a 4-cluster 64-dim block basis (8 experts/cluster), so
  hidden states stay cluster-flavored across the stack — dL=1
  predictions land in the right cluster and consecutive same-cluster
  tokens reuse experts.
- `random_container`: same config, pure random oracle weights — the
  machinery-validation control. **Caveat** (same as the base project's
  tools/measure_router_locality.py): synthetic random-weight recall is
  dominated by the constant selection bias, NOT a tuning input.

Measured on the dev Mac (slots=4/layer, pilot_k=8, sync I/O):

| run                       | pilot recall | demand_loads | hits | misses |
|---------------------------|--------------|--------------|------|--------|
| cluster + pilot           | 384/384 (1.000) | 0         | 338  | 70     |
| cluster, no pilot         | —            | 70           | 338  | 70     |
| random + pilot (control)  | 346/384 (0.901) | 33        | —    | —      |

Coverage: every cluster miss was pilot-predicted (demand 0 vs 70). The
random control's 0.901 is the constant-bias artifact above (its recall
floor is |top-N ∩ bias-top|/topk, not chance 8/32); the gate asserts
`recall_random < recall_cluster` and `recall_cluster >= 0.70`. The m5g
fixture run (E=8, pilot_k=5, async pool): recall 45/48, demand 0,
waits 0, dequant 0.59 ms over 16 fills.

Real-scale expectations (§7): 12,384 experts, top-8, 48 MiB payloads;
~10–14 GB cache ≈ 210–290 resident experts (1.7–2.4%); decode tok/s ≈
B / (7.9 GiB × miss-rate) with B the NVMe slab bandwidth — pilot recall
is the lever, to be measured on real weights at M7 with
tools/measure_router_locality.py (its caveat applies until then).

## Long-context DSA KV policy (the R7 decision, pinned)

- **No KV eviction, no KV quantization.** Both change attention outputs
  — a quality-invariant violation needing explicit user approval. The
  engine keeps full-fidelity BF16 caches up to the caller-chosen
  `kv_cap` and FAILS LOUDLY (-1) beyond it; `apus_gmodel_state_bytes`
  gives the exact footprint for budgeting.
- **Layout kept**: per-head K/V caches head-major `[H][cap*d]`
  (attention reads are per-head streams; m5g-gated), indexer caches
  token-major `[cap*ID]` (the pool rebuild reads whole rows). A
  token-major K/V re-layout was considered and rejected: with no
  eviction it buys nothing, and it would churn the gated m5g state
  layout for zero numeric benefit.
- **Correction to §7's "~17 GB at 1M ctx"**: that estimate priced a
  LATENT-form cache (kv_lora 512 + packed indexer ≈ 1.5 KB/token/layer).
  The landed m5g design caches the EXPANDED per-head K/V (64 heads x
  256) + indexer rows ≈ 64.5 KB/token/layer ≈ **710 KB/token over the
  11 DSA layers** (~23 GB at 32K tokens, ~700 GB at 1M). On the 32 GB
  dev machine with ~14.4 GiB dense+smalls resident and a multi-GB expert
  cache, full-fidelity DSA context is therefore on the order of
  10–20K tokens.
  Latent-form caching (re-expand per attention, or gather only the
  indexer-selected set — bitwise-safe in principle: masked lanes
  contribute exact +0.0f) plus selected-set attention is the designed
  path to §7's numbers and is deferred to M7+ with its own oracle gate.
  KDA state is O(1): 34 layers x FP32 [64,128,128] ≈ 136 MiB + conv
  states, context-length independent, as designed.

## Files

- `c/gcache.h` — the GLM expert cache (`APUS_GCACHE_IMPLEMENTATION`).
- `c/gpilot.h` — the GLM pilot (`APUS_GPILOT_IMPLEMENTATION`).
- `c/gmodel.h` — ADDITIVE: `apus_gmodel_open2`/`ApusGmodelTierCfg`
  (tiered open), tiered MoE wiring, `ApusGmodelHooks` +
  `apus_gmodel_set_hooks`, `apus_gmodel_pilot_view`,
  `apus_gmodel_cache`, `apus_gmodel_state_bytes`. Eager path untouched.
- `tests/m6g/test_m6g.c` — the gate.
- `tests/m6g/gen_cluster.py` — locality fixture generator.
- `tests/m5g/test_m5g.c` — two added `#define`s
  (`APUS_GCACHE_IMPLEMENTATION`, `APUS_COMPAT_IMPLEMENTATION`) the
  gmodel TU now needs; digest unchanged.
