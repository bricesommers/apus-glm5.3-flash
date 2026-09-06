# tests/m8g — GLM MTP (classic NextN) oracle fixtures (M8a) + C gate (M8b)

Milestone M8a: numpy goldens for the MTP (NextN) draft head —
`tools/oracle.py` `mtp_forward` / `mtp_chain` — on synthetic tiny-config
containers that carry the `layers.<L>.*` MTP block through the real M1
converter path (`apus-mtp-*` shard group). The M8b C port (`c/gmtp.h` +
the per-token-interleaved batched decode path, see the 2026-09-04 M8
KICKOFF entry in docs/STATUS.md) is gated BITWISE against these goldens.

Run:

    make golden-m8g      # regenerate fixtures (numpy goldens)
    make test-m8g        # golden-m8g + fixture integrity verify + the M8b
                         # C gate (tests/m8g/test_m8g.c) at APUS_THREADS
                         # 1/4/8 (outputs diffed) + the bin/apus CLI legs
    make ubsan-m8g       # the C gate under UBSan (-ffp-contract=off pinned)

`tests/m8g/golden/` is gitignored; regeneration is deterministic from the
seed recorded in each case's manifest.

## The M8b C gate (tests/m8g/test_m8g.c, 248 checks)

Per case (kda_top, dsa_top):

- **model + h surface**: prefill/decode logits and the hnorm input built
  from the M8b h-out surface (`apus_gmodel_prefill_h` /
  `apus_gmodel_decode_batch` h_out [s,hc*dim] + yn_out [s,dim]) vs
  pre_logits / dec_logits / pair_h — validates the engine upstream of the
  MTP glue. Two-tier (probe): bitwise when the host expf matches numpy's
  baseline, tolerance (m5g measured classes) with teacher-forced
  selections otherwise.
- **decode-batch interleave** (THE hard blocker): 6 tokens decoded
  one-by-one vs ONE `apus_gmodel_decode_batch` call from identical state —
  per-position logits and the full state digest (KDA conv+rec, DSA caches
  over the live rows, pos) BITWISE equal. Unconditional, host-independent:
  the batch is bitwise by CONSTRUCTION, not tolerance — see below.
- **mtp_forward replay + draft chain** vs replay_logits / replay_out_h /
  chain_drafts / chain_logits / chain_out_h goldens (two-tier; run B
  teacher-forces router/indexer selections AND the chain seed/draft ids —
  the near-tie class).
- **equivalence**: spec decode (depth 1/2/3) vs non-spec decode, greedy
  AND sampled (temp 0.8, top_p 0.95, seed 12345) — emitted streams bitwise
  identical; **rollback**: state digest after the spec run == digest after
  decoding exactly the emitted tokens non-speculatively (rejected drafts
  leave no trace: pos, KDA conv/rec states, DSA n + live cache rows).
- **forced-draft patterns** (`draft_override` replacing the MTP proposer):
  truth-oracle (100% accept incl. the full-match bonus path), garbage
  (0%), mixed (partial) — streams + state digests still bitwise ==
  non-spec, accept stats match the pattern. Random-weight acceptance is
  ~0, so these patterns are how the accept paths get exercised.
- **tiered**: the greedy depth-3 equivalence re-run on the M6 cache
  (slots_per_layer=1, synchronous I/O) serving the MTP slabs at store
  layer n_main+0 — stream + state digest bitwise == the eager run.

The Makefile `test-m8g` adds two `bin/apus` CLI legs: `--spec --spec-k 3`
vs non-spec streams bitwise on the dsa_top container, and `--spec` on an
MTP-less model dir must fail (loud).

## The M8b decode-batch interleave (why bitwise by construction)

GLM's §4 KDA ordering contract (chunked prefill != recurrent decode
bitwise) means the parent engine's chunk-invariant batched verify does NOT
transfer. The verify batch instead uses `apus_gmodel_decode_batch`:

- KDA layers: `apus_gkda_forward(decode=1, s>1)` — batched projections
  (row-independent bitwise, the m3g pin), then the conv window stepped
  token by token (`apus_gkda_conv_prefill` IS the chained per-token body)
  and `apus_gkda_recurrent` chaining s tokens through the same state.
- DSA layers: `apus_gdsa_decode_batch` — the per-token call loop: each
  token's kv append, indexer cache append + full-pool rebuild, selection
  and attention run at cache count base+t+1 (NEVER base+s). The one-shot
  batched forward is the PREFILL ordering and is NOT decode-equal: its
  single pool rebuild would include not-yet-written tokens (masked, but
  the top-k row layout clamps differently) and its attention softmax
  pairwise sums would run over base+s entries. The interleave IS the
  sequential call sequence, hence bitwise.
- mHC maps/collapse/expand, norms, MoE: per-token independent (the MoE
  stays s-wide batched — that is where the slab-I/O amortization win is;
  DSA dense projections run as per-token GEMVs — a measured-performance
  choice, bitwise-free; batching them is M8c perf work).

Rollback: ApusGsnap memcpy-restores pos + every KDA layer's conv+rec
state; DSA layers rewind n only (rejected-tail rows are dead capacity —
the deterministic re-feed rewrites them; indexer/attention never read
beyond n). The MTP state (DSA-kind) rolls back pos + n the same way. The
M6 cache is never rolled back (numerics-neutral by contract).

## Semantics pinned here

MTP block (`layers.<num_hidden_layers>.*` in the checkpoint/container; HF
drops it at load — the normative reference is SGLang `deepseek_nextn.py`,
and the container facts were verified against
`weights/glm-5.3-flash/apus.index.json`):

- Draft step: `eh = cat([rms_norm(embed(tok), enorm),
  rms_norm(prev_h, hnorm)], -1)` [s, 8192] → `x = bf16_linear(eh,
  eh_proj)` (**eh_proj is BF16** [4096, 8192] in the real checkpoint, not
  FP8) → ONE full decoder layer (DSA + sparse MoE — the real MTP block
  HAS a shared expert, `layers.45.mlp.shared_experts.*`) with PLAIN
  residuals (no mHC): each residual add is one bf16 rounding
  (eager-torch add semantics) → fused add+norm `out = rms_norm(x,
  shared_head.norm)` → `logits = bf16_linear(out, shared lm_head)`. The
  block shares the main model's embed_tokens and lm_head (it has none of
  its own).
- Drafts are always **argmax** of the MTP logits. The chain feeds the
  draft's own post-`shared_head.norm` hidden back as `prev_h` with the
  drafted token's embedding (vLLM PR #47448 semantics; the parent
  engine's `../Apus/c/mtp.h` `apus_spec_chain` shape).
- **hnorm input**: the trunk stream is mHC [s, 4, 4096]; hnorm is
  4096-wide. PINNED EMPIRICALLY on the real 306 GiB container — the
  post-gate7 re-pin (2026-09-05; the original pin was measured with the
  kv_b_proj split bug and is invalid): **`postnorm`** (the post-final-norm
  row — the SGLang/vLLM reference semantics) won the engine acceptance
  sweep decisively (85.3% accept, 2.53 tok/batch at spec-k 3, 1.47×
  faster than non-spec; table in the STATUS re-pin entry). The pinned
  value lives in `oracle.MTP_HNORM_INPUT` (used by `oracle.mtp_hnorm_input`
  and these fixtures) and `c/gmtp.h APUS_GMTP_HNORM_SRC`
  (-D-overridable).
- **(h, id) pairing**: `oracle.MTP_PAIR_LAG` = **1** (PINNED with the
  same sweep — the DeepSeek-V3/SGLang EAGLE bookkeeping: (h_{p-1}, tok_p)
  predicts tok_{p+1}; the C engine implements the lag-1 replay by
  shifting the hidden tap one position and carrying `h_prev`).
  lag 0 = pair (h_p, tok_p) at position p predicts tok_{p+1} (the parent
  engine's convention). The m8g fixtures at lag 1 carry
  `replay_len = n_tokens - 1` (position 0 has no pair) and the gate's
  h-surface check reads pair rows pairing-aware.
- **Accept rule** (from the M8 KICKOFF, ported from base M8): a draft is
  accepted iff it equals the MAIN MODEL'S OWN pick at that position
  (argmax for greedy; the model's own apus_sample draw for sampled — one
  RNG uniform per emitted token, drafts consume none). Invariant:
  bitwise identity of the per-seed token stream.
- The MTP block's LayerState is DSA-kind; its pos tracks the main
  model's (bumped by `mtp_forward` itself).

## Fixture inventory (`golden/<case>/`, cases `kda_top` / `dsa_top`)

Two tiny configs — `kda_top`: L=2 [KDA, KDA] (mlp dense, sparse);
`dsa_top`: L=2 [KDA, DSA] (mlp dense, sparse) — plus the MTP block at
layers.2. Both exercise the draft chain seeded from a main-model decode
step (the C engine's h-out surface differs by top-layer kind). Per case:

- `config.json` — flat oracle fixture config (the C parser format, m5g).
- `container/` — the M1 v2 container under test (mtp shard group).
- `prompt_ids` [12] i32, `decode_ids` [2] i32.
- `pre_logits` [12, V] f32, `dec_logits` [2, V] f32 — the main-model
  stream (the M8b test reuses these to validate its h-out surface
  upstream of the MTP glue).
- `pair_ids` [s] i32, `pair_h` [s, dim] f32 — the batched `mtp_forward`
  replay input (true pairs per the pinned pairing; s = 14 at lag 0).
- `replay_logits` [s, V] f32, `replay_out_h` [s, dim] f32 — batched
  `mtp_forward` goldens.
- `chain_drafts` [3] i32, `chain_logits` [3, V] f32, `chain_out_h`
  [3, dim] f32 — the draft chain: drafts[0] = argmax of the replay's
  last logits row, drafts[1..] from `mtp_chain` over the replay-built
  state (the engine flow).
- `forced_idx_*` / `forced_topk_*` — teacher-forcing selections for the
  M8b tolerance tier (main MoE/DSA layers prefill+decode, MTP replay
  [s, topk]/[s, width], MTP chain per step), m5g conventions.
- `probe_x` / `probe_y` — the f32 exp probe (host-transcendental
  detection, m4g/m5g convention).
- `manifest.json` / `manifest.txt` — dims, seed, router min margin, the
  pinned `hnorm_input`/`pair_lag`, FNV-1a digests.

## Conventions

- Router near-ties are margin-protected (min biased-score top-k boundary
  gap ≥ 1e-4 across EVERY MoE call incl. the MTP replay rows and chain
  steps; the whole case is re-drawn below that — the m4g/m5g policy).
- The `golden-m8g` recipe pins numpy to its baseline exp kernel via
  `NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX AVX512_SPR"` so the
  goldens are host-independent on x86 (macOS arm64 needs no pin) — same
  as golden-m4g/m4h/m5g.
- Numerics ordering notes M8b must replicate: the oracle's `_mm`
  sequential-k two-rounding matmul; `rms_norm`'s two bf16 roundings; the
  eh concat order (enorm(embed) FIRST, hnorm(prev_h) SECOND); plain
  residual adds bf16-rounded once each; `shared_head.norm` applied to
  the final summed x (the fused add+norm target); argmax drafts.
