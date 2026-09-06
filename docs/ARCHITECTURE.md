# apus-glm-5.3-flash — Architecture & Implementation Plan

Local inference engine for **[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**
(`model_type: glm5_next`, text-only) on consumer hardware, by streaming routed
experts from NVMe through a disk → RAM cache hierarchy.

This project is an **adapter of the apus engine** (the DeepSeek-V4-Flash repo,
`../Apus`, M15: Windows port, tri-platform CI, m1–m15 test battery), seeded
2026-08-29 via `git clone --no-hardlinks` (code only, no weights, no remotes).
KDA, sigmoid-router semantics, and BF16 kernels port across the adapter seam
from `../Apus-Ling-3.0-Flash-bf16` (which is behind on engine work — no
Windows port; do not back-port engine pieces from it). Both sibling repos are
READ-ONLY references. Ultimate lineage:
[colibri](https://github.com/JustVugg/colibri) (Apache-2.0); per-file
attribution headers are carried.

**Scope: text-only.** The checkpoint's vision tower (`model.visual.*`) is
excluded and stripped at conversion. MTP is deferred to M8+ (it does not
affect main-model logits).

**Status: M0–M8b complete (see docs/STATUS.md — authoritative).
Weight container sealed 2026-09-04; real-model smoke PASSED bitwise;
real-model spec smoke (M8c) is the remaining weight-gated item.**

---

## 1. Goals, non-goals, hard invariants

Goals:
- Run GLM-5.3-Flash locally with an OpenAI-compatible server.
- C11 core, no dependencies beyond libc/pthreads (OpenMP optional, GPU
  backends optional and strictly additive).
- Insufficient fast memory costs **speed only, never quality**.

Hard invariants (any violation requires explicit user approval):
- Never silently change numeric precision of any tensor or computation.
  Any token-altering change is presumptively wrong unless it is a documented,
  measured reorder class in the relevant test README.
- Never change router semantics (scoring fn, bias correction, top-k,
  normalization, scaling).
- Expert weights are stored and computed from the original FP8-E4M3 +
  F32 `weight_scale_inv` representation. No transcoding to int4/int8/UE8M0
  as a "convenience"; the converter is a byte-identical repack.
- Lab-original weights only (zai-org official repo); no community quants.

Non-goals: training/fine-tuning, multi-user high-throughput serving, vision,
MTP (until M8+). The dev target is Apple Silicon; Linux/x86_64 (AVX2) and
Windows (MinGW-w64/UCRT64, shims in `c/compat.h`) must keep working — the
M15 tri-platform engine is a hard-won base feature, do not regress it.

## 2. Target hardware profile

- Machine: **MacBook Pro, Apple M1 Pro, 32 GB unified memory**.
- CPU: ARM NEON → **NEON kernels are the primary path**; AVX2 is the x86
  port and must stay bitwise-pinned to the scalar reference.
- GPU: Apple GPU via **Metal, zero-copy unified memory** (M7). Until then
  the engine is CPU+disk.
- I/O: macOS `F_NOCACHE` pread + pthread I/O pool (`c/compat.h` pattern);
  io_uring is Linux-only.
- Storage: internal NVMe for the converted container (see §8 for the disk
  conflict; weights may end up on an external drive via symlink per project
  rules).
- Weights: **not downloaded** — explicit WAIT, needs user go-ahead.

## 3. GLM-5.3-Flash — verified architecture reference

Sources, in order of authority:
1. **HF transformers `main`, PR #48342 (+#47625)** — the ONLY native
   implementation of `glm5_next` (not in v5.16.0; no vLLM/SGLang port of the
   main model). A verbatim copy lives in-repo at
   `reference/inference/modeling_glm5_next.py` (generated from
   `modular_glm5_next.py`, also in-repo). **The HF source wins any conflict
   with this document.**
2. `reference/config.json` (`text_config`), `generation_config.json`,
   `tokenizer_config.json`, `chat_template.jinja`, `tokenizer.json`.
3. Phase A checkpoint anatomy (2026-08-29): safetensors header reads over
   all 62 shards (ranged metadata reads only, no weight download) — tensor
   names, shapes, dtypes.

Note on HF kernels: several reference functions
(`chunk_kda`/`fused_recurrent_kda`/`causal_conv1d`, `RMSNorm`,
`RMSNormGated`) are decorated `use_kernel_*_from_hub_with_fallback` — with
the `fla`/hub kernels installed, HF runs Triton kernels; otherwise the
pure-PyTorch fallbacks in `modeling_glm5_next.py` execute. **The oracle
(M0) must pin the pure-PyTorch fallback path** so goldens are
environment-independent and bitwise reproducible.

### 3.1 Top level

| Field | Value |
|---|---|
| Layers | 45 (+1 classic-NextN MTP layer, deferred) |
| Hidden size | 4096 |
| Vocab | 154,880 (BPE, `tokenizer.json`) |
| Context | 1,048,576 |
| Layer pattern | 34 KDA linear-attention + 11 DSA full-attention (layers 3,7,…,43), interleaved 3:1 |
| Residual stream | **mHC** ×4 on all 45 main layers (Sinkhorn-20) |
| MoE | layers 3–44 sparse (288 routed + 1 shared), layers 0–2 dense |
| Norms | weighted RMSNorm eps=1e-5 (pre-attn, pre-MLP, final); unweighted RMSNorm eps=1e-5 inside mHC |
| RoPE | **none anywhere in the text model** (`qk_rope_head_dim: 0`; `position_embeddings=None` in the text forward) |
| Embed/head | untied (`tie_word_embeddings: false`), BF16 |
| EOS | {154820, 154827, 154829}; pad 154820 |
| Checkpoint | FP8-E4M3, 128×128 blocks, F32 `weight_scale_inv`; 62 shards, 305.8 GiB |

Layer map: `layer_types` in config — KDA on all layers except
{3,7,11,15,19,23,27,31,35,39,43} which are `deepseek_sparse_attention`.
`mlp_layer_types`: dense on 0–2 (`first_k_dense_replace: 3`), sparse on
3–44.

### 3.2 KDA linear attention (34 layers; `Glm5NextTextLinearAttention`)

Kimi-style delta-rule linear attention, 64 heads × 128 (`qkv_dim` 8192):

- Projections (no bias): `q/k/v_proj` 4096→8192 each, concatenated and run
  through a **fused causal depthwise conv, k=4** over the 24,576 channels
  (SiLU activation). Decode uses the single-step conv-state update
  (state = last k−1 = 3 inputs per channel); prefill uses the full causal
  conv. Conv weights are BF16 in the checkpoint; HF keeps `conv1d` in its
  FP32-strict module list (upcast at load, conv computed in weight dtype) —
  the oracle must record which it does, and the C side must match.
- **Forget gate (low-rank)**: `f_a` 4096→128, `f_b` 128→8192, F32 `dt_bias`
  [8192], F32 `A_log` [64]. With `gate_lower_bound: -5.0` the gate is the
  safe-lower-bound form: `g = -5 · sigmoid(exp(A_log)·(f_b(f_a(x)) +
  dt_bias))` per head-channel, so the decay `exp(g) ∈ (e⁻⁵, 1)`.
- **Beta (input gate)**: `β = sigmoid(b_proj(x))`, `b_proj` 4096→64.
- **Delta rule**: q,k are l2-normalized (eps 1e-6, in FP32, inside the
  kernel), scale 128^-0.5; recurrence per token:
  `S ← S·exp(g); δ ← (v − Sᵀk)·β; S ← S + k·δᵀ; o ← Sᵀq`.
  **State S is FP32** [64, 128, 128] per layer (~4 MiB), persisted in FP32
  across tokens — KDA context cost is O(1) in sequence length.
- **Output**: gated RMSNorm (`o_norm`, per-head 128 dims, eps 1e-5, strict
  FP32 norm, sigmoid gate from low-rank `g_a` 4096→128 / `g_b` 128→8192),
  then `o_proj` 8192→4096. All KDA smalls BF16 in the checkpoint (A_log,
  dt_bias F32).
- Prefill vs decode: prefill runs the **chunked** algorithm
  (`chunk_kimi_delta_attention`, chunk size 64, FP32, UT-transform intra-
  chunk solve); decode runs the **recurrent** single step
  (`recurrent_kimi_delta_attention`, FP32). See §4 — these two orderings
  are NOT bit-identical and the engine pins each phase separately.

### 3.3 DSA layers (11 layers: 3,7,…,43; `Glm5NextTextAttention` + indexer)

MLA, **pure NoPE** — `qk_rope_head_dim: 0`, no rotary anywhere:

- Q LoRA: `q_a_proj` 4096→1536 (FP8), weighted RMSNorm(1536, 1e-5),
  `q_b_proj` 1536→64×256 = 16384 (FP8).
- KV LoRA: `kv_a_proj_with_mqa` 4096→512 (FP8; the +rope tail has width 0),
  weighted RMSNorm(512, 1e-5), `kv_b_proj` 512→64×(256+256) = 32768
  (**BF16**, in `modules_to_not_convert`). Per head: k_nope 256, v 256.
- Attention: 64 heads, qk_head_dim 256, scaling 256^-0.5, FP32 softmax,
  sparse mask from the indexer's top-k indices (`o_proj` 16384→4096, FP8).
  The KV cache stores the 512-dim latent per token per DSA layer.

**Lightning Indexer** (`Glm5NextTextIndexer`, all params BF16, runs under
`no_grad`): 32 heads × 128.
- Queries from the q latent: `wq_b` 1536→4096; keys `wk` 4096→128 +
  LayerNorm(128, eps 1e-6, with bias); per-head weights `weights_proj`
  4096→32, scaled 32^-0.5; score scale 128^-0.5, ReLU, sum over heads.
- **k-pool compression (kpool=4)**: per-token packed cache state
  [k(128) | gate(128) | valid(1)]. Pools of 4 tokens anchored at the first
  valid token; pool key = softmax(gate_scores + learned `ape[4,128]`)
  weighted average of member keys (`compress_gate` 4096→128).
- Selection: top-512 pools (= 2048/4) by index score; a pool is selectable
  only if its **last** token is visible (causality); selected pools expand
  back to 2048 raw token indices. The current incomplete **tail pool is
  always appended** (up to 3 raw tokens) → selection width **2051**;
  invalid entries are −1.
- Unlike the base engine's CSA indexer: **no Hadamard rotation, no FP4/FP8
  score quantization** — scores stay FP32 end to end. This is a fresh port,
  not a parameter tweak (§5).
- `indexer_types` is all `"full"` in this checkpoint: every DSA layer runs
  its own indexer (the cross-layer top-k sharing machinery exists in HF but
  is unused). `index_share_for_mtp_iteration: true` matters only at M8+.

### 3.4 mHC residual stream (all 45 main layers; `Glm5NextTextHyperConnection`)

Hidden state is 4 streams (`hc_mult: 4`): embedding output is replicated 4×
at input; shape [s, 4, 4096] throughout. Two HC sites per layer (attn, ffn).
Per site, per token: flatten streams → **unweighted RMSNorm eps=1e-5,
applied BEFORE the fn matmul** (bitwise-sensitive order difference vs the
base engine, which applies rsqrt after) → `hc_*_fn` [24, 16384] BF16 →
split into pre/post/comb logits, scaled by `hc_*_scale[3]` F32 and biased
by `hc_*_base[24]` F32:
- pre  = sigmoid(·) + 1e-6 (stream collapse weights),
- post = 2·sigmoid(·) (sublayer output placement),
- comb = softmax(−1) + eps, then Sinkhorn-Knopp: column-normalize (+eps),
  then 19 × (row-normalize +eps, column-normalize +eps) — 20 iterations
  total (`hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`) → doubly stochastic 4×4.

Sublayer wiring (decoder layer): `attn_hc` collapses streams → **weighted
RMSNorm(4096, 1e-5)** (`input_layernorm`) → attention; streams ←
post·F + combᵀ·residual (note the comb transpose). Same shape around the
MLP with `ffn_hc` + `post_attention_layernorm`.

**Head collapse is an unweighted mean** over the 4 streams
(`Glm5NextTextHyperHead` — explicitly unlike DeepSeek-V4's sigmoid-gated
head), followed by the final weighted RMSNorm and the BF16 lm_head. The
MTP layer uses plain residuals (no `hc_*`).

### 3.5 MoE router + experts (layers 3–44; must be bit-faithful)

- Router (`Glm5NextTextTopkRouter`): logits in FP32
  (`moe_router_dtype: float32`), **sigmoid** scoring; F32
  `e_score_correction_bias` [288] added for **top-k selection only**
  (`noaux_tc`); weights gathered from *unbiased* sigmoid scores, normalized
  to sum 1 (`norm_topk_prob`, denominator +1e-20), × `routed_scaling_factor`
  **2.5**. Top-8 of 288.
- Group-limiting is a **no-op** (`n_group: 1`, `topk_group: 1`) — the code
  path exists but selects the single group; do not implement group masking.
- Expert MLP: SwiGLU with clamping (`swiglu_limit: 10`): gate clamped to
  max 10, up clamped to ±10, then silu(gate)·up. Routed experts: 288,
  inter 2048. Shared expert: 1, inter 2048 (always active, added to the
  routed sum).
- Layers 0–2: dense MLP, inter 12288, same clamping.
- Checkpoint anatomy (Phase A): experts are 6 tensors each (3 FP8 weights +
  3 F32 scale tensors), coalescable into one **24.0059 MiB slab**
  (**25,171,968 B** — corrected at M1 from the Phase A figure 25,190,400 B,
  which implied 16 B per F32 scale element); total 12,384 routed experts
  (43 MoE layers incl. MTP × 288) = 311,729,651,712 B ≈ **290.32 GiB**
  (corrected from "~297 GiB", which was never consistent with either
  slab figure).

### 3.6 Weight formats and precision

Normative compute path (user-confirmed 2026-08-29): **FP8-as-storage,
BF16-compute** — dequant (q × `weight_scale_inv` per 128×128 block, **F32
scales, NOT UE8M0**) → BF16 → BF16 matmul. This replaces the base engine's
fp4/UE8M0 normative paths: `c/fp4.h` is pruned at M0, `c/fp8.h` semantics
are replaced, BF16 GEMV/GEMM kernels port from the bf16 sibling repo.

| Tensors | Format |
|---|---|
| Routed experts (w1/w2/w3), dense-MLP, q_a/q_b, kv_a, o_proj (DSA), shared expert | FP8-E4M3 + F32 `weight_scale_inv` per 128×128 block |
| kv_b_proj, indexer (wq_b/wk/k_norm/weights_proj/ape/gate), router weight, KDA smalls (q/k/v/conv/f_a/f_b/g_a/g_b/b/o_proj/o_norm), mHC fn, all norms, embed, lm_head | BF16 |
| A_log, dt_bias, e_score_correction_bias, hc base/scale | F32 |

HF additionally keeps `conv1d` in its FP32-strict load list (storage BF16,
upcast at load) — see §3.2. `quantization_config`: `quant_method: fp8`,
`fmt: e4m3`, `weight_block_size: [128,128]`, `activation_scheme: dynamic`;
`modules_to_not_convert` enumerates exactly the BF16/F32 sets above.

### 3.7 Tokenizer & chat template

- HF `tokenizers`-backend BPE, vocab 154,880 (`reference/tokenizer.json`);
  M2 ports it to `c/tok.h` with exhaustive golden probing (base-project
  method: codepoint probe corpus vs HF tokenizers).
- Chat template (`reference/chat_template.jinja`, jinja-gated goldens at
  M2): prefix `[gMASK]<sop>`; turns wrapped in
  `<|system|>/<|user|>/<|assistant|>/<|observation|>`; `<think>` blocks;
  XML tool calls (`<tool_call>name<arg_key>k</arg_key><arg_value>v
  </arg_value>…</tool_call>`); a `Reasoning Effort: {Low|High|Max}` system
  prefix (`reasoning_effort` ∈ {low, high}, default max).
- Generation defaults (`generation_config.json`): temperature 1.0,
  top_p 0.95; EOS {154820, 154827, 154829}.

### 3.8 MTP layer (deferred to M8+)

Layer 45 is a classic NextN head (Phase A header anatomy): `eh_proj` BF16
[4096, 8192], enorm/hnorm, shared_head.norm, one full DSA+MoE layer,
**no hc_* (plain residuals)**, shares embed/lm_head. HF does **not**
implement it (`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", …]`);
the closest reference is SGLang `glm4_moe_nextn.py`. Deferred: it does not
affect main-model logits.

## 4. KDA ordering contract (pinned)

The HF reference has two KDA evaluation orderings that are **not
bit-identical** to each other:

- **Prefill** — `chunk_kimi_delta_attention`: chunk size 64, FP32, with
  cumulative-log-decay intra-chunk attention, a sequential UT-transform
  solve per chunk, and inter-chunk state propagation via matmuls.
- **Decode** — `recurrent_kimi_delta_attention`: one FP32 delta-rule step
  per token (§3.2 recurrence).

Both are mathematically the same recurrence, but the chunked path's FP32
operation order (cumsum, matmul reductions, the in-place triangular solve)
differs from the per-token loop, so their outputs diverge at the last bits
and can flip a sampled token downstream. This is a property of the
reference itself, not of our port.

**Contract (pinned at M0):**
1. Engine **prefill == HF chunked path**, gated bitwise against the oracle
   on prefill fixtures.
2. Engine **decode == HF recurrent step**, gated bitwise against the oracle
   on decode fixtures.
3. **No cross-phase bitwise expectation**: a sequence run as
   prefill-then-decode is not required to bit-match the same sequence run
   as one long prefill, in either the reference or the engine. Token-level
   equivalence claims (M5+) are always made within a fixed phase split.
4. The C implementation therefore carries **two separate verified
   orderings** (`kda_prefill` / `kda_decode_step`), each pinned to its own
   golden set; neither may "simplify" into the other. Prefill continuations
   (chunked prefill on top of a cached state) follow the HF
   `initial_state`-carrying chunked call and get their own fixtures.

## 5. Engine plan: ports, seam crossings, new work

**Ports unchanged from the base engine (`../Apus`, this repo's inherited `c/`):**
- DSA indexer machinery: sparse top-k gather/scatter, sparse-mask
  attention plumbing, indexer cache scaffolding (the GLM scoring variant is
  new — below).
- `c/mhc.h` mHC machinery: 4× stream state, Sinkhorn-20 (identical hc
  constants) — modulo the bitwise-sensitive norm-before-fn order (§3.4)
  and the unweighted-mean head collapse.
- FP8 block container + shard/manifest tiering (`c/st.h`,
  `apus.index.json` manifest pattern), cache hierarchy
  (`c/cache.h` ESlot/LFRU/pins/RSS guard, `c/pool.h` pthread pool),
  miss-overlap I/O, pilot prefetch scaffolding (`c/pilot.h`),
  `c/compat.h` tri-platform shims, server (`tools/server.py`), chat
  (`tools/chat.py`), sampling (`c/sample.h`).
- Weight-container design: ordinary safetensors shards + hardened pread
  index; coalesced per-expert slabs (one miss = one pread); crash-resumable
  converter driver. The GLM converter reuses this machinery with the
  glm5_next schema (below).

**Ports across the adapter seam from `../Apus-Ling-3.0-Flash-bf16`:**
- `c/kda.h` — KDA prefill/decode kernels (re-verified here against the
  glm5_next oracle; the donor targeted a different model).
- Sigmoid-router semantics (noaux_tc selection-only bias, norm_topk_prob).
- BF16 GEMV/GEMM kernels (`c/bf16.h`/`c/blas.h`) — scalar-first,
  NEON/AVX2 bitwise-pinned.
- Flagged: the donor is behind on engine work (no Windows port). Only the
  model-math pieces cross the seam; never engine infrastructure.

**New in this adapter:**
- E4M3 + F32-`weight_scale_inv` 128×128-block dequant replacing the fp4
  (MXFP4/UE8M0) and fp8-UE8M0 paths; BF16 compute kernels behind it (§3.6).
- GLM tokenizer (`c/tok.h`, vocab 154,880) and chat encoding
  (`c/encoding.h`, §3.7) — replaces the dsv4 BPE tables and DSML format.
- glm5_next converter schema (`tools/convert.py` rewrite): real tensor
  naming from the GLM checkpoint index, 6-tensor expert slabs
  (24.0059 MiB), 12,384 experts, vision tower stripped, MTP tensors in a
  separate shard group for M8+ lazy load.
- Indexer variant: kpool-4 compressed pools + always-selected tail,
  no Hadamard, no score quantization (§3.3).
- Oracle: `tools/oracle.py` port of `reference/inference/modeling_glm5_next.py`
  (pure-PyTorch fallback paths pinned, §3 note).
- GLM Metal backend (`c/backend_gmetal.mm`, M7b): BF16 GEMV/GEMM + fused
  FP8-block-dequant GEMM shaders, hooked in `c/bf16.h`/`c/gdsa.h`,
  bitwise == the pinned CPU kernels, ephemeral zero-copy wraps (§7 Metal
  tier). (The V4 `c/backend_metal.mm` was removed 2026-09-05 with the V4
  engine cleanup.)
- Pruned at M0: `c/fp4.h`, `c/dspark.h`, old `c/mtp.h` (V4-only code);
  the whole retained V4 engine was removed 2026-09-05 (user decision —
  it remains in `../Apus`).

## 6. Weight container format (glm5_next schema)

Decisions carried from the base engine:
- **Byte-identical repack** of the official checkpoint — no requantization,
  no transcode. Verification is byte-compare plus dequant spot-checks.
- Ordinary safetensors output shards + `st.h` index; manifest
  `apus.index.json` (config hash, tensor map, format version).
- **Coalesced 6-tensor expert slabs** (3 FP8 weights + 3 F32 scales,
  24.0059 MiB): a cache miss is one pread into one slab, zero-copy views.
- Shard-by-shard, disk-safe, crash-resumable conversion (fixed header
  reserve, written-set dedup) — the M1 converter of the base engine is the
  template; GLM tensor naming comes from the checkpoint's
  `model.safetensors.index.json`, never assumed.
- MTP tensors in a separate shard group (lazy-loaded at M8+); vision
  tensors stripped.

Exact schema landed at M1 (`tests/m1/README.md` is the spec): container
names = checkpoint names minus `model.language_model.` (lm_head.weight
verbatim), slab member order gate/up/down × (weight, weight_scale_inv),
MTP in an `apus-mtp-*` group, and deferral for the one split expert in
the real index (MTP expert (45,197), up_proj pair in shard 2).

## 7. Tiering design (disk → RAM → compute) — LANDED at M6

Per-token expert demand (decode, no cache hits): 42 MoE layers × 8 experts
× 24.0059 MiB ≈ **7.9 GiB/token** worst-case cold FP8 reads (+MTP when
active). This is the number every tier decision fights.

On the M1 Pro there is one 32 GB physical pool; tiers are budget classes
within it, plus disk (tunable `APUS_*` env, RSS guard is a hard requirement):
- **Resident dense+smalls set**: total checkpoint 305.78 GiB minus 290.32
  GiB routed experts ⇒ 15.46 GiB of non-expert bytes, of which ≈1 GiB is
  the stripped vision tower ⇒ ≈**14.4 GiB** (embed/head ~2.4 GiB BF16,
  KDA smalls ~8.8 GiB, DSA ~1.6 GiB, FP8 dense MLPs + shared experts with
  F32 scales, router, mHC, norms), wired. (Corrected at M1 from ≈8.8 GiB,
  which used the wrong 297 GiB expert total.)
- **KDA state**: 34 layers × FP32 [64,128,128] ≈ 136 MiB, plus conv states —
  constant in context length.
- **DSA KV + indexer cache (the M6/R7 decision, pinned)**: full-fidelity
  BF16, **no eviction, no quantization** (either changes attention outputs —
  a quality-invariant change needing explicit user approval). The caller
  sizes `kv_cap` to the session; `apus_gmodel_state_bytes(m, kv_cap)`
  returns the exact footprint; the forward fails loudly (-1) past cap.
  Layout kept from m5g: per-head K/V head-major `[H][cap*d]` (attention
  reads are per-head streams; m5g-gated), indexer rows token-major
  `[cap*ID]` (the pool rebuild reads whole rows). **Correction to the
  pre-M6 "~17 GB at 1M" estimate below**: that priced a LATENT-form cache
  (kv_lora 512 + packed indexer ≈ 1.5 KB/token/layer); the landed design
  caches the EXPANDED per-head K/V (64×256 k + 64×256 v + 2×128 indexer
  codes ≈ 64.5 KB/token/layer ≈ **0.71 MB/token** over the 11 DSA layers —
  ~23 GB at 32K tokens, ~700 GB at 1M). On the 32 GB dev machine
  full-fidelity DSA context is therefore on the order of 10–20K tokens.
  Latent-form caching (re-expand per attention, or gather only the
  indexer-selected set — bitwise-safe in principle: masked lanes
  contribute exact +0.0f) plus selected-set attention is the designed
  path to the old numbers, deferred to M7+ with its own oracle gate.
- **Expert cache (`c/gcache.h`, landed)**: slab-streaming LRU behind the
  M5 `apus_gmodel_expert()` seam (`apus_gmodel_open2` tiered mode). One
  pread per miss (the M1 coalescing invariant, instrumented), the I/O
  worker dequantizes the 6-member slab to BF16 with the m3g kernel into
  one contiguous payload `[gate | up | down]` = 3·2048·4096 codes =
  **48 MiB at real scale** — the cache budget is therefore counted in
  PAYLOAD bytes: ~10–14 GB holds ~210–290 of 12,384 experts (1.7–2.4%).
  Hit and miss return byte-identical BF16 weights (dequant is
  deterministic; tests/m6g gates eager == big-cache == 1-slot == pilot
  digests). Policy: per-layer LRU + per-forward working set with
  end-of-block promotion; generation-tagged miss overlap on a pthread
  I/O pool (`APUS_IO_THREADS`); demand/speculative job classes with
  hot-first pop (the MoE union storm overtakes pilot loads); RSS guard
  (`APUS_RSS_GUARD_MB`) drops coldest LRU payloads at layer boundaries;
  payload/staging buffer recycling (the V4 M6c mmap-churn lesson).
  Budget: `APUS_GEXPERT_CACHE_MB` (default 6144).
- **Pilot prefetch (`c/gpilot.h`, landed)**: dL=1 router lookahead — at
  layer L's post-attention hook (`ApusGmodelHooks`) the pilot computes
  layer L+1's router input with L+1's own ffn-mHC + post-norm + router
  weights (the gated gmhc/gdsa/gmoe pieces) and issues speculative hints;
  prefill unions per-token predictions. No pilot thread — the gcache I/O
  pool provides the overlap. Numerics never touched (digest equality
  gated). Measured on synthetic locality fixtures (tests/m6g; the
  random-weights constant-bias caveat applies): cluster recall 1.000,
  prefetch coverage demand 0 vs 70 without pilot. Real-weight recall is
  an M7 measurement (tools/measure_router_locality.py carries over).
- **Metal tier (M7b, landed)**: the GLM Metal backend
  (`c/backend_gmetal.mm`) offloads the BF16 GEMV/GEMM family (hooked once
  in `c/bf16.h`, catching every GLM matmul — KDA/DSA projections, router,
  experts, dense MLP, head) and the fused FP8-block-dequant GEMM (hooked
  in `c/gdsa.h`, DSA q_a/q_b/kv_a/o_proj — skips the per-call dequant
  materialization). BITWISE == the pinned CPU kernels (the shaders
  reproduce the sequential-k two-rounding order; the only measured class
  is the fp32-subnormal denormal flush, unreachable from normative data —
  tests/m7b/README.md). Buffer policy (P4): zero-copy only, two classes —
  REGISTERED model-owned dense weights wrapped ONCE at open
  (`apus_gmetal_register_region`, unregistered at close) + the M7b
  ephemeral per-op policy for gcache payloads/activations/outputs —
  nothing unregistered is held across ops, hence nothing across the
  M6 `layer_end`/RSS-guard boundary (the invariant holds by construction,
  no invalidation protocol; zero extra RSS, so the Metal tier needs no
  budget of its own against the expert cache). Floors (P4 real-scale
  tuned): `APUS_GMETAL_MIN_KB` (bf16, default OFF — every bf16 decode
  GEMV shape is a measured net loss under real-scale memory pressure;
  32768/0 re-enables), `APUS_GMETAL_FP8_MIN_KB` (default 0 — the fused
  path pays at every size, measured x2.7–4.0 at the DSA shapes),
  `APUS_GMETAL_MAX_M` (default 1 — prefill GEMMs stay CPU, the shaders
  have no m-blocking). KDA
  conv/recurrent, indexer, mHC, norms, router scoring stay CPU
  (ordering-sensitive; future per-kernel bitwise designs).

Performance model (to be validated at M7 on real weights): with NVMe at
~B GB/s on 24 MiB random reads and miss rate m, tok/s ≈ B / (7.9 GiB × m).
No tok/s target is quoted until measured.

## 8. Memory/disk situation (flagged conflict)

- Official checkpoint: **305.8 GiB**, 62 safetensors shards (FP8-E4M3).
- Converted container: ≈ same size again (~306 GiB) — the repack is
  byte-identical, so source and container coexist at ~**612 GiB**.
- Internal disk at seed time (2026-08-29): **456 GiB free** ⇒ **does not
  fit side by side**. CONFLICT FLAGGED to the user; resolution pending
  (free space, or user-approved shard cleanup after conversion, or
  download/convert directly onto an external drive with a `weights/`
  symlink per project rules).
- **Download NOT started — explicit WAIT since 2026-08-29.** Run `df -h .`
  at each milestone start; M7 is the first milestone that needs real
  weights — ask the user before downloading.

## 9. Kernel plan and numerics constraints

Carried from the base engine, unchanged:
- C11, stb-style single-TU headers (`#define APUS_<NAME>_IMPLEMENTATION`
  in exactly one TU), root `Makefile` is the only build system,
  `-Wall -Wextra` clean, no new dependencies.
- **`-ffp-contract=off` pinned** everywhere.
- Scalar reference first; NEON (primary) and AVX2 ports **bitwise-pinned**
  to it; Windows via MinGW-w64/UCRT64 shims in `c/compat.h`.
- Golden-I/O verification against the numpy oracle; bit-exactness is the
  dominant gate; **thread-count-independent digests at APUS_THREADS=1/4/8**.
- UBSan + `leaks` on macOS (ASan is broken on the dev Mac).

Kernel work items for this adapter:
- `c/fp8blk.h` (M3 ✅): E4M3 × F32-scale 128×128-block dequant → BF16;
  dequant exhaustive test (scalar == NEON == AVX2, bitwise). (Planned as
  the `c/fp8.h` rewrite; landed as a new header. Shared scalar helpers
  re-homed to `c/num.h`; the V4 `c/fp8.h`/`c/fp4.h` were removed
  2026-09-05 with the V4 engine.)
- `c/bf16.h` (M3 ✅, ported from the bf16 sibling, trimmed of the ILP/BLAS
  perf classes): BF16 GEMV (decode) / GEMM (prefill), FP32 accumulation
  order pinned to the oracle's BF16-matmul semantics (scalar == NEON ==
  AVX2 == oracle `_mm`, bitwise). `c/blas.h` (Accelerate dispatch) is a
  later perf-milestone port, gated as its own reorder class.
- `c/gkda.h` (M4b ✅, ported from the Ling bf16 sibling's `c/kda.h` and
  corrected against the GLM oracle — divergences listed in
  tests/m4h/README.md): chunked prefill + recurrent decode per §4; FP32
  state; conv-state update; fp32 l2norm and the eps-1e-5 gated o_norm
  per §3.2.
- `c/gdsa.h` (M4b ✅, fresh port — the V4 `c/attn.h` indexer variant is
  NOT reused for numerics): NoPE MLA path (simpler than base — no rotary,
  no sink, no de-rotation) + the GLM indexer variant (§3.3).
- `c/gmhc.h` / `c/gmoe.h` (M4a ✅): the §3.4 order fix (norm before fn
  matmul) + mean head collapse, and the §3.5 sigmoid router + swiglu
  experts; Sinkhorn-20 FP32 exactly as the reference. (The V4
  `c/mhc.h`/`c/moe.h` were removed 2026-09-05 with the V4 engine.)

## 10. Milestone roadmap

Each milestone has its own `tests/mX/` gate + README.

- **M0 — docs + oracle + pruning** (this file, AGENTS.md, STATUS.md;
  `tools/oracle.py` port of `modeling_glm5_next.py`; prune `c/fp4.h`,
  `c/dspark.h`, old `c/mtp.h`; pin the KDA ordering contract, §4; pull
  small reference files into `reference/` — done Phase A).
- **M1 — converter + downloader** ✅: glm5_next schema, 6-tensor expert
  slabs (25,171,968 B = 24.0059 MiB), 12,384 experts = 311,729,651,712 B
  ≈ 290.32 GiB, byte-identical, resumable, split-expert deferral.
  (Validated on synthetic fixtures + the real index while the download
  is in WAIT; gate `make test-m1`.)
- **M2 — tokenizer + chat encoding**: `c/tok.h` BPE vocab 154,880;
  `c/encoding.h` GLM template, jinja-gated goldens.
- **M3 — kernels** ✅: `c/num.h` (scalar helpers re-homed from `c/fp4.h`),
  `c/fp8blk.h` (E4M3+F32-scale block dequant), `c/bf16.h` (BF16 GEMV/GEMM);
  scalar-first, NEON/AVX2 bitwise-pinned (tests/m3g).
- **M4 — sublayers vs oracle** ✅: mHC, MoE router+experts (M4a,
  tests/m4g), KDA, DSA+indexer (M4b, tests/m4h).
- **M5 — full forward** ✅: `c/gmodel.h` (config parser, M1 v2 container
  loader with one-pread-per-expert slabs, state + scratch arena, prefill
  + decode per §4), synthetic weights, bitwise vs oracle (tests/m5g).
  `apus_gmodel_expert()` is the M6 cache seam.
- **M6 — tiering/cache** ✅: `c/gcache.h` (expert slab streaming behind
  `apus_gmodel_expert()`, one pread per miss, dequant-on-fill to 48 MiB
  BF16 payloads, LRU + working set + I/O pool + RSS guard), `c/gpilot.h`
  (dL=1 router-lookahead prefetch), `apus_gmodel_open2` tiered wiring
  (bitwise-neutral), `apus_gmodel_state_bytes` + the pinned long-context
  KV policy (§7). Gate: `make test-m6g` (tests/m6g).
- **M7a — serving** ✅: `c/apus.c` GLM-only (the container manifest's
  `model_type: "glm5_next"` required; anything else is a hard error),
  `bin/apus run/serve` end to end on synthetic
  containers (tiered streaming + pilot via `apus_gmodel_open2`),
  `tools/server.py` (OpenAI-compatible HTTP/SSE gateway, GLM schema:
  `clear_thinking`/`reasoning_effort`, tools sibling, `<tool_call>`
  parsing, 400s on encoding failures) + `tools/chat.py`. Gate:
  `make test-m7a` (tests/m7a — scripted GLM parrot containers, bitwise
  C == oracle token streams, eager == tiered+pilot end to end).
- **M7b — GLM Metal** ✅ (`c/backend_gmetal.mm` + `c/backend_gmetal.h`;
  the V4 `c/backend_metal.mm` was removed 2026-09-05): BF16 GEMV/GEMM + fused
  FP8-block-dequant GEMM shaders, BITWISE == the pinned CPU kernels
  (measured fp32-subnormal denormal-flush class documented in
  tests/m7b/README.md), ephemeral zero-copy wraps (no pointer cache; the
  M6 layer_end invariant by construction), `--metal`/APUS_METAL=1 enables
  the backend in `bin/apus_metal`. Gate: `make test-m7b` (tests/m7b —
  the GLM test_gkernels /
  test_gmodel bitwise gates and the m7a server suite on the Metal
  binary). Dead V4-fixture suites removed (tests/m4b/m4c/m5/m6a/m6b/
  m9c/m9d — roles covered by m4g/m4h/m5g/m6g/m7a). **Leftover: the
  real-model smoke — first task needing weights; ask the user before
  downloading.**
- **M8+ — MTP** (classic NextN, SGLang `glm4_moe_nextn.py` reference),
  speculative decoding, perf.

## 11. Risks / open questions

- R1: **Disk conflict** (§8): 456 GiB free vs ~612 GiB needed. Blocks the
  download; resolution is a user decision.
- R2: **Golden-host for real-weight verification**: the HF reference at
  305.8 GiB (FP8; ~600+ GiB dequantized to BF16) cannot run on the dev
  Mac. M5 uses synthetic-weight fixtures vs the oracle (sufficient for
  bitwise gating); real-model smoke at M7 needs the actual download and,
  for token-level comparison, either a rented big-GPU host running the HF
  reference or acceptance of fixture-only evidence. Decision due by M7.
- R3: **HF kernel dispatch** (§3 note): with `fla`/hub kernels installed,
  HF silently runs Triton kernels instead of the in-file fallbacks. The
  oracle must pin the fallback path and record its environment; otherwise
  goldens are not reproducible across machines.
- R4: **conv1d dtype nuance** (§3.2): checkpoint stores conv weights BF16;
  HF's FP32-strict load list upcasts them. The oracle pins one behavior;
  the C side must match it bitwise. Resolve at M4 (KDA sublayer gate).
- R5: **MTP has no HF reference** — SGLang `glm4_moe_nextn.py` is the
  closest implementation. Deferred with the feature (M8+); re-derive from
  checkpoint headers then.
- R6: **Donor drift**: `../Apus-Ling-3.0-Flash-bf16` is behind on engine
  work. Model-math ports only; any missing engine feature is reimplemented
  here, not back-ported.
- R7: **DSA KV cache at long context** (§7): RESOLVED at M6 as designed —
  full-fidelity BF16, no eviction, no quantization, loud failure past a
  caller-sized kv_cap (`apus_gmodel_state_bytes` for budgeting). The
  expanded per-head layout costs ~0.71 MB/token over the 11 DSA layers
  (the pre-M6 "~17 GB at 1M" estimate priced a latent-form cache);
  latent-form caching + selected-set attention is the designed path to
  longer contexts, deferred to M7+ with its own oracle gate.
