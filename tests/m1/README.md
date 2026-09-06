# tests/m1 — converter + downloader (M1, glm5_next schema)

Dependency: the project venv at `../../.venv` (numpy; safetensors used for
read cross-checks only). No pytest — plain `unittest`. Fully offline: all
shards are synthesized locally; nothing downloads weights.

Run all:

```sh
../../.venv/bin/python -m unittest discover -s . -v
```

or individually, e.g.:

```sh
../../.venv/bin/python -m unittest test_3_schema -v
```

## Container schema (the spec the C loader will consume at M4–M6)

Output = ordinary safetensors shards + manifest + standard index:

- `apus-NNNNN.safetensors` — main group: layers 0–44 + embed/norm/lm_head.
- `apus-mtp-NNNNN.safetensors` — MTP group: layers.45.* (deferred NextN
  head; separate group so M8+ can lazy-load it).
- `apus.index.json` — manifest: format_version 2, model_type glm5_next,
  config_hash, header_reserve, n_main_layers, expert_slab_members,
  shard_groups, tensor_map (name -> shard, absolute file offset, nbytes,
  dtype, shape), expert_slabs (block, expert, shard, offset, nbytes),
  stripped (vision tensor/byte counts).
- `model.safetensors.index.json` — standard index over the output (the
  M0 oracle's ShardSet and the C loader read this).
- `apus.convert.state.json` — internal crash-resume state (not part of
  the container contract).

Naming: checkpoint names minus the `model.language_model.` prefix
(`model.language_model.layers.3.self_attn.q_a_proj.weight` →
`layers.3.self_attn.q_a_proj.weight`); `lm_head.weight` verbatim
(top-level in the real checkpoint, untied). This is exactly the M0
oracle's convention, so M4/M5 cross-load the container directly.
`model.visual.*` (347 tensors in the real index) is STRIPPED — text-only
scope — and counted in the manifest.

Expert slabs: the 6 tensors of each routed expert are contiguous and
adjacent in one output shard, in the pinned order
`gate_proj.weight, gate_proj.weight_scale_inv, up_proj.weight,
up_proj.weight_scale_inv, down_proj.weight, down_proj.weight_scale_inv`
(one pread per expert fetch). Payloads are byte-identical repacks — no
requantization. FP8-E4M3 payloads are scanned for the NaN codes
0x7F/0xFF and refused (they never occur in a trained checkpoint; M0
flagged they never occur synthetically — verify on real shards at M7).

Crash-resumability: 16 MiB fixed header reserve per output shard,
atomic state flush after every tensor append / seal / slab update,
verify-before-trust on resume (sealed shards: size + header hash; open
shard: torn tail truncated), written-set dedup. Output is a
deterministic function of the input set: interrupted+resumed runs and
shard-by-shard driver runs are byte-identical to one-shot runs.

Split experts: the real index splits exactly ONE expert across input
shards — MTP expert (45,197): its up_proj pair is in
model-00002-of-00062, the rest in model-00001-of-00062. The converter
defers incomplete slabs (state `pending_slabs`) and appends them once
every member's shard is converted; resolution order is deterministic.
The download driver HELD-deletes: a source shard referenced by a pending
slab is deleted only after the slab resolves and all its members are
byte-verified (state `unverified_resolved`).

## Of-record sizes (real checkpoint; pinned, verified against
## reference/model.safetensors.index.json + config.json at 2026-08-30)

- Shards: 62; tensors: 76,108 (75,761 text + 347 vision);
  total_size 328,326,771,576 B = **305.78 GiB**.
- FP8-E4M3 + F32 `weight_scale_inv`, 128×128 blocks
  (`weight_block_size: [128,128]`, 37,338 pairs, never split across
  shards).
- **Per-expert slab: 25,171,968 B = 24.0059 MiB**
  = 3 × 2048×4096 B (gate/up/down FP8) + 3 × 16×32×4 B (F32 scales).
- Routed experts: 12,384 = 43 MoE layers (3–44 + MTP 45) × 288.
- **Experts total: 311,729,651,712 B = 290.32 GiB.**
- Dense+vision remainder: 16,597,119,864 B ≈ 15.46 GiB.

DEVIATION from Phase A (STATUS 2026-08-29 / ARCHITECTURE §3.5): those
quote slab ≈ 25,190,400 B (24.02 MiB) and experts ≈ 297 GiB. The
config-derived exact numbers above are authoritative; Phase A's slab
implies 16 B per F32 scale element (the 18,432 B delta is exactly
3 × 512 × 16 B), i.e. a Phase A arithmetic slip, and 297 GiB was never
consistent with either slab (12,384 × 25,190,400 B = 290.53 GiB). The
converter validates scale dtype/shape pairing (F32, ceil(O/128) ×
ceil(K/128)) on every tensor at conversion time, so if the real shard
headers ever disagree with the 128×128 rule the conversion fails loudly
— re-check at M7 when real shards exist.

Layer inventory (index-verified): 34 KDA layers (all but 3,7,…,43), 11
DSA layers (3,7,…,43) + MTP layer 45 (full DSA+MoE, no hc_*); MoE on
3–44 + 45; dense MLP (FP8, inter 12288) on 0–2; hc_* on 0–44 only.

## What each file does

- `stutil.py` — minimal *manual* safetensors reader/writer (8-byte LE
  header length + JSON header + raw data). The safetensors library is
  deliberately not used for writing: numpy has no F8_E4M3/BF16 dtypes
  and the fixtures must be raw bytes we fully control.
- `fixtures.py` — synthetic glm5_next checkpoint at tiny scale (H=64,
  5 main layers + MTP: 3 dense KDA, 1 DSA MoE, 1 KDA MoE; 4 experts;
  6 vision tensors; FP8 with VALID E4M3 codes only, F32 scales with
  ceil-block shapes; per-expert slab 1,548 B). Mirrors the real split
  expert TWICE (main layer 3 + MTP layer 5) to exercise deferral in both
  shard groups; 3 input shards with a weight_map index + config.json +
  support files.
- `test_1_byte_identity.py` — every kept output tensor's bytes equal the
  source bytes; dtype/shape preserved; naming matches the oracle
  convention; vision stripped + counted; MTP in its own group; manifest
  consistent; safetensors-lib parse cross-check; std index for the
  oracle loader.
- `test_2_coalescing.py` — each expert's 6 tensors consecutive in the
  shard header in the pinned order, contiguous+adjacent in the data
  region, single-shard (split experts reassembled), manifest slab
  records match the actual header offsets.
- `test_3_schema.py` — converter validation (FP8/scale pairing dtype +
  ceil-shape, orphans, non-FP8 scales, missing pairs, E4M3 NaN 0x7F/0xFF
  refusal, name whitelist, MTP rules, vision strip, flat-dir incomplete
  expert) + the E4M3/F32-scale dequant reference for the M3 kernel
  (landmarks, subnormals, monotonicity, hand-computed block dequant).
- `test_4_resume.py` — crash mid-conversion, torn-write tail, crash at
  seal boundary, crash mid-slab-resolution, no-op rerun: final output
  always byte-identical to the uninterrupted run.
- `test_5_index_realism.py` — validates the converter's assumptions
  against the REAL `reference/model.safetensors.index.json`: prefixes,
  whitelist coverage, 6-tensor expert groups, FP8 pairs never split,
  exactly-one split expert (45,197), layer inventories, of-record size
  arithmetic.
- `test_6_download_driver.py` — offline (local "remote" dir) end-to-end
  download.py run: clean run == one-shot reference; split-shard held
  until resolution then deleted; kill mid-download and mid-conversion,
  restart, partial `.part` resume; support files land next to the
  container.

download.py network mode is untested here (no weights may be
downloaded); offline/--source-dir mode exercises the same state machine.
