"""Synthetic glm5_next (GLM-5.3-Flash) checkpoint fixtures for M1 tests.

Mimics the real checkpoint's naming scheme and dtypes at tiny scale
(verified against reference/model.safetensors.index.json by
test_5_index_realism.py). Checkpoint names carry the real
"model.language_model." / "model.visual." prefixes; the converter strips
the language prefix, keeps "lm_head.weight" verbatim, and drops vision.

Layout (5 main layers + MTP, mirroring the real structure):

  layers 0-2: KDA self_attn + dense MLP (FP8 + F32 weight_scale_inv)
  layer 3:    DSA self_attn (MLA + indexer) + MoE
  layer 4:    KDA self_attn + MoE
  layer 5:    MTP — DSA + MoE + eh_proj/enorm/hnorm/shared_head.norm,
              NO hc_* (plain residuals)
  model.visual.*: 6 vision tensors (stripped at conversion, counted)

FP8 payloads contain only VALID E4M3 codes (0x7F/0xFF NaN codes excluded
— the converter refuses them; test_3 checks that path with a dedicated
corrupt shard).

Split experts (mirroring the real index, which splits MTP expert
(45,197) across input shards 1/2):

  expert (3, 1): up_proj pair in shard 1, the rest in shard 2
  expert (5, 2): down_proj pair in the shard before the MTP shard

Dims (hidden H=64) are chosen so the FP8 block-scale invariants hold
exactly like the real model (scale shape == ceil(O/128) x ceil(K/128),
F32); the dense MLP inter of 160 exercises non-trivial ceil shapes
([2,1]/[1,2] scale blocks).

Per expert: 3 x (8*64) FP8 + 3 x (1*1) F32 scale = 1536 + 12 = 1548 B.
"""

import json
import os

import numpy as np

import stutil

H = 64              # hidden size
VOCAB = 96          # tiny stand-in for 154,880
HC = 4              # hc_mult (real: 4) -> hc_fn [24, 4*H]
N_LAYERS_MAIN = 5   # main layers 0..4
MTP_LAYER = 5       # real: 45
KDA_LAYERS = (0, 1, 2, 4)     # real: 34 layers
DSA_LAYERS = (3,)             # real: 3,7,...,43
MOE_LAYERS = (3, 4)           # real: 3..44
DENSE_LAYERS = (0, 1, 2)      # real: 0..2 (first_k_dense_replace 3)

# KDA (real: 64 heads x 128, conv k=4, low-rank 128)
LIN_HEADS = 2
LIN_DIM = 8
QKV = LIN_HEADS * LIN_DIM     # 16
CONV_K = 4

# DSA (real: q_lora 1536, kv_lora 512, 64 heads x 256; indexer 32 x 128)
Q_LORA = 8
KV_LORA = 4
N_HEADS = 2
QD = 8                  # qk_nope_head_dim
VD = 8                  # v_head_dim
I_HEADS = 2
I_DIM = 4
KPOOL = 2               # real: 4

# MoE (real: 288 experts, inter 2048; shared 1 x 2048; dense inter 12288)
N_EXPERTS = 4
MOE_INTER = 8
DENSE_INTER = 160       # exercises ceil(160/128) = 2 scale blocks

N_VISION = 6

SPLIT_MAIN = (3, 1)     # expert whose up_proj pair moves to shard 1
SPLIT_MTP = (5, 2)      # expert whose down_proj pair moves one shard earlier

SLAB_ORDER = ["gate_proj.weight", "gate_proj.weight_scale_inv",
              "up_proj.weight", "up_proj.weight_scale_inv",
              "down_proj.weight", "down_proj.weight_scale_inv"]

PER_EXPERT_BYTES = 3 * (MOE_INTER * H) + 3 * 4  # 1548
assert PER_EXPERT_BYTES == 1548

# Real-checkpoint of-record sizes (pinned in tests/m1/README.md):
#   slab = 3 * 2048*4096 (FP8) + 3 * 16*32*4 (F32 scales)
#        = 25,171,968 B = 24.0059 MiB
#   12,384 experts (43 MoE layers x 288) = 311,729,651,712 B ~ 290.32 GiB
REAL_SLAB_BYTES = 3 * 2048 * 4096 + 3 * (2048 // 128) * (4096 // 128) * 4
assert REAL_SLAB_BYTES == 25171968

LANG = "model.language_model."
VIS = "model.visual."


def _rand(rng, dtype, shape):
    n = stutil.tensor_nbytes(dtype, shape)
    if dtype == "F8_E4M3":
        # valid E4M3 codes only: 0x7F and 0xFF (NaN) excluded
        vals = rng.randint(0, 254, size=n).astype(np.uint8)
        vals[vals >= 0x7F] += 1   # 0x7F..0xFE -> 0x80..0xFF... remap below
        vals[vals == 0xFF] = 0xFE  # keep 0xFF (NaN) out
        return vals.tobytes()
    return rng.randint(0, 256, size=n, dtype=np.uint8).tobytes()


def _fp8_pair(prefix, rows, cols):
    return [
        (f"{prefix}.weight", "F8_E4M3", [rows, cols]),
        (f"{prefix}.weight_scale_inv", "F32",
         [(rows + 127) // 128, (cols + 127) // 128]),
    ]


def _kda_attn(pre):
    t = []
    for p in ("q", "k", "v"):
        t.append((f"{pre}.self_attn.{p}_proj.weight", "BF16", [QKV, H]))
        t.append((f"{pre}.self_attn.{p}_conv1d.weight", "BF16",
                  [QKV, 1, CONV_K]))
    t.append((f"{pre}.self_attn.f_a_proj.weight", "BF16", [LIN_DIM, H]))
    t.append((f"{pre}.self_attn.f_b_proj.weight", "BF16", [QKV, LIN_DIM]))
    t.append((f"{pre}.self_attn.g_a_proj.weight", "BF16", [LIN_DIM, H]))
    t.append((f"{pre}.self_attn.g_b_proj.weight", "BF16", [QKV, LIN_DIM]))
    t.append((f"{pre}.self_attn.A_log", "F32", [LIN_HEADS]))
    t.append((f"{pre}.self_attn.dt_bias", "F32", [QKV]))
    t.append((f"{pre}.self_attn.b_proj.weight", "BF16", [LIN_HEADS, H]))
    t.append((f"{pre}.self_attn.o_norm.weight", "BF16", [LIN_DIM]))
    t.append((f"{pre}.self_attn.o_proj.weight", "BF16", [H, QKV]))
    return t


def _dsa_attn(pre):
    t = []
    t += _fp8_pair(f"{pre}.self_attn.q_a_proj", Q_LORA, H)
    t.append((f"{pre}.self_attn.q_a_layernorm.weight", "BF16", [Q_LORA]))
    t += _fp8_pair(f"{pre}.self_attn.q_b_proj", N_HEADS * QD, Q_LORA)
    t += _fp8_pair(f"{pre}.self_attn.kv_a_proj_with_mqa", KV_LORA, H)
    t.append((f"{pre}.self_attn.kv_a_layernorm.weight", "BF16", [KV_LORA]))
    t.append((f"{pre}.self_attn.kv_b_proj.weight", "BF16",
              [N_HEADS * (QD + VD), KV_LORA]))
    t += _fp8_pair(f"{pre}.self_attn.o_proj", H, N_HEADS * VD)
    idx = f"{pre}.self_attn.indexer"
    t.append((f"{idx}.wq_b.weight", "BF16", [I_HEADS * I_DIM, Q_LORA]))
    t.append((f"{idx}.wk.weight", "BF16", [I_DIM, H]))
    t.append((f"{idx}.k_norm.weight", "BF16", [I_DIM]))
    t.append((f"{idx}.k_norm.bias", "BF16", [I_DIM]))
    t.append((f"{idx}.weights_proj.weight", "BF16", [I_HEADS, H]))
    t.append((f"{idx}.index_kpool_compress_ape", "BF16", [KPOOL, I_DIM]))
    t.append((f"{idx}.index_kpool_compress_gate", "BF16", [I_DIM, H]))
    return t


def _expert(pre, skip_suffixes=()):
    t = []
    for suffix in SLAB_ORDER:
        if suffix in skip_suffixes:
            continue
        rows, cols = (H, MOE_INTER) if suffix.startswith("down_proj") \
            else (MOE_INTER, H)
        if suffix.endswith(".weight"):
            t.append((f"{pre}.{suffix}", "F8_E4M3", [rows, cols]))
        else:
            t.append((f"{pre}.{suffix}", "F32",
                      [(rows + 127) // 128, (cols + 127) // 128]))
    return t


def _moe_mlp(pre, skip_experts=()):
    """MoE tensors. skip_experts maps expert id -> suffixes held back for
    the split-shard placement."""
    t = []
    t.append((f"{pre}.mlp.gate.weight", "BF16", [N_EXPERTS, H]))
    t.append((f"{pre}.mlp.gate.e_score_correction_bias", "F32",
              [N_EXPERTS]))
    for e in range(N_EXPERTS):
        t += _expert(f"{pre}.mlp.experts.{e}",
                     skip_suffixes=skip_experts.get(e, ()))
    for p, rows, cols in (("gate_proj", MOE_INTER, H),
                          ("up_proj", MOE_INTER, H),
                          ("down_proj", H, MOE_INTER)):
        t += _fp8_pair(f"{pre}.mlp.shared_experts.{p}", rows, cols)
    return t


def _dense_mlp(pre):
    t = []
    for p, rows, cols in (("gate_proj", DENSE_INTER, H),
                          ("up_proj", DENSE_INTER, H),
                          ("down_proj", H, DENSE_INTER)):
        t += _fp8_pair(f"{pre}.mlp.{p}", rows, cols)
    return t


def _hc(pre):
    mix = (2 + HC) * HC   # 24, like the real model
    t = []
    for sub in ("attn", "ffn"):
        t.append((f"{pre}.hc_{sub}_fn", "BF16", [mix, HC * H]))
        t.append((f"{pre}.hc_{sub}_base", "F32", [mix]))
        t.append((f"{pre}.hc_{sub}_scale", "F32", [3]))
    return t


def _layer_specs(layer):
    """All tensors of one main-model layer, as (name, dtype, shape)."""
    pre = f"{LANG}layers.{layer}"
    t = [(f"{pre}.input_layernorm.weight", "BF16", [H]),
         (f"{pre}.post_attention_layernorm.weight", "BF16", [H])]
    t += _kda_attn(pre) if layer in KDA_LAYERS else _dsa_attn(pre)
    if layer in MOE_LAYERS:
        skip = {}
        if layer == SPLIT_MAIN[0]:
            skip[SPLIT_MAIN[1]] = ("up_proj.weight",
                                   "up_proj.weight_scale_inv")
        t += _moe_mlp(pre, skip_experts=skip)
    else:
        t += _dense_mlp(pre)
    t += _hc(pre)
    return t


def _mtp_specs():
    pre = f"{LANG}layers.{MTP_LAYER}"
    t = [(f"{pre}.input_layernorm.weight", "BF16", [H]),
         (f"{pre}.post_attention_layernorm.weight", "BF16", [H])]
    t += _dsa_attn(pre)
    t += _moe_mlp(pre, skip_experts={
        SPLIT_MTP[1]: ("down_proj.weight", "down_proj.weight_scale_inv")})
    t.append((f"{pre}.eh_proj.weight", "BF16", [H, 2 * H]))
    t.append((f"{pre}.enorm.weight", "BF16", [H]))
    t.append((f"{pre}.hnorm.weight", "BF16", [H]))
    t.append((f"{pre}.shared_head.norm.weight", "BF16", [H]))
    return t


def split_main_specs():
    """The up_proj pair of the split main-layer expert (placed in an
    earlier shard, mirroring the real checkpoint's split expert)."""
    pre = f"{LANG}layers.{SPLIT_MAIN[0]}.mlp.experts.{SPLIT_MAIN[1]}"
    return [(f"{pre}.up_proj.weight", "F8_E4M3", [MOE_INTER, H]),
            (f"{pre}.up_proj.weight_scale_inv", "F32", [1, 1])]


def split_mtp_specs():
    pre = f"{LANG}layers.{SPLIT_MTP[0]}.mlp.experts.{SPLIT_MTP[1]}"
    return [(f"{pre}.down_proj.weight", "F8_E4M3", [H, MOE_INTER]),
            (f"{pre}.down_proj.weight_scale_inv", "F32", [1, 1])]


def _global_specs():
    return [
        (f"{LANG}embed_tokens.weight", "BF16", [VOCAB, H]),
        (f"{LANG}norm.weight", "BF16", [H]),
        ("lm_head.weight", "BF16", [VOCAB, H]),
    ]


def _vision_specs():
    return [
        (f"{VIS}patch_embed.proj.weight", "BF16", [8, 12]),
        (f"{VIS}patch_embed.proj.bias", "BF16", [8]),
        (f"{VIS}blocks.0.attn.qkv.weight", "BF16", [24, 8]),
        (f"{VIS}blocks.0.attn.qkv.bias", "BF16", [24]),
        (f"{VIS}blocks.0.mlp.gate_proj.weight", "BF16", [16, 8]),
        (f"{VIS}post_layernorm.weight", "BF16", [8]),
    ]


def make_fixture_tree(root, n_shards=3, seed=1234):
    """Write a fake glm5_next checkpoint under root/. Returns shard names.

    Layout mimics the real one: globals + vision (+ the split main
    expert's up_proj pair) in shard 1, main layers spread over the middle
    shards (with the split MTP expert's down_proj pair in the last of
    them), MTP last. Two experts are split across input shards, exactly
    like MTP expert (45,197) in the real index.
    """
    rng = np.random.RandomState(seed)
    assert n_shards >= 3
    groups = [[] for _ in range(n_shards)]
    groups[0] += _global_specs() + _vision_specs() + split_main_specs()
    for layer in range(N_LAYERS_MAIN):
        groups[1 + (layer % (n_shards - 2))] += _layer_specs(layer)
    groups[n_shards - 2] += split_mtp_specs()
    groups[-1] += _mtp_specs()

    weight_map = {}
    names = []
    for i, specs in enumerate(groups):
        fname = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        tensors = [(name, dtype, shape, _rand(rng, dtype, shape))
                   for name, dtype, shape in specs]
        stutil.write_shard(os.path.join(root, fname), tensors)
        for name, _, _, _ in tensors:
            weight_map[name] = fname
        names.append(fname)

    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        total = sum(
            stutil.tensor_nbytes(d, s)
            for specs in groups for _, d, s in specs
        )
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "hidden_size": H,
                "num_hidden_layers": N_LAYERS_MAIN,
                "num_nextn_predict_layers": 1,
                "moe_intermediate_size": MOE_INTER,
                "intermediate_size": DENSE_INTER,
                "n_routed_experts": N_EXPERTS,
                "first_k_dense_replace": 3,
            },
        }, f)
    # Small support files (the download driver copies them next to the
    # container; contents are arbitrary).
    with open(os.path.join(root, "tokenizer.json"), "w") as f:
        json.dump({"version": "fixture"}, f)
    with open(os.path.join(root, "tokenizer_config.json"), "w") as f:
        json.dump({"fixture": True}, f)
    with open(os.path.join(root, "generation_config.json"), "w") as f:
        json.dump({"temperature": 1.0}, f)
    with open(os.path.join(root, "chat_template.jinja"), "w") as f:
        f.write("[gMASK]<sop>{% for m in messages %}{{ m['content'] }}"
                "{% endfor %}")
    return names
