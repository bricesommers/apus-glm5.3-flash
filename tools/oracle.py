#!/usr/bin/env python3
"""tools/oracle.py — M0 golden-reference oracle for GLM-5.3-Flash (glm5_next).

A numpy-only port of the HuggingFace reference implementation, the ONLY
normative source: reference/inference/modeling_glm5_next.py (auto-generated
from modular_glm5_next.py; cited below as glm5:N). Scope: TEXT model only
(vision tower excluded). The main model is the 45 layers; the MTP layer
(layers.45, classic NextN) is implemented at M8a (mtp_forward/mtp_chain
below) following the SGLang deepseek_nextn.py forward — HF drops layer 45
at load, so there is no HF reference for it. model_forward returns the
final mHC stream `h` alongside logits; mtp_hnorm_input() collapses it to
the single hidden the MTP hnorm consumes (the M8a empirical pin, see
tools/mtp_pin.py).

Two modes, exactly as the inherited base oracle:

  * "f64" — the same algorithm with all arithmetic in float64 (no bf16
    rounding). The numerical "truth".
  * "f32" — dtype-faithful mode: bf16 rounding applied at every point the
    reference casts to bf16, fp32 where the reference is fp32 (KDA state,
    forget gate, router, mHC maps, indexer scores). This mode is the golden
    target the C engine is verified against.

FP8 weight storage is ALGORITHMIC and applied in both modes (like the base
oracle's QAT simulation): the normative compute path (user-pinned) is
  dequant: W = E4M3(code) * weight_scale_inv per 128x128 block (F32 scales,
  one F32 multiply, rounded)  ->  round to BF16 (RNE)  ->  BF16 matmul with
  FP32 accumulate  ->  BF16 output.
bf16 rounding is round-to-nearest-even via the float32 bit trick
(bf16_round below); the C side MUST replicate it exactly.

KDA ORDERING CONTRACT (user-pinned at M0):
  The HF reference runs KDA prefill through the CHUNKED kernel
  (chunk_kimi_delta_attention, glm5:483-579, chunk_size 64) and single-token
  decode through the RECURRENT kernel (recurrent_kimi_delta_attention,
  glm5:428-479). Both compute the same math in fp32 but with different
  summation orderings, so they are NOT bit-identical to each other. This
  oracle exposes them separately — prefill() always uses the chunked path,
  decode_step() always uses the recurrent path — and the C engine is gated
  bitwise per phase: prefill vs chunked, decode vs recurrent. Cross-phase
  agreement is only mathematical (see tests/m0/README.md).

Fixture weight naming mirrors the real checkpoint (reference/
model.safetensors.index.json) with the "model.language_model." prefix
dropped, e.g. "layers.3.self_attn.q_a_proj.weight" +
".weight_scale_inv" (F32), "layers.0.self_attn.q_conv1d.weight" (BF16).
Synthetic fixture dims keep every FP8 matrix a multiple of 128 in both
dims (the real 128x128 block grid is NOT shrunk for fixtures).

This file is self-contained on purpose: the safetensors IO and the E4M3
codec are inlined (adapted from tests/m1/stutil.py and tests/m3
gen_golden.py) because the V4-era test dirs are pruned at M0 — do NOT
reintroduce cross-test imports here.
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASTER_SEED = 20260830

# ---------------------------------------------------------------------------
# Config: load the real text_config, with overrides for tiny synthetic
# fixtures. glm5_next facts live in reference/config.json -> "text_config".
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(ROOT, "reference", "config.json")


def load_text_config(path=CONFIG_PATH, **overrides):
    """Load reference/config.json, return the flattened oracle cfg dict for
    the TEXT model. Overrides are applied last; layer pattern lists are
    regenerated when num_hidden_layers / first_k_dense_replace change (see
    _layer_patterns)."""
    with open(path) as f:
        tc = json.load(f)["text_config"]
    la = tc["linear_attn_config"]
    cfg = {
        "hidden_size": tc["hidden_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "hidden_act": tc["hidden_act"],               # "silu"
        # mHC
        "hc_mult": tc["hc_mult"],
        "hc_sinkhorn_iters": tc["hc_sinkhorn_iters"],
        "hc_eps": tc["hc_eps"],
        # MLP / MoE
        "first_k_dense_replace": tc["first_k_dense_replace"],
        "intermediate_size": tc["intermediate_size"],
        "moe_intermediate_size": tc["moe_intermediate_size"],
        "n_routed_experts": tc["n_routed_experts"],
        "n_shared_experts": tc["n_shared_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "routed_scaling_factor": tc["routed_scaling_factor"],
        "norm_topk_prob": tc["norm_topk_prob"],
        "n_group": tc["n_group"],
        "topk_group": tc["topk_group"],
        "swiglu_limit": tc["swiglu_limit"],
        # MLA (pure NoPE: qk_rope_head_dim == 0, no RoPE anywhere)
        "num_attention_heads": tc["num_attention_heads"],
        "q_lora_rank": tc["q_lora_rank"],
        "kv_lora_rank": tc["kv_lora_rank"],
        "qk_rope_head_dim": tc["qk_rope_head_dim"],
        "qk_nope_head_dim": tc["qk_nope_head_dim"],
        "v_head_dim": tc["v_head_dim"],
        # Lightning indexer
        "index_n_heads": tc["index_n_heads"],
        "index_head_dim": tc["index_head_dim"],
        "index_topk": tc["index_topk"],
        "index_kpool": tc["index_kpool"],
        "index_kpool_always_select_tail":
            tc["index_kpool_always_select_tail"],
        # KDA
        "linear_num_heads": la["num_heads"],
        "linear_head_dim": la["head_dim"],
        "linear_conv_kernel_dim": la["short_conv_kernel_size"],
        "linear_lower_bound": la["gate_lower_bound"],   # -5.0 -> safe path
        # layer patterns (explicit lists in the real config)
        "layer_types": list(tc["layer_types"]),
        "mlp_layer_types": list(tc["mlp_layer_types"]),
        "indexer_types": list(tc["indexer_types"]),
    }
    assert cfg["qk_rope_head_dim"] == 0, "oracle is pure-NoPE only"
    assert cfg["linear_lower_bound"] is not None, \
        "oracle implements the safe-gate sigmoid path only"
    cfg.update(overrides)
    _layer_patterns(cfg)
    return cfg


def _layer_patterns(cfg):
    """(Re)generate layer_types / mlp_layer_types / indexer_types for
    cfg["num_hidden_layers"]. Rule (verified to reproduce the real config's
    explicit 45-layer lists EXACTLY): DSA at layers 3,7,11,... (i % 4 == 3),
    KDA elsewhere; dense MLP below first_k_dense_replace, sparse after.
    "shared" indexer layers are not implemented (the real config is all
    "full")."""
    L = cfg["num_hidden_layers"]
    if len(cfg["layer_types"]) != L:
        cfg["layer_types"] = [
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
            for i in range(L)]
    if len(cfg["mlp_layer_types"]) != L:
        cfg["mlp_layer_types"] = [
            "dense" if i < cfg["first_k_dense_replace"] else "sparse"
            for i in range(L)]
    cfg["indexer_types"] = ["full"] * L
    for lt in cfg["layer_types"]:
        assert lt in ("linear_attention", "deepseek_sparse_attention"), lt


def make_tiny_config(**overrides):
    """Tiny fixture config: real glm5_next values with small dims. Every FP8
    matrix stays a multiple of 128 in both dims (the real block grid)."""
    tiny = {
        "hidden_size": 256,
        "vocab_size": 384,
        "num_hidden_layers": 5,      # KDA 0,1,2,4; DSA 3; dense 0-2; MoE 3,4
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "n_routed_experts": 8,
        "num_experts_per_tok": 3,
        "num_attention_heads": 2,
        "q_lora_rank": 128,
        "kv_lora_rank": 128,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "index_n_heads": 2,
        "index_head_dim": 128,
        "index_topk": 8,             # select_k = 2 pools; width 8 + 3 tail
        "linear_num_heads": 2,
        "linear_head_dim": 128,
    }
    tiny.update(overrides)
    return load_text_config(**tiny)


# ---------------------------------------------------------------------------
# bf16 helpers (numpy has no bf16; values are carried as f32 with a rounded
# mantissa, stored on disk as raw uint16). PINNED semantics — the C side
# replicates bf16_round bit for bit (round-to-nearest-even, ties to even
# mantissa, via the 0x7FFF + LSB bias trick).
# ---------------------------------------------------------------------------


def bf16_round(x):
    """Round-to-nearest-even to bf16, returned as float32 holding a bf16 value."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    u = (u + bias) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def f32_to_bf16_bytes(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()


def bf16_bytes_to_f32(b):
    return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
            << np.uint32(16)).view(np.float32).copy()


def _dt(f64):
    return np.float64 if f64 else np.float32


def _B(x, f64):
    """bf16 rounding boundary: applied in f32-faithful mode only."""
    return x if f64 else bf16_round(x)


def _mm(a, b):
    """Cross-platform DETERMINISTIC matmul (inherited convention, M12b).

    The reduction runs in the INPUT dtype with a FIXED sequential k-order
    using only broadcast elementwise ops — per output element each step is a
    single IEEE-exact multiply or add, identical on every platform — so
    fixtures regenerate bitwise-identically everywhere. Accumulating in fp32
    (f32-faithful mode) keeps the oracle in the same precision class as the
    C kernels; the tolerance budget then covers only summation-ORDER
    differences (sequential oracle vs the kernels' own orders).
    """
    dt = np.result_type(a, b)
    a_dt = np.asarray(a, dtype=dt)
    b_dt = np.asarray(b, dtype=dt)
    acc = np.zeros(a_dt.shape[:-1] + b_dt.shape[1:], dtype=dt)
    for k in range(a_dt.shape[-1]):
        acc += a_dt[..., k:k + 1] * b_dt[k]
    return acc


def _bmm(a, b):
    """Batched _mm over the leading axis: a [H,M,K] @ b [H,K,N] -> [H,M,N]."""
    return np.stack([_mm(a[h], b[h]) for h in range(a.shape[0])])


# ---------------------------------------------------------------------------
# FNV-1a 64 digests (inherited convention; the C side mirrors this).
# ---------------------------------------------------------------------------


def fnv1a(chunks):
    """FNV-1a 64 over a list of byte chunks."""
    h = 14695981039346656037
    for c in chunks:
        for b in c:
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def array_digest(arr):
    return fnv1a([np.ascontiguousarray(arr).tobytes()])


# ---------------------------------------------------------------------------
# FP8-E4M3 codec with F32 weight_scale_inv per 128x128 block (inlined; the
# base's UE8M0 path does NOT apply here — glm5_next scales are plain F32).
# ---------------------------------------------------------------------------


def _e4m3_table():
    """All 256 E4M3 codes -> f32. exp bias 7; exp=15 mant=7 is NaN."""
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s = -1.0 if (c & 0x80) else 1.0
        e = (c >> 3) & 0x0F
        m = c & 0x07
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[c] = s * v
    return t


E4M3_TABLE = _e4m3_table()


def e4m3_quant_rne(y):
    """Round-to-nearest-even quantize f32 array y (already clamped to +-448)
    to E4M3 codes. Ties pick the even code index. NaN codes 0x7F/0xFF are
    never produced."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    sign = np.signbit(y)
    a = np.abs(y)
    pos = E4M3_TABLE[:0x7F]                       # 0x00..0x7E (0x7F is NaN)
    idx = np.clip(np.searchsorted(pos, a), 0, len(pos) - 1)
    lo = np.clip(idx - 1, 0, len(pos) - 1)
    hi = idx
    dlo = a - pos[lo]
    dhi = pos[hi] - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & ((hi & 1) == 0))
    codes = np.where(pick_hi, hi, lo).astype(np.uint8)
    return np.where(sign, codes | np.uint8(0x80), codes)


def fp8_store(W):
    """Quantize raw f32 [O, K] (O,K multiples of 128) to FP8-E4M3 codes +
    F32 scale_inv per 128x128 block: scale = amax/448 (floor 2**-126 for
    zero blocks), plain F32 — NOT pow2/UE8M0. Returns
    (codes_bytes, scale_bytes, code_shape, scale_shape)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    O, K = W.shape
    assert O % 128 == 0 and K % 128 == 0, (O, K)
    blocks = W.reshape(O // 128, 128, K // 128, 128).transpose(0, 2, 1, 3)
    amax = np.maximum(np.abs(blocks).max(axis=(2, 3)), np.float32(2.0 ** -126))
    scale = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
    y = np.clip(blocks / scale[:, :, None, None], -448.0, 448.0)
    codes = e4m3_quant_rne(y.astype(np.float32))
    codes = codes.transpose(0, 2, 1, 3).reshape(O, K)
    return (codes.tobytes(), scale.tobytes(), (O, K),
            (O // 128, K // 128))


def fp8_dequant(codes, scales, f64):
    """codes u8 [O,K], scales f32 [O/128,K/128] -> dequantized matrix in the
    mode dtype. f32 mode: ONE f32 multiply per element (q * scale_inv),
    matching the normative dequant step."""
    dt = _dt(f64)
    w = E4M3_TABLE[codes].astype(dt)
    s = scales.astype(dt)
    return w * np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)


# ---------------------------------------------------------------------------
# Linear layers. The normative FP8 path (user-pinned): dequant -> BF16 ->
# BF16 matmul (fp32 accumulate) -> BF16 out. BF16 weights are used directly.
# ---------------------------------------------------------------------------


def fp8_linear(x, w_codes, w_scales, f64):
    """FP8-E4M3 blockwise 128x128 dense linear (weight-only quant)."""
    dt = _dt(f64)
    w = _B(fp8_dequant(w_codes, w_scales, f64), f64)   # dequant -> bf16
    y = _mm(_B(x, f64).astype(dt), w.astype(dt).T)
    return _B(y, f64)


def bf16_linear(x, w, f64):
    """Plain bf16 matmul: fp32-accumulate, bf16 out (torch semantics)."""
    dt = _dt(f64)
    y = _mm(_B(x, f64).astype(dt), w.astype(dt).T)
    return _B(y, f64)


def f32_linear(x, w, f64):
    """fp32 matmul, no output rounding (router, mHC fn: bf16-valued weights
    but the reference casts both operands to fp32)."""
    dt = _dt(f64)
    return _mm(x.astype(dt), w.astype(dt).T)


# ---------------------------------------------------------------------------
# Norms, activations.
# ---------------------------------------------------------------------------


def rms_norm(x, w, eps, f64):
    """Weighted RMSNorm (glm5:65-83): fp32 internal, cast to bf16, THEN
    multiply by the weight — two bf16 roundings, in the reference's order."""
    dt = _dt(f64)
    xf = x.astype(dt)
    var = (xf * xf).mean(-1, keepdims=True)
    t = _B(xf * (1.0 / np.sqrt(var + eps)), f64)       # .to(input_dtype)
    return _B(w.astype(dt) * t, f64)                   # weight * t


def unweighted_rms_norm(x, eps, f64):
    """Glm5NextTextUnweightedRMSNorm (glm5:210-216): mHC input norm, fp32
    in/out (input is already .float()), NO weight, NO bf16 rounding."""
    dt = _dt(f64)
    xf = x.astype(dt)
    var = (xf * xf).mean(-1, keepdims=True)
    return xf * (1.0 / np.sqrt(var + eps))


def rms_norm_gated(x, w, gate, eps, f64):
    """RMSNormGated (glm5:339-359): strict fp32 (norm, weight, sigmoid gate),
    one bf16 rounding at the very end."""
    dt = _dt(f64)
    xf = x.astype(dt)
    var = (xf * xf).mean(-1, keepdims=True)
    y = xf * (1.0 / np.sqrt(var + eps))
    y = w.astype(dt) * y
    y = y * sigmoid(gate.astype(dt))
    return _B(y, f64)


def layer_norm(x, w, b, eps, f64):
    """nn.LayerNorm (indexer k_norm, glm5:764): fp32 internal (biased var),
    weight + bias, one bf16 rounding at the output."""
    dt = _dt(f64)
    xf = x.astype(dt)
    mu = xf.mean(-1, keepdims=True)
    var = ((xf - mu) ** 2).mean(-1, keepdims=True)
    y = (xf - mu) * (1.0 / np.sqrt(var + eps))
    y = y * w.astype(dt) + b.astype(dt)
    return _B(y, f64)


def l2norm(x, eps, f64):
    """FLA-aligned l2norm (glm5:417-425): x / sqrt(sum(x^2) + eps), fp32."""
    dt = _dt(f64)
    xf = x.astype(dt)
    inv = np.sqrt((xf * xf).sum(axis=-1, keepdims=True) + np.asarray(eps, dtype=dt))
    return xf / inv


def sigmoid(x):
    """Numerically stable sigmoid in the input dtype."""
    x = np.asarray(x)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu(x):
    return x * sigmoid(x)


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def topk_stable(row, k):
    """Descending top-k; ties broken to the LOWER index (pinned — torch.topk
    tie order is unspecified; matches the base oracle's ambiguity A6)."""
    k = min(k, row.shape[-1])
    return np.argsort(-row, kind="stable", axis=-1)[..., :k]


# ---------------------------------------------------------------------------
# mHC (glm5:219-295 + layer wiring 1317-1328). The maps (pre/post/comb) are
# fp32 end-to-end; the norm-before-fn is the UNWEIGHTED RMSNorm
# (eps = rms_norm_eps) applied BEFORE the fn matmul — the bitwise-sensitive
# order difference vs the base engine's mhc (which applied rsqrt AFTER).
# ---------------------------------------------------------------------------


def hc_split_sinkhorn(mixes, scale, base, hc, iters, eps, f64):
    """pre/post/comb split + gates + Sinkhorn (glm5:279-290). Same recipe
    family as the base engine: row softmax + per-element eps, col-norm with
    eps in the denominator, then (iters-1) x (row-norm, col-norm)."""
    dt = _dt(f64)
    mixes = mixes.astype(dt)
    pre = sigmoid(mixes[..., :hc] * dt(scale[0]) + base[:hc].astype(dt)) + eps
    post = 2.0 * sigmoid(mixes[..., hc:2 * hc] * dt(scale[1])
                         + base[hc:2 * hc].astype(dt))
    comb = (mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc)
            * dt(scale[2]) + base[2 * hc:].astype(dt).reshape(hc, hc))
    comb = softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre(h, fn, scale, base, cfg, f64, interm=None, tag=""):
    """HyperConnection.forward (glm5:267-295). h [s, hc, dim]; returns
    (collapsed [s, dim] bf16, post, comb)."""
    dt = _dt(f64)
    s = h.shape[0]
    hc = cfg["hc_mult"]
    flat = unweighted_rms_norm(h.reshape(s, hc * cfg["hidden_size"]),
                               cfg["rms_norm_eps"], f64)          # 278
    mixes = f32_linear(flat, fn, f64)                             # 279 (fp32)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc,
                                        cfg["hc_sinkhorn_iters"],
                                        cfg["hc_eps"], f64)
    if interm is not None and tag:
        interm[tag + "_pre"] = pre.copy()
        interm[tag + "_post"] = post.copy()
        interm[tag + "_comb"] = comb.copy()
    # collapse: fp32 weighted sum over the stream axis, bf16 out (294)
    y = (pre[..., None] * h.astype(dt)).sum(axis=1)
    return _B(y, f64), post, comb


def hc_post(x, residual, post, comb, f64):
    """Decoder-layer mHC expand (glm5:1317-1319, 1326-1328): post and comb
    are CAST TO BF16 first, the comb@residual matmul is a bf16 matmul (fp32
    accumulate), the final add rounds to bf16. x [s, dim], residual
    [s, hc, dim] -> y [s, hc, dim]."""
    dt = _dt(f64)
    p = _B(post, f64)                                  # post.to(dtype)
    c = _B(comb, f64)                                  # comb.to(dtype)
    t1 = _B(p[..., None].astype(dt) * x[:, None, :].astype(dt), f64)
    t2 = _B(_bmm(np.swapaxes(c, -1, -2).astype(dt),
                 residual.astype(dt)), f64)            # comb^T @ residual
    return _B(t1 + t2, f64)


# ---------------------------------------------------------------------------
# KDA — Kimi Delta Attention (glm5:305-734). Conv + gates are bf16-valued,
# the delta-rule core is fp32 (f64 in truth mode). BOTH paths are
# implemented exactly as the reference does them:
#   prefill -> chunk_kimi_delta_attention (chunk 64)
#   decode  -> recurrent_kimi_delta_attention
# ---------------------------------------------------------------------------


def kda_conv_prefill(mixed, conv_w, state_in, f64):
    """causal_conv1d_fn (glm5:394-414) over the concatenated conv state
    (update_conv_state, glm5:670-672). mixed [s, 3qkv] bf16 values; conv_w
    [3qkv, k]; state_in [k-1, 3qkv] or None (zeros). Conv + silu in fp32,
    bf16 out. Returns (out [s, 3qkv], new_state [k-1, 3qkv])."""
    dt = _dt(f64)
    s, cd = mixed.shape
    k = conv_w.shape[-1]
    prev = (np.zeros((k - 1, cd), dtype=dt) if state_in is None
            else state_in.astype(dt))
    ext = np.concatenate([prev, mixed.astype(dt)], axis=0)   # [k-1+s, cd]
    out = np.zeros((s, cd), dtype=dt)
    for i in range(k):                                       # fixed tap order
        out += ext[i:i + s] * conv_w[:, i].astype(dt)[None, :]
    out = _B(silu(out), f64)
    new_state = _B(ext[-(k - 1):], f64)      # conv state holds bf16 values
    return out, new_state


def kda_conv_decode(mixed1, conv_w, state, f64):
    """causal_conv1d_update (glm5:375-391): one token against the [k-1, cd]
    state. Returns (out [1, cd], new_state)."""
    dt = _dt(f64)
    k = conv_w.shape[-1]
    ext = np.concatenate([state.astype(dt), mixed1.astype(dt)], axis=0)
    out = np.zeros((1, ext.shape[1]), dtype=dt)
    for i in range(k):
        out += ext[i:i + 1] * conv_w[:, i].astype(dt)[None, :]
    return _B(silu(out), f64), _B(ext[1:], f64)


def kda_forget_gate(x, P, f64):
    """Glm5NextTextForgetGate (glm5:305-335), safe-gate path: f_a/f_b are
    bf16 linears; g, A, sigmoid in fp32; returns fp32 g in (lower_bound, 0].
    x [s, dim] -> g [s, H, D]."""
    dt = _dt(f64)
    H, D = P["heads"], P["head_dim"]
    fg = bf16_linear(bf16_linear(x, P["f_a"], f64), P["f_b"], f64)   # 322
    g = fg.astype(dt) + P["dt_bias"].astype(dt)                      # 323
    decay = np.exp(P["A_log"].astype(dt))                            # 325
    g = g.reshape(-1, H, D)
    return P["lower_bound"] * sigmoid(decay[None, :, None] * g)      # 329


def _triple_masked(qc, kc, w, dt):
    """out[i,j] = sum_d q[i,d]*k[j,d]*w[i,j,d] with the reference's per-d
    rounding order ((q*k) * w) and sequential-d accumulation. qc [H,N,C,D],
    kc [H,N,C,D], w [H,N,C,C,D] (the decay mask is PER-CHANNEL, glm5:533)."""
    H, N, C, D = qc.shape
    acc = np.zeros((H, N, C, C), dtype=dt)
    # inf/NaN products can only land at masked (j >= i) positions — see
    # kda_chunk — so the warnings are suppressed here as well.
    with np.errstate(over="ignore", invalid="ignore"):
        for d in range(D):
            acc += (qc[..., d][..., None] * kc[..., d][..., None, :]) * w[..., d]
    return acc


def kda_chunk(q, k, v, g, beta, state_in, f64):
    """chunk_kimi_delta_attention (glm5:483-579), batch squeezed out,
    chunk_size 64. q,k,v [s,H,D] bf16 values; g [s,H,D] fp32; beta [s,H]
    bf16 values; state_in [H,D,D] or None. Returns (out [s,H,D] bf16,
    state [H,D,D] fp32). All core math in the mode dtype (fp32 faithful).

    Padding note (verified benign): g is cumsum'ed WITHIN each chunk, so
    zero-padded tail rows inherit the last real row's cumulative decay; the
    state update's g[..., -1] is therefore correct even for partial chunks.
    """
    dt = _dt(f64)
    C = 64
    s, H, D = q.shape
    Dv = v.shape[-1]
    # to [H, s, D], fp32 (498-500); l2norm AFTER the cast (504-505)
    qh = l2norm(np.swapaxes(q, 0, 1).astype(dt), 1e-6, f64)
    kh = l2norm(np.swapaxes(k, 0, 1).astype(dt), 1e-6, f64)
    vh = np.swapaxes(v, 0, 1).astype(dt)
    gh = np.swapaxes(g, 0, 1).astype(dt)
    bh = np.swapaxes(beta, 0, 1).astype(dt)
    scale = dt(D ** -0.5)                                            # 510
    pad = (C - s % C) % C                                            # 511
    sp = s + pad

    def zpad(a):
        return np.pad(a, ((0, 0), (0, pad), (0, 0)))

    qp = zpad(qh) * scale                                            # 515
    kp, vp, gp = zpad(kh), zpad(vh), zpad(gh)                        # 516-518
    bp = np.pad(bh, ((0, 0), (0, pad)))                              # 519
    v_beta = vp * bp[..., None]                                      # 520
    k_beta = kp * bp[..., None]                                      # 521
    N = sp // C

    def chunks(a):
        return a.reshape(H, N, C, a.shape[-1])

    qc, kc = chunks(qp), chunks(kp)
    gc = chunks(gp).cumsum(axis=2)                                   # 531
    kbc, vbc = chunks(k_beta), chunks(v_beta)
    # decay_mask [H,N,C,C,D]: exp(g_i - g_j) (533). For j > i the exponent
    # is POSITIVE (g is a cumsum of negative decays) and exp overflows —
    # exactly as in the reference, where those entries are then REPLACED by
    # masked_fill (not multiplied out). The overflow/invalid products can
    # only occur at masked positions (valid j < i have exponent <= 0), so
    # they never contaminate the result; suppress the numpy warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        decay = np.exp(gc[:, :, :, None, :] - gc[:, :, None, :, :])
    # attn[i,j] = -sum_d k_beta[i,d]*key[j,d]*decay[i,j]; keep j < i (534)
    attn = -_triple_masked(kbc, kc, decay, dt)
    attn = np.where(np.triu(np.ones((C, C), bool), 0)[None, None],
                    dt(0.0), attn)
    # forward substitution: attn[i,:i] += attn[i,:i] @ attn[:i,:i] (535-538)
    for i in range(1, C):
        row = attn[:, :, i, :i].copy()
        sub = attn[:, :, :i, :i].copy()
        attn[:, :, i, :i] = row + (row[..., None] * sub).sum(axis=-2)
    attn = attn + np.eye(C, dtype=dt)[None, None]                    # 540
    value = _bmm(attn.reshape(H * N, C, C),
                 vbc.reshape(H * N, C, Dv)).reshape(H, N, C, Dv)     # 541
    k_cumdecay = _bmm(attn.reshape(H * N, C, C),
                      (kbc * np.exp(gc)).reshape(H * N, C, D)
                      ).reshape(H, N, C, D)                          # 542

    S = (np.zeros((H, D, Dv), dtype=dt) if state_in is None
         else state_in.astype(dt).copy())                            # 544-548
    out = np.zeros((H, N, C, Dv), dtype=dt)
    keep_lower = np.triu(np.ones((C, C), bool), 1)[None, None]       # 551
    for i in range(N):                                               # 552
        q_i, k_i, g_i = qc[:, i], kc[:, i], gc[:, i]
        # inter chunk (559): (q * exp(g)) @ S
        attn_inter = _bmm(q_i * np.exp(g_i), S)
        # intra chunk (561): sum_d q[i,d]*k[j,d]*decay[i,j], keep j < i
        attn_intra = _triple_masked(q_i[:, None], k_i[:, None],
                                    decay[:, i][:, None], dt)[:, 0]
        attn_intra = np.where(keep_lower[0], dt(0.0), attn_intra)
        v_prime = _bmm(k_cumdecay[:, i], S)                          # 563
        v_new = value[:, i] - v_prime                                # 564
        out[:, i] = attn_inter + _bmm(attn_intra, v_new)             # 566
        # state update (567-570)
        g_last = g_i[:, -1]                                          # [H, D]
        S = S * np.exp(g_last)[..., None] + _bmm(
            (k_i * np.exp(g_last[:, None] - g_i)).transpose(0, 2, 1),
            v_new)

    out = out.reshape(H, sp, Dv)[:, :s]                              # 575-576
    return _B(np.swapaxes(out, 0, 1), f64), S


def kda_recurrent(q, k, v, g, beta, state_in, f64):
    """recurrent_kimi_delta_attention (glm5:428-479), batch squeezed out.
    q,k,v [s,H,D] bf16 values; g [s,H,D] fp32; beta [s,H]; state_in [H,D,D]
    or None. Returns (out [s,H,D] bf16, state [H,D,D] fp32)."""
    dt = _dt(f64)
    s, H, D = q.shape
    Dv = v.shape[-1]
    qh = l2norm(q.astype(dt), 1e-6, f64) * dt(D ** -0.5)             # 445-453
    kh = l2norm(k.astype(dt), 1e-6, f64)
    vh = v.astype(dt)
    gh = g.astype(dt)
    bh = beta.astype(dt)
    S = (np.zeros((H, D, Dv), dtype=dt) if state_in is None
         else state_in.astype(dt).copy())                            # 458-462
    out = np.zeros((s, H, Dv), dtype=dt)
    for i in range(s):                                               # 465
        g_i = np.exp(gh[i])[..., None]                               # 469
        b_i = bh[i][..., None]                                       # 470
        S = S * g_i                                                  # 472
        kv_mem = (S * kh[i][..., None]).sum(axis=-2)                 # 473
        delta = (vh[i] - kv_mem) * b_i                               # 474
        S = S + kh[i][..., None] * delta[..., None, :]               # 476
        out[i] = (S * qh[i][..., None]).sum(axis=-2)                 # 477
    return _B(out, f64), S


def kda_forward(P, x, st, f64, decode, interm=None):
    """Glm5NextTextLinearAttention.forward (glm5:628-734), batch squeezed.
    x [s, dim] bf16; st is the LayerState (conv_state, rec_state mutated).
    decode=True takes the single-token recurrent path (the KDA ORDERING
    CONTRACT — see module docstring)."""
    dt = _dt(f64)
    s = x.shape[0]
    H, D, qkv = P["heads"], P["head_dim"], P["qkv_dim"]
    # q/k/v projections + concat (643-650)
    mixed = np.concatenate([bf16_linear(x, P["q_w"], f64),
                            bf16_linear(x, P["k_w"], f64),
                            bf16_linear(x, P["v_w"], f64)], axis=-1)
    if decode:                                                       # 659-666
        assert s == 1
        mixed, st.conv_state = kda_conv_decode(mixed, P["conv_w"],
                                               st.conv_state, f64)
    else:                                                            # 667-684
        mixed, st.conv_state = kda_conv_prefill(mixed, P["conv_w"],
                                                st.conv_state, f64)
    q, k, v = np.split(mixed, 3, axis=-1)                            # 686-690
    if interm is not None:
        interm["kda_mixed"] = mixed.copy()          # post-conv [s, 3*qkv]
    q = q.reshape(s, H, D)
    k = k.reshape(s, H, D)
    v = v.reshape(s, H, D)
    g = kda_forget_gate(x, P, f64)                                   # 697
    beta = _B(sigmoid(bf16_linear(x, P["b_w"], f64).astype(dt)), f64)  # 698
    if decode:                                                       # 701-712
        core, st.rec_state = kda_recurrent(q, k, v, g, beta,
                                           st.rec_state, f64)
    else:                                                            # 713-724
        core, st.rec_state = kda_chunk(q, k, v, g, beta,
                                       st.rec_state, f64)
    if interm is not None:
        interm["kda_core"] = core.copy()
        interm["kda_g"] = g.copy()
        interm["kda_beta"] = beta.copy()
    gate = bf16_linear(bf16_linear(x, P["g_a"], f64),
                       P["g_b"], f64).reshape(s, H, D)               # 730
    out = rms_norm_gated(core, P["o_norm"], gate,
                         P["eps"], f64).reshape(s, qkv)              # 731
    return bf16_linear(out, P["o_w"], f64)                           # 732


# ---------------------------------------------------------------------------
# DSA — MLA pure-NoPE + Lightning indexer (glm5:737-1257). q_a/q_b/kv_a/o
# are FP8; kv_b and all indexer weights are BF16. No RoPE anywhere.
# ---------------------------------------------------------------------------


def indexer_forward(P, x, q_resid, st, f64, interm=None):
    """Glm5NextTextIndexer.forward (glm5:774-878), batch squeezed, no padding
    (valid_channel is all-ones). Appends this call's tokens to the indexer
    cache in st, rebuilds ALL k-pools from the cached states (the reference
    re-pools the whole cache every forward), and returns topk indices
    [s, index_topk + kpool - 1] (-1 = invalid)."""
    dt = _dt(f64)
    s = x.shape[0]
    IH, ID = P["idx_heads"], P["idx_dim"]
    kpool = P["idx_kpool"]
    q = bf16_linear(q_resid, P["idx_wq_b"], f64).reshape(s, IH, ID)  # 798
    k_new = layer_norm(bf16_linear(x, P["idx_wk"], f64),
                       P["idx_knorm_w"], P["idx_knorm_b"], 1e-6, f64)  # 799
    gate_new = bf16_linear(x, P["idx_gate"], f64)                    # 801
    if interm is not None:
        interm["idx_k_new"] = k_new.copy()
        interm["idx_gate_new"] = gate_new.copy()
    st.idx_k = (k_new.astype(np.float32) if st.idx_k is None
                else np.concatenate([st.idx_k, k_new.astype(np.float32)]))
    st.idx_gate = (gate_new.astype(np.float32) if st.idx_gate is None
                   else np.concatenate([st.idx_gate,
                                        gate_new.astype(np.float32)]))
    n = st.idx_k.shape[0]
    kv_pos = np.arange(n)
    q_pos = n - s + np.arange(s)                                     # 895

    # --- get_pooled_states (900-973): full pools only, from token 0 ---
    np_all = (n + kpool - 1) // kpool
    n_full = n // kpool                        # pools with all 4 tokens valid
    if n_full > 0:
        gk = st.idx_gate[:n_full * kpool].reshape(n_full, kpool, ID).astype(dt)
        kk = st.idx_k[:n_full * kpool].reshape(n_full, kpool, ID).astype(dt)
        logits = gk + P["idx_ape"].astype(dt)[None]                  # 963
        probs = _B(softmax(logits, axis=1), f64)                     # 965-967
        pool_keys = _B((probs * kk).sum(axis=1), f64)                # 968
    else:
        pool_keys = np.zeros((0, ID), dtype=dt)

    # --- scores (826-831): fp32 matmuls, NO bf16 rounding ---
    if n_full > 0:
        sc = np.stack([_mm(q[:, h].astype(dt), pool_keys.T.astype(dt))
                       for h in range(IH)], axis=1)                  # 826
        sc = np.maximum(sc * dt(P["idx_scale"]), dt(0.0))            # 827
        w = bf16_linear(x, P["idx_wproj"], f64).astype(dt) \
            * dt(IH ** -0.5)                                         # 830
        index_scores = np.zeros((s, n_full), dtype=dt)
        for h in range(IH):                                          # 831
            index_scores += w[:, h:h + 1] * sc[:, h]
    else:
        index_scores = np.zeros((s, 0), dtype=dt)

    # --- candidate validity (834-845): pool's LAST token must be visible ---
    pool_end = np.arange(n_full) * kpool + (kpool - 1)
    valid_cand = pool_end[None, :] <= q_pos[:, None]                 # causal
    scores_m = np.where(valid_cand, index_scores,
                        np.finfo(np.float32).min.astype(dt))         # 842
    select_k = min(P["idx_topk"] // kpool, n_full)                   # 848
    if select_k > 0:
        sel = topk_stable(scores_m, select_k)                        # 853
        sel_valid = np.take_along_axis(valid_cand, sel, axis=1)      # 856
        sel_pools = np.where(sel_valid, sel, -1)
        topk = np.repeat(sel_pools, kpool, axis=1)                   # 861
        offs = np.tile(np.arange(kpool), (s, select_k))
        topk = np.where(topk >= 0, topk * kpool + offs, -1)
    else:
        topk = np.full((s, 0), -1, dtype=np.int64)

    # --- always-selected tail: the current incomplete pool (868, 975-1025)
    if P["idx_tail"]:
        visible_count = q_pos + 1                                    # 1008
        tail_count = visible_count % kpool                           # 1009
        tail_start = visible_count - tail_count                      # 1012
        offs = np.arange(kpool - 1)
        tail = tail_start[:, None] + offs[None]                      # 1013
        tail = np.where(offs[None] < tail_count[:, None], tail, -1)  # 1016-1023
        topk = np.concatenate([topk, tail], axis=1)                  # 1025

    # pad to index_topk + (kpool - 1), truncate (873-875)
    width = P["idx_topk"] + (kpool - 1 if P["idx_tail"] else 0)
    if topk.shape[1] < width:
        topk = np.concatenate([topk, np.full((s, width - topk.shape[1]),
                                             -1, dtype=np.int64)], axis=1)
    topk = topk[:, :width].astype(np.int32)
    if interm is not None:
        interm["idx_scores"] = index_scores.copy()
        interm["idx_topk"] = topk.copy()
    return topk


def dsa_attention(q, k_cache, v_cache, topk, scale, f64, interm=None):
    """eager_attention_forward (glm5:1040-1062) with the topk mask
    (1219-1257) over the FULL cached kv (the reference's additive-mask
    semantics — duplicated/invalid indices are excluded, everything not
    selected gets finfo.min). q [s,H,Dq] bf16, k/v [n,H,D] bf16."""
    dt = _dt(f64)
    s, H, Dq = q.shape
    n = k_cache.shape[0]
    # boolean visibility per query over the full kv (1233-1247)
    vis = np.zeros((s, n), dtype=bool)
    for t in range(s):
        ids = topk[t]
        ids = ids[(ids >= 0) & (ids < n)]
        vis[t, ids] = True
    neg = np.finfo(np.float32).min.astype(dt)                        # 1254-1256
    out = np.zeros((s, H, v_cache.shape[-1]), dtype=dt)
    probs = np.zeros((s, H, n), dtype=dt)
    for h in range(H):
        sc = _B(_mm(_B(q[:, h], f64).astype(dt),
                    _B(k_cache[:, h], f64).astype(dt).T), f64)       # 1053
        sc = _B(sc * dt(scale), f64)                                 # 1053
        sc = sc + np.where(vis, dt(0.0), neg)                        # 1055
        p = _B(softmax(sc.astype(dt), axis=-1), f64)                 # 1057
        probs[:, h] = p
        out[:, h] = _B(_mm(p.astype(dt),
                           _B(v_cache[:, h], f64).astype(dt)), f64)  # 1059
    if interm is not None:
        interm["attn_probs"] = probs
    return out


def dsa_forward(P, x, st, f64, interm=None):
    """Glm5NextTextAttention.forward (glm5:1156-1217), batch squeezed, all
    indexer_types "full". x [s, dim] bf16; st carries the expanded kv cache
    + indexer cache. Returns [s, dim]."""
    s = x.shape[0]
    H = P["heads"]
    qd, vd = P["qk_head_dim"], P["v_head_dim"]
    q_resid = rms_norm(fp8_linear(x, *P["q_a"], f64),
                       P["q_a_norm"], P["eps"], f64)                 # 1168
    q = fp8_linear(q_resid, *P["q_b"], f64).reshape(s, H, qd)        # 1169
    kv = fp8_linear(x, *P["kv_a"], f64)                              # 1171
    k_pass = rms_norm(kv, P["kv_a_norm"], P["eps"], f64)             # 1173
    kvb = bf16_linear(k_pass, P["kv_b"], f64)                        # 1146
    # glm5:1147 (expand_kv): view [s, H, qd+vd] then split k/v PER HEAD
    # ([..., :qd] / [..., qd:]) — interleaved [k|v] per head, NOT
    # head-major [all k | all v] (the pre-anchor layout; caught by
    # tests/m0/check_vs_hf.py, 2026-09-05 — gate-7 divergence fix).
    kvb = kvb.reshape(s, H, qd + vd)
    k_new = kvb[..., :qd]                                            # 1147
    v_new = kvb[..., qd:]
    if interm is not None:
        interm["dsa_q_resid"] = q_resid.copy()
        interm["dsa_q"] = q.copy()
        interm["dsa_k_new"] = k_new.copy()
        interm["dsa_v_new"] = v_new.copy()
    if st.k_cache is None:                                           # 1179-1180
        st.k_cache = k_new.astype(np.float32)
        st.v_cache = v_new.astype(np.float32)
    else:
        st.k_cache = np.concatenate([st.k_cache, k_new.astype(np.float32)])
        st.v_cache = np.concatenate([st.v_cache, v_new.astype(np.float32)])
    topk = indexer_forward(P, x, q_resid, st, f64, interm)           # 1182-1188
    o = dsa_attention(q, st.k_cache, st.v_cache, topk,
                      P["attn_scale"], f64, interm)                  # 1200-1213
    if interm is not None:
        interm["dsa_attn"] = o.copy()
    return fp8_linear(o.reshape(s, H * vd), *P["o_proj"], f64)       # 1215-1216


# ---------------------------------------------------------------------------
# MLP / MoE (glm5:86-207). SwiGLU with swiglu_limit clamps: gate clamped
# ABOVE only, up clamped both sides. FP8 weights everywhere.
# ---------------------------------------------------------------------------


def swiglu_mlp(x, gate_w, up_w, down_w, limit, f64):
    """Glm5NextTextMLP.forward (glm5:98-104) / one expert (120-142)."""
    dt = _dt(f64)
    g = fp8_linear(x, *gate_w, f64).astype(dt)
    u = fp8_linear(x, *up_w, f64).astype(dt)
    g = np.minimum(g, limit)                                         # 102
    u = np.clip(u, -limit, limit)                                    # 103
    h = _B(silu(g), f64)
    h = _B(h * _B(u, f64), f64)
    return fp8_linear(h, *down_w, f64)                               # 104


def router_forward(P, x, f64, interm=None):
    """Glm5NextTextTopkRouter (glm5:158-183): fp32 logits, sigmoid scores,
    bias for SELECTION only (noaux_tc), group-limiting a no-op at n_group=1,
    top-k over biased scores, weights from UNBIASED scores, norm + scale."""
    dt = _dt(f64)
    logits = f32_linear(x, P["gate_w"], f64)                         # 160
    scores = sigmoid(logits)                                         # 161
    biased = scores + P["gate_bias"].astype(dt)                      # 162
    # n_group=1 / topk_group=1: group selection keeps everything (163-176)
    if P["n_group"] != 1 or P["topk_group"] != 1:
        raise NotImplementedError("group-limited routing not implemented")
    idx = topk_stable(biased, P["top_k"])                            # 177
    w = np.take_along_axis(scores, idx, axis=1)                      # 178
    if P["norm_topk_prob"]:
        w = w / (w.sum(axis=-1, keepdims=True) + dt(1e-20))          # 180-181
    w = w * dt(P["route_scale"])                                     # 182
    if interm is not None:
        interm["router_scores"] = scores.copy()
        interm["router_scores_biased"] = biased.copy()
        interm["router_idx"] = idx.astype(np.int32)
        interm["router_w"] = w.copy()
    return w, idx


def moe_forward(P, x, f64, interm=None):
    """Glm5NextTextMoE.forward (glm5:200-207) with the eager experts loop
    (120-135): per-expert contribution bf16-rounded, accumulated in
    ASCENDING EXPERT INDEX order with a bf16 rounding per add; shared expert
    added last with one more bf16 rounding."""
    dt = _dt(f64)
    w, idx = router_forward(P, x, f64, interm)
    y = np.zeros_like(x, dtype=dt)
    y = _B(y, f64)
    for e in range(P["n_routed"]):                                   # 127
        tok, slot = np.where(idx == e)
        if tok.size == 0:
            continue
        down = swiglu_mlp(x[tok], *P["experts"][e], P["limit"], f64)
        contrib = _B(down.astype(dt) * w[tok, slot, None].astype(dt), f64)
        y[tok] = _B(y[tok] + contrib, f64)                           # 134
    if interm is not None:
        interm["moe_routed"] = y.copy()
    shared = swiglu_mlp(x, *P["shared"], P["limit"], f64)            # 206
    if interm is not None:
        interm["moe_shared"] = shared.copy()
    y = _B(y + shared.astype(dt), f64)
    if interm is not None:
        interm["moe_out"] = y.copy()
    return y


# ---------------------------------------------------------------------------
# Decoder layer + full model (glm5:1260-1330, 1432-1495).
# ---------------------------------------------------------------------------


def block_forward(P, cfg, h, st, f64, decode, interm=None):
    """Glm5NextTextDecoderLayer.forward (glm5:1281-1330). h [s, hc, dim];
    returns the same. st is the layer's LayerState (mutated)."""
    if interm is None:
        interm = {}
    residual = h
    x, post, comb = hc_pre(h, P["hc_attn_fn"], P["hc_attn_scale"],
                           P["hc_attn_base"], cfg, f64, interm, "attn_hc")
    x = rms_norm(x, P["input_norm"], cfg["rms_norm_eps"], f64)       # 1297
    if P["kind"] == "kda":
        x = kda_forward(P, x, st, f64, decode, interm)               # 1300
    else:
        x = dsa_forward(P, x, st, f64, interm)                       # 1307
    h = hc_post(x, residual, post, comb, f64)                        # 1317-1319
    interm["post_attn_h"] = h.copy()

    residual = h
    x, post, comb = hc_pre(h, P["hc_ffn_fn"], P["hc_ffn_scale"],
                           P["hc_ffn_base"], cfg, f64, interm, "ffn_hc")
    x = rms_norm(x, P["post_norm"], cfg["rms_norm_eps"], f64)        # 1324
    if P["mlp_kind"] == "dense":
        x = swiglu_mlp(x, P["mlp_gate"], P["mlp_up"], P["mlp_down"],
                       cfg["swiglu_limit"], f64)                     # 1271-1272
    else:
        x = moe_forward(P, x, f64, interm)
    h = hc_post(x, residual, post, comb, f64)                        # 1326-1328
    return h, interm


def model_forward(Ps, top, cfg, ids, states, f64, decode, layer_interm=None):
    """Glm5NextTextModel.forward + lm_head (glm5:1432-1495, 2180-2182),
    batch squeezed. ids [s]; states = per-layer LayerState list (mutated).
    Returns (logits [s, V], h [s, hc, dim]) — h is kept for the deferred
    MTP layer (M8+).

    layer_interm (ADDITIVE, M5): when a list is passed, each layer's
    block_forward interm dict is appended (plus "block_out_h" — the
    per-layer block output h, bf16-valued) so model-level gates
    (tests/m5g) can harvest router selections, indexer top-k, and
    per-layer hidden states. Existing callers pass nothing."""
    dt = _dt(f64)
    ids = np.asarray(ids, dtype=np.int64)
    h = top["embed"][ids].astype(dt)                       # bf16-valued rows
    h = np.repeat(h[:, None, :], cfg["hc_mult"], axis=1)   # 1478
    for i, P in enumerate(Ps):                             # 1481-1492
        h, im = block_forward(P, cfg, h, states[i], f64, decode)
        if layer_interm is not None:
            im["block_out_h"] = h.copy()
            layer_interm.append(im)
    y = _B(h.astype(dt).mean(axis=1), f64)                 # 1494: HyperHead
    yn = rms_norm(y, top["norm"], cfg["rms_norm_eps"], f64)  # 1494
    logits = bf16_linear(yn, top["head"], f64)             # 2182
    return logits, h


# ---------------------------------------------------------------------------
# MTP (classic NextN, layers.<L>.*) — M8a. HF drops layer 45 at load; the
# normative reference is SGLang deepseek_nextn.py (glm5_next's NextN is a
# thin subclass). Depth-1 draft head sharing the main model's embed_tokens
# and lm_head:
#   eh = cat([rms_norm(embed(tok), enorm), rms_norm(prev_h, hnorm)], -1)
#   x  = bf16_linear(eh, eh_proj)                    # BF16 [dim, 2*dim]
#   ONE full decoder layer (DSA + sparse MoE) with PLAIN residuals (no mHC,
#   eager-torch add semantics: each residual add is bf16-rounded)
#   out = rms_norm(x, shared_head.norm)              # fused add+norm target
#   logits = bf16_linear(out, shared lm_head)
# Draft tokens are always argmax of these logits. The chain feeds the
# draft's own post-shared_head.norm hidden `out` back as prev_h for the
# next draft step, with the drafted token's embedding (vLLM PR #47448
# semantics; matches the parent engine's ../Apus c/mtp.h apus_spec_chain).
#
# (h, id) pairing: a true pair is (prev_h, tok) BOTH at position p, and the
# step predicts position p+1 — the parent engine's convention
# (../Apus/c/mtp.h apus_spec_prefill/step replay true pairs (H[j],
# batch[j]) with H[j] the POST-token hidden at the token's own position).
# NOTE: this differs by one hidden position from DeepSeek-V3/SGLang EAGLE
# bookkeeping, which pairs (h_{p-1}, tok_p). GLM-5.3-Flash's pairing is
# pinned EMPIRICALLY by tools/mtp_pin.py (M8a) — see docs/STATUS.md.
# ---------------------------------------------------------------------------

# Which main-model hidden feeds hnorm. Candidates evaluated by
# tools/mtp_pin.py on the real container (acceptance vs the greedy stream):
#   "prenorm_mean"  — the HyperHead mean, bf16-rounded (model_forward's `y`)
#   "postnorm"      — after the final rms_norm (model_forward's `yn`)
#   "hc0".."hc3"    — an individual mHC stream slot
# PINNED value: see docs/STATUS.md M8a entry (tools/mtp_pin.py results).
MTP_HNORM_INPUT = "postnorm"  # PINNED post-gate7 (M8a re-pin 2026-09-05: engine sweep 85.3% accept / 2.53 tok-per-batch — the SGLang/vLLM reference semantics)

# (h, id) pairing lag: 0 = (h_p, tok_p) predicts tok_{p+1} (parent-engine
# convention); 1 = (h_{p-1}, tok_p) predicts tok_{p+1} (DeepSeek/SGLang
# bookkeeping). PINNED by tools/mtp_pin.py alongside MTP_HNORM_INPUT.
MTP_PAIR_LAG = 1  # PINNED post-gate7 (the EAGLE bookkeeping; the re-pin sweep's decisive winner)


def mtp_hnorm_input(h, top, cfg, f64, candidate=MTP_HNORM_INPUT):
    """Collapse the trunk's mHC stream h [s, hc, dim] to the single
    [s, dim] hidden the MTP hnorm consumes (the M8a open question — the
    trunk stream is 4x4096, hnorm is 4096-wide). `candidate` overrides the
    pinned MTP_HNORM_INPUT (tools/mtp_pin.py sweeps all of them)."""
    dt = _dt(f64)
    if candidate == "prenorm_mean":
        return _B(h.astype(dt).mean(axis=1), f64)          # HyperHead mean
    if candidate == "postnorm":
        y = _B(h.astype(dt).mean(axis=1), f64)
        return rms_norm(y, top["norm"], cfg["rms_norm_eps"], f64)
    if candidate.startswith("hc"):
        return h[:, int(candidate[2:]), :].astype(dt)
    raise ValueError(f"unknown hnorm candidate: {candidate}")


def mtp_forward(Pm, top, cfg, ids, prev_h, mtp_state, f64, interm=None):
    """Batched MTP (NextN) forward. ids [s], prev_h [s, dim] — row i is the
    pair (prev_h[i], ids[i]) at consecutive positions (the CALLER chooses
    the pairing; the engine uses the pinned MTP_PAIR_LAG). mtp_state is the
    MTP block's own DSA-kind LayerState (mutated; its pos is bumped by s —
    it tracks the main model's position). Returns (logits [s, V],
    out_h [s, dim]) where out_h is the post-shared_head.norm hidden (the
    chaining input)."""
    dt = _dt(f64)
    ids = np.asarray(ids, dtype=np.int64)
    eps = cfg["rms_norm_eps"]
    e = rms_norm(top["embed"][ids], Pm["enorm"], eps, f64)
    hh = rms_norm(np.asarray(prev_h), Pm["hnorm"], eps, f64)
    eh = np.concatenate([e, hh], axis=-1)                  # [s, 2*dim]
    x = bf16_linear(eh, Pm["eh_proj"], f64)
    residual = x
    x = rms_norm(x, Pm["input_norm"], eps, f64)
    x = dsa_forward(Pm, x, mtp_state, f64, interm)
    x = _B(residual + x, f64)                              # plain residual
    residual = x
    x = rms_norm(x, Pm["post_norm"], eps, f64)
    x = moe_forward(Pm, x, f64, interm)
    x = _B(residual + x, f64)                              # plain residual
    out = rms_norm(x, Pm["shared_norm"], eps, f64)         # fused add+norm
    logits = bf16_linear(out, top["head"], f64)            # shared lm_head
    mtp_state.pos += ids.shape[0]
    return logits, out


def mtp_chain(Pm, top, cfg, seed_id, seed_h, mtp_state, depth, f64,
              interm=None):
    """Draft chain (parent ApusSpec apus_spec_chain shape): the seed pair
    (seed_h, seed_id) is fed first and its argmax is draft[0]; draft[i]
    (i >= 1) chains from the pair (out_{i-1}, draft[i-1]) — the draft's own
    post-shared_head.norm hidden + the previous drafted token. MUTATES
    mtp_state (draft pairs pollute the attention state — the engine
    snapshots/restores around chains). Returns (draft_ids [depth] int64,
    step_logits [depth, V], step_h [depth, dim]). interm (ADDITIVE): when
    a list is passed, each step's own interm dict (router/indexer
    selections etc.) is appended; a dict is passed to every step as-is."""
    cur_id = int(seed_id)
    cur_h = np.asarray(seed_h)
    drafts = np.zeros(depth, dtype=np.int64)
    step_logits = []
    step_h = []
    for i in range(depth):
        im = {} if isinstance(interm, list) else interm
        lg, out_h = mtp_forward(Pm, top, cfg, [cur_id], cur_h[None, :],
                                mtp_state, f64, im)
        if isinstance(interm, list):
            interm.append(im)
        cur_id = int(np.argmax(lg[0]))                    # argmax drafts
        cur_h = out_h[0]
        drafts[i] = cur_id
        step_logits.append(lg[0])
        step_h.append(out_h[0])
    return drafts, np.stack(step_logits), np.stack(step_h)


# ---------------------------------------------------------------------------
# Layer state (decode-carried) + serialization + digest.
# ---------------------------------------------------------------------------


class LayerState:
    """Decode-carried state for one layer. KDA: conv_state [k-1, 3qkv]
    (bf16 values) + rec_state [H, D, D] fp32. DSA: expanded kv cache
    [n, H, D] (bf16 values) + indexer caches idx_k/idx_gate [n, ID]."""

    def __init__(self, cfg, layer_idx, kind=None):
        self.pos = 0
        if kind is None:
            kind = ("kda" if cfg["layer_types"][layer_idx]
                    == "linear_attention" else "dsa")
        assert kind in ("kda", "dsa"), kind
        self.kind = kind
        if self.kind == "kda":
            qkv = cfg["linear_num_heads"] * cfg["linear_head_dim"]
            self.conv_state = np.zeros((cfg["linear_conv_kernel_dim"] - 1,
                                        3 * qkv), dtype=np.float32)
            self.rec_state = None
        else:
            self.k_cache = None
            self.v_cache = None
            self.idx_k = None
            self.idx_gate = None


def new_model_states(cfg):
    return [LayerState(cfg, i) for i in range(cfg["num_hidden_layers"])]


def state_arrays(st):
    """Flatten the decode-carried state to named arrays (f32)."""
    out = {"pos": np.array(st.pos, np.int64), "kind": np.array(
        [1 if st.kind == "kda" else 2], np.int64)}
    if st.kind == "kda":
        out["conv_state"] = st.conv_state.astype(np.float32)
        if st.rec_state is not None:
            out["rec_state"] = st.rec_state.astype(np.float32)
    else:
        for name in ("k_cache", "v_cache", "idx_k", "idx_gate"):
            a = getattr(st, name)
            if a is not None:
                out[name] = a.astype(np.float32)
    return out


def state_digest(states):
    """FNV-1a over native-byte state arrays in canonical order. Same-mode
    comparison only (KDA rec_state dtype differs f32/f64)."""
    chunks = []
    for st in states:
        chunks.append(np.asarray(st.pos, np.int64).tobytes())
        for name in sorted(state_arrays(st)):
            if name in ("pos", "kind"):
                continue
            chunks.append(np.ascontiguousarray(state_arrays(st)[name]).tobytes())
    return fnv1a(chunks)


# ---------------------------------------------------------------------------
# Public entry points (the per-phase contract surface).
# ---------------------------------------------------------------------------


def prefill(Ps, top, cfg, ids, f64=False, layer_interm=None):
    """One-shot prefill from position 0. KDA layers use the CHUNKED path.
    Returns (logits [s, V], h [s, hc, dim], states). layer_interm: see
    model_forward (ADDITIVE, M5)."""
    states = new_model_states(cfg)
    logits, h = model_forward(Ps, top, cfg, ids, states, f64, decode=False,
                              layer_interm=layer_interm)
    for st in states:
        st.pos = len(ids)
    return logits, h, states


def decode_step(Ps, top, cfg, token_id, states, f64=False, layer_interm=None):
    """One decode token at position states[*].pos. KDA layers use the
    RECURRENT path. Returns logits [V]. layer_interm: see model_forward
    (ADDITIVE, M5)."""
    logits, h = model_forward(Ps, top, cfg, [token_id], states, f64,
                              decode=True, layer_interm=layer_interm)
    for st in states:
        st.pos += 1
    return logits[0]


# ===========================================================================
# Synthetic weight generation + safetensors IO (inlined from tests/m1
# stutil.py — this file is self-contained by design, see module docstring).
# ===========================================================================

_ST_DTYPE_SIZES = {
    "I8": 1, "U8": 1, "F8_E8M0": 1, "F8_E4M3": 1,
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8,
}


def write_shard(path, tensors):
    """Write tensors = [(name, dtype, shape, payload_bytes), ...] as one
    safetensors shard, manually (8-byte LE header length + JSON + data)."""
    import struct
    header = {}
    off = 0
    data = []
    for name, dtype, shape, payload in tensors:
        n = _ST_DTYPE_SIZES[dtype]
        for d in shape:
            n *= d
        assert len(payload) == n, (name, len(payload), n)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(payload)]}
        off += len(payload)
        data.append(payload)
    doc = json.dumps(header, separators=(",", ":")).encode()
    doc += b" " * ((-len(doc)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(doc)))
        f.write(doc)
        for payload in data:
            f.write(payload)


def read_shard(path):
    """Parse a shard manually. Returns (header_dict, data_start)."""
    import struct
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def read_tensor_bytes(path):
    """Returns dict name -> raw payload bytes."""
    header, data_start = read_shard(path)
    out = {}
    with open(path, "rb") as f:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            out[name] = f.read(end - begin)
    return out


def gen_layer_tensors(cfg, layer_idx, rng):
    """Generate one layer's tensors as (name, dtype, shape, payload) records
    using the real checkpoint naming (minus the "model.language_model."
    prefix). Layer types come from cfg["layer_types"]/["mlp_layer_types"]."""
    L = f"layers.{layer_idx}"
    d = cfg["hidden_size"]
    hc = cfg["hc_mult"]
    recs = []

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))
                ).astype(np.float32)

    def fp8(name, W):
        cb, sb, csh, ssh = fp8_store(W)
        recs.append((f"{L}.{name}.weight", "F8_E4M3", csh, cb))
        recs.append((f"{L}.{name}.weight_scale_inv", "F32", ssh, sb))

    def bf16(name, W):
        recs.append((f"{L}.{name}", "BF16", W.shape, f32_to_bf16_bytes(W)))

    def f32raw(name, W):
        recs.append((f"{L}.{name}", "F32", W.shape,
                     np.ascontiguousarray(W, np.float32).tobytes()))

    def norm_w(name, n):
        bf16(name, (1.0 + 0.05 * rng.standard_normal(n)).astype(np.float32))

    norm_w("input_layernorm.weight", d)
    norm_w("post_attention_layernorm.weight", d)

    if cfg["layer_types"][layer_idx] == "linear_attention":
        H, D = cfg["linear_num_heads"], cfg["linear_head_dim"]
        qkv = H * D
        k = cfg["linear_conv_kernel_dim"]
        for p in ("q", "k", "v"):
            bf16(f"self_attn.{p}_proj.weight", rnd(qkv, d))
            bf16(f"self_attn.{p}_conv1d.weight",
                 (0.25 * rng.standard_normal((qkv, 1, k))).astype(np.float32))
        bf16("self_attn.f_a_proj.weight", rnd(D, d))
        bf16("self_attn.f_b_proj.weight", rnd(qkv, D))
        f32raw("self_attn.A_log",
               (0.5 * rng.standard_normal(H)).astype(np.float32))
        f32raw("self_attn.dt_bias",
               (0.5 * rng.standard_normal(qkv)).astype(np.float32))
        bf16("self_attn.b_proj.weight", rnd(H, d))
        bf16("self_attn.g_a_proj.weight", rnd(D, d))
        bf16("self_attn.g_b_proj.weight", rnd(qkv, D))
        norm_w("self_attn.o_norm.weight", D)
        bf16("self_attn.o_proj.weight", rnd(d, qkv))
    else:
        H = cfg["num_attention_heads"]
        ql, kl = cfg["q_lora_rank"], cfg["kv_lora_rank"]
        qd, vd = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
        fp8("self_attn.q_a_proj", rnd(ql, d))
        norm_w("self_attn.q_a_layernorm.weight", ql)
        fp8("self_attn.q_b_proj", rnd(H * qd, ql))
        fp8("self_attn.kv_a_proj_with_mqa", rnd(kl, d))
        norm_w("self_attn.kv_a_layernorm.weight", kl)
        bf16("self_attn.kv_b_proj.weight", rnd(H * (qd + vd), kl))
        fp8("self_attn.o_proj", rnd(d, H * vd))
        IH, ID = cfg["index_n_heads"], cfg["index_head_dim"]
        bf16("self_attn.indexer.wq_b.weight", rnd(IH * ID, ql))
        bf16("self_attn.indexer.wk.weight", rnd(ID, d))
        norm_w("self_attn.indexer.k_norm.weight", ID)
        bf16("self_attn.indexer.k_norm.bias",
             (0.05 * rng.standard_normal(ID)).astype(np.float32))
        bf16("self_attn.indexer.weights_proj.weight", rnd(IH, d))
        bf16("self_attn.indexer.index_kpool_compress_ape",
             (0.05 * rng.standard_normal((cfg["index_kpool"], ID))
              ).astype(np.float32))
        bf16("self_attn.indexer.index_kpool_compress_gate", rnd(ID, d))

    if cfg["mlp_layer_types"][layer_idx] == "dense":
        inter = cfg["intermediate_size"]
        fp8("mlp.gate_proj", rnd(inter, d))
        fp8("mlp.up_proj", rnd(inter, d))
        fp8("mlp.down_proj", rnd(d, inter))
    else:
        E = cfg["n_routed_experts"]
        inter = cfg["moe_intermediate_size"]
        bf16("mlp.gate.weight",
             (0.05 * rng.standard_normal((E, d))).astype(np.float32))
        f32raw("mlp.gate.e_score_correction_bias",
               (0.1 * rng.standard_normal(E)).astype(np.float32))
        for e in range(E):
            fp8(f"mlp.experts.{e}.gate_proj", rnd(inter, d))
            fp8(f"mlp.experts.{e}.up_proj", rnd(inter, d))
            fp8(f"mlp.experts.{e}.down_proj", rnd(d, inter))
        sinter = inter * cfg["n_shared_experts"]
        fp8("mlp.shared_experts.gate_proj", rnd(sinter, d))
        fp8("mlp.shared_experts.up_proj", rnd(sinter, d))
        fp8("mlp.shared_experts.down_proj", rnd(d, sinter))

    mix = (2 + hc) * hc
    for sub in ("attn", "ffn"):
        bf16(f"hc_{sub}_fn", rnd(mix, hc * d))
        f32raw(f"hc_{sub}_base",
               (0.25 * rng.standard_normal(mix)).astype(np.float32))
        f32raw(f"hc_{sub}_scale",
               (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32))
    return recs


def gen_toplevel_tensors(cfg, rng):
    """embed/norm BF16, lm_head BF16 (untied in the real checkpoint)."""
    d, V = cfg["hidden_size"], cfg["vocab_size"]

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))
                ).astype(np.float32)

    return [
        ("embed_tokens.weight", "BF16", (V, d),
         f32_to_bf16_bytes(rnd(V, d, std=1.0))),
        ("norm.weight", "BF16", (d,),
         f32_to_bf16_bytes((1.0 + 0.05 * rng.standard_normal(d)).astype(np.float32))),
        ("lm_head.weight", "BF16", (V, d), f32_to_bf16_bytes(rnd(V, d))),
    ]


def gen_mtp_tensors(cfg, layer_idx, rng):
    """Generate the MTP (NextN) block's tensors as (name, dtype, shape,
    payload) records under layers.<layer_idx>.* — a full DSA + sparse-MoE
    decoder layer with PLAIN residuals (NO hc_* tensors — convert.py's
    validate_mtp_rules rejects them on MTP layers) plus the e/h glue
    (enorm/hnorm/eh_proj, BF16 like the real checkpoint) and its own
    shared_head.norm. Dtype conventions mirror the main model."""
    L = f"layers.{layer_idx}"
    d = cfg["hidden_size"]
    recs = []

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))
                ).astype(np.float32)

    def fp8(name, W):
        cb, sb, csh, ssh = fp8_store(W)
        recs.append((f"{L}.{name}.weight", "F8_E4M3", csh, cb))
        recs.append((f"{L}.{name}.weight_scale_inv", "F32", ssh, sb))

    def bf16(name, W):
        recs.append((f"{L}.{name}", "BF16", W.shape, f32_to_bf16_bytes(W)))

    def f32raw(name, W):
        recs.append((f"{L}.{name}", "F32", W.shape,
                     np.ascontiguousarray(W, np.float32).tobytes()))

    def norm_w(name, n):
        bf16(name, (1.0 + 0.05 * rng.standard_normal(n)).astype(np.float32))

    # e/h glue (BF16 in the real checkpoint — eh_proj [dim, 2*dim])
    norm_w("enorm.weight", d)
    norm_w("hnorm.weight", d)
    bf16("eh_proj.weight", rnd(d, 2 * d))

    # decoder layer: input/post norms + DSA self-attn (as the main model's
    # DSA branch, same tensor names)
    norm_w("input_layernorm.weight", d)
    norm_w("post_attention_layernorm.weight", d)
    H = cfg["num_attention_heads"]
    ql, kl = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    qd, vd = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    fp8("self_attn.q_a_proj", rnd(ql, d))
    norm_w("self_attn.q_a_layernorm.weight", ql)
    fp8("self_attn.q_b_proj", rnd(H * qd, ql))
    fp8("self_attn.kv_a_proj_with_mqa", rnd(kl, d))
    norm_w("self_attn.kv_a_layernorm.weight", kl)
    bf16("self_attn.kv_b_proj.weight", rnd(H * (qd + vd), kl))
    fp8("self_attn.o_proj", rnd(d, H * vd))
    IH, ID = cfg["index_n_heads"], cfg["index_head_dim"]
    bf16("self_attn.indexer.wq_b.weight", rnd(IH * ID, ql))
    bf16("self_attn.indexer.wk.weight", rnd(ID, d))
    norm_w("self_attn.indexer.k_norm.weight", ID)
    bf16("self_attn.indexer.k_norm.bias",
         (0.05 * rng.standard_normal(ID)).astype(np.float32))
    bf16("self_attn.indexer.weights_proj.weight", rnd(IH, d))
    bf16("self_attn.indexer.index_kpool_compress_ape",
         (0.05 * rng.standard_normal((cfg["index_kpool"], ID))
          ).astype(np.float32))
    bf16("self_attn.indexer.index_kpool_compress_gate", rnd(ID, d))

    # sparse MoE (router + FP8 experts + shared expert), as the main model
    E = cfg["n_routed_experts"]
    inter = cfg["moe_intermediate_size"]
    bf16("mlp.gate.weight",
         (0.05 * rng.standard_normal((E, d))).astype(np.float32))
    f32raw("mlp.gate.e_score_correction_bias",
           (0.1 * rng.standard_normal(E)).astype(np.float32))
    for e in range(E):
        fp8(f"mlp.experts.{e}.gate_proj", rnd(inter, d))
        fp8(f"mlp.experts.{e}.up_proj", rnd(inter, d))
        fp8(f"mlp.experts.{e}.down_proj", rnd(d, inter))
    sinter = inter * cfg["n_shared_experts"]
    fp8("mlp.shared_experts.gate_proj", rnd(sinter, d))
    fp8("mlp.shared_experts.up_proj", rnd(sinter, d))
    fp8("mlp.shared_experts.down_proj", rnd(d, sinter))

    # fused add+norm before the SHARED lm_head (the block has no own head)
    norm_w("shared_head.norm.weight", d)
    return recs


def write_weights(cfg, out_dir, seed=MASTER_SEED, mtp=False):
    """Generate all weights and write apus-style shards + index json.
    mtp=True also writes the NextN block as layers.<num_hidden_layers>.*
    (plain residuals, no hc_* — the convert.py MTP schema)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed + 90000)
    dense_recs = gen_toplevel_tensors(cfg, rng)
    expert_recs = []
    for li in range(cfg["num_hidden_layers"]):
        lr = np.random.default_rng(seed + 1000 * li)
        for rec in gen_layer_tensors(cfg, li, lr):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)
    if mtp:
        L = cfg["num_hidden_layers"]
        mr = np.random.default_rng(seed + 1000 * L)
        for rec in gen_mtp_tensors(cfg, L, mr):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)
    shards = [("apus-00001.safetensors", dense_recs)]
    if expert_recs:
        shards.append(("apus-00002.safetensors", expert_recs))
    weight_map, total = {}, 0
    for fname, recs in shards:
        write_shard(os.path.join(out_dir, fname), recs)
        for name, dtype, shape, payload in recs:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f, indent=1)


# ---------------------------------------------------------------------------
# Loader: reads the fixture safetensors back into oracle params, exercising
# the real naming/format path the C loader must implement.
# ---------------------------------------------------------------------------


class ShardSet:
    def __init__(self, weights_dir):
        with open(os.path.join(weights_dir,
                               "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.dir = weights_dir
        self._cache = {}

    def raw(self, name):
        fname = self.weight_map[name]
        if fname not in self._cache:
            self._cache[fname] = read_tensor_bytes(os.path.join(self.dir,
                                                                fname))
        return self._cache[fname][name]

    def meta(self, name):
        fname = self.weight_map[name]
        header, _ = read_shard(os.path.join(self.dir, fname))
        return header[name]["dtype"], header[name]["shape"]

    def f32(self, name):
        """F32 or BF16 tensor -> f32 array (bf16 values stay bf16-valued)."""
        dtype, shape = self.meta(name)
        b = self.raw(name)
        if dtype == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        if dtype == "BF16":
            return bf16_bytes_to_f32(b).reshape(shape)
        raise ValueError(f"{name}: {dtype}")

    def fp8(self, name):
        """-> (codes u8 [O,K], scale_inv f32 [O/128,K/128])."""
        dt, csh = self.meta(name + ".weight")
        assert dt == "F8_E4M3", (name, dt)
        codes = np.frombuffer(self.raw(name + ".weight"), np.uint8).reshape(csh)
        dt, ssh = self.meta(name + ".weight_scale_inv")
        assert dt == "F32", (name, dt)
        scales = np.frombuffer(self.raw(name + ".weight_scale_inv"),
                               np.float32).reshape(ssh)
        return codes.copy(), scales.copy()


def load_layer_params(shards, cfg, layer_idx, kind=None, mlp_kind=None):
    """Build the oracle param dict for one layer from the safetensors.
    kind/mlp_kind override cfg["layer_types"]/["mlp_layer_types"] (used by
    load_mtp_params for the NextN block at layer index num_hidden_layers,
    which the cfg lists do not cover)."""
    L = f"layers.{layer_idx}"
    ltype = cfg["layer_types"][layer_idx] if kind is None else kind
    mtype = (cfg["mlp_layer_types"][layer_idx] if mlp_kind is None
             else mlp_kind)
    P = {
        "input_norm": shards.f32(f"{L}.input_layernorm.weight"),
        "post_norm": shards.f32(f"{L}.post_attention_layernorm.weight"),
        "mlp_kind": mtype,
        "eps": cfg["rms_norm_eps"],
    }
    if kind is None and mlp_kind is None:           # main layers only
        P.update({
            "hc_attn_fn": shards.f32(f"{L}.hc_attn_fn"),
            "hc_attn_base": shards.f32(f"{L}.hc_attn_base"),
            "hc_attn_scale": shards.f32(f"{L}.hc_attn_scale"),
            "hc_ffn_fn": shards.f32(f"{L}.hc_ffn_fn"),
            "hc_ffn_base": shards.f32(f"{L}.hc_ffn_base"),
            "hc_ffn_scale": shards.f32(f"{L}.hc_ffn_scale"),
        })
    if ltype == "linear_attention":
        H, D = cfg["linear_num_heads"], cfg["linear_head_dim"]
        qkv = H * D
        conv = np.concatenate([shards.f32(f"{L}.self_attn.{p}_conv1d.weight")
                               [:, 0, :] for p in ("q", "k", "v")], axis=0)
        P.update({
            "kind": "kda", "heads": H, "head_dim": D, "qkv_dim": qkv,
            "q_w": shards.f32(f"{L}.self_attn.q_proj.weight"),
            "k_w": shards.f32(f"{L}.self_attn.k_proj.weight"),
            "v_w": shards.f32(f"{L}.self_attn.v_proj.weight"),
            "conv_w": conv,                       # [3*qkv, k], q|k|v order
            "f_a": shards.f32(f"{L}.self_attn.f_a_proj.weight"),
            "f_b": shards.f32(f"{L}.self_attn.f_b_proj.weight"),
            "A_log": shards.f32(f"{L}.self_attn.A_log"),
            "dt_bias": shards.f32(f"{L}.self_attn.dt_bias"),
            "b_w": shards.f32(f"{L}.self_attn.b_proj.weight"),
            "g_a": shards.f32(f"{L}.self_attn.g_a_proj.weight"),
            "g_b": shards.f32(f"{L}.self_attn.g_b_proj.weight"),
            "o_norm": shards.f32(f"{L}.self_attn.o_norm.weight"),
            "o_w": shards.f32(f"{L}.self_attn.o_proj.weight"),
            "lower_bound": cfg["linear_lower_bound"],
        })
    else:
        H = cfg["num_attention_heads"]
        qd, vd = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
        P.update({
            "kind": "dsa", "heads": H,
            "qk_head_dim": qd + cfg["qk_rope_head_dim"],
            "v_head_dim": vd,
            "attn_scale": (qd + cfg["qk_rope_head_dim"]) ** -0.5,
            "q_a": shards.fp8(f"{L}.self_attn.q_a_proj"),
            "q_a_norm": shards.f32(f"{L}.self_attn.q_a_layernorm.weight"),
            "q_b": shards.fp8(f"{L}.self_attn.q_b_proj"),
            "kv_a": shards.fp8(f"{L}.self_attn.kv_a_proj_with_mqa"),
            "kv_a_norm": shards.f32(f"{L}.self_attn.kv_a_layernorm.weight"),
            "kv_b": shards.f32(f"{L}.self_attn.kv_b_proj.weight"),
            "o_proj": shards.fp8(f"{L}.self_attn.o_proj"),
            "idx_heads": cfg["index_n_heads"],
            "idx_dim": cfg["index_head_dim"],
            "idx_topk": cfg["index_topk"],
            "idx_kpool": cfg["index_kpool"],
            "idx_tail": cfg["index_kpool_always_select_tail"],
            "idx_scale": cfg["index_head_dim"] ** -0.5,
            "idx_wq_b": shards.f32(f"{L}.self_attn.indexer.wq_b.weight"),
            "idx_wk": shards.f32(f"{L}.self_attn.indexer.wk.weight"),
            "idx_knorm_w": shards.f32(f"{L}.self_attn.indexer.k_norm.weight"),
            "idx_knorm_b": shards.f32(f"{L}.self_attn.indexer.k_norm.bias"),
            "idx_wproj": shards.f32(f"{L}.self_attn.indexer.weights_proj.weight"),
            "idx_ape": shards.f32(
                f"{L}.self_attn.indexer.index_kpool_compress_ape"),
            "idx_gate": shards.f32(
                f"{L}.self_attn.indexer.index_kpool_compress_gate"),
        })
    if P["mlp_kind"] == "dense":
        P["mlp_gate"] = shards.fp8(f"{L}.mlp.gate_proj")
        P["mlp_up"] = shards.fp8(f"{L}.mlp.up_proj")
        P["mlp_down"] = shards.fp8(f"{L}.mlp.down_proj")
    else:
        P.update({
            "gate_w": shards.f32(f"{L}.mlp.gate.weight"),
            "gate_bias": shards.f32(f"{L}.mlp.gate.e_score_correction_bias"),
            "top_k": cfg["num_experts_per_tok"],
            "route_scale": cfg["routed_scaling_factor"],
            "norm_topk_prob": cfg["norm_topk_prob"],
            "n_group": cfg["n_group"],
            "topk_group": cfg["topk_group"],
            "n_routed": cfg["n_routed_experts"],
            "limit": cfg["swiglu_limit"],
            "experts": [(shards.fp8(f"{L}.mlp.experts.{e}.gate_proj"),
                         shards.fp8(f"{L}.mlp.experts.{e}.up_proj"),
                         shards.fp8(f"{L}.mlp.experts.{e}.down_proj"))
                        for e in range(cfg["n_routed_experts"])],
            "shared": (shards.fp8(f"{L}.mlp.shared_experts.gate_proj"),
                       shards.fp8(f"{L}.mlp.shared_experts.up_proj"),
                       shards.fp8(f"{L}.mlp.shared_experts.down_proj")),
        })
    return P


def load_mtp_params(shards, cfg, layer_idx=None):
    """Load the MTP (NextN) block at layers.<num_hidden_layers>.* — a full
    DSA + sparse-MoE decoder layer (PLAIN residuals, no hc_* tensors) plus
    the e/h glue (enorm/hnorm/eh_proj) and its own shared_head.norm. The
    block shares the main model's embed_tokens/lm_head (via `top`)."""
    L = cfg["num_hidden_layers"] if layer_idx is None else layer_idx
    P = load_layer_params(shards, cfg, L, kind="deepseek_sparse_attention",
                          mlp_kind="sparse")
    P.update({
        "enorm": shards.f32(f"layers.{L}.enorm.weight"),
        "hnorm": shards.f32(f"layers.{L}.hnorm.weight"),
        "eh_proj": shards.f32(f"layers.{L}.eh_proj.weight"),
        "shared_norm": shards.f32(f"layers.{L}.shared_head.norm.weight"),
    })
    return P


def new_mtp_state(cfg):
    """Fresh DSA-kind LayerState for the MTP block (its pos tracks the main
    model's; bumped by mtp_forward itself)."""
    return LayerState(cfg, cfg["num_hidden_layers"], kind="dsa")


def load_model_params(shards, cfg):
    """Load every layer + top-level params from the fixture safetensors."""
    Ps = [load_layer_params(shards, cfg, i)
          for i in range(cfg["num_hidden_layers"])]
    top = {
        "embed": shards.f32("embed_tokens.weight"),
        "norm": shards.f32("norm.weight"),
        "head": shards.f32("lm_head.weight"),
    }
    return Ps, top


# ===========================================================================
# Fixture driver: weights + golden I/O + manifest with digests.
# ===========================================================================

FIXTURE_SEQUENCES = {
    "prefill": ("prefill_len68", 68),          # name, length (2 KDA chunks)
    "decode": ("decode_from68", 68, 6),        # name, prompt len, steps
}


def _gen_ids(cfg, seed, s):
    return np.random.default_rng(seed).integers(0, cfg["vocab_size"],
                                                size=s).astype(np.int64)


def _save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def generate(fixtures_dir, cfg=None, clean=True, seed=MASTER_SEED):
    """Generate the M0 fixture set (synthetic weights + golden logits +
    state digests) for the tiny config. Weights are written as safetensors
    and loaded back through ShardSet, so the oracle consumes exactly what
    the C loader will consume."""
    if cfg is None:
        cfg = make_tiny_config()
    if clean and os.path.isdir(fixtures_dir):
        import shutil
        shutil.rmtree(fixtures_dir)
    os.makedirs(fixtures_dir, exist_ok=True)
    with open(os.path.join(fixtures_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=1)

    weights_dir = os.path.join(fixtures_dir, "weights")
    write_weights(cfg, weights_dir, seed)
    shards = ShardSet(weights_dir)
    Ps, top = load_model_params(shards, cfg)

    gdir = os.path.join(fixtures_dir, "golden")
    manifest = {"seed": seed, "config": "config.json",
                "weights": "weights/", "sequences": {},
                "digests": {}}

    pname, plen = FIXTURE_SEQUENCES["prefill"]
    pids = _gen_ids(cfg, seed + 17 * plen, plen)
    l32, _, st32 = prefill(Ps, top, cfg, pids, f64=False)
    l64, _, st64 = prefill(Ps, top, cfg, pids, f64=True)
    l32b, _, _ = prefill(Ps, top, cfg, pids, f64=False)
    assert np.array_equal(l32, l32b), "f32 prefill not deterministic"
    d = os.path.join(gdir, pname)
    _save_npy(os.path.join(d, "input_ids.npy"), pids)
    _save_npy(os.path.join(d, "logits_f32.npy"), l32.astype(np.float32))
    _save_npy(os.path.join(d, "logits_f64.npy"), l64.astype(np.float64))
    manifest["sequences"][pname] = {"kind": "prefill", "len": plen}
    manifest["digests"][pname] = {
        "logits_f32": array_digest(l32.astype(np.float32)),
        "logits_f64": array_digest(l64.astype(np.float64)),
        "state_f32": state_digest(st32),
    }
    print(f"  {pname} done")

    dname, dplen, steps = FIXTURE_SEQUENCES["decode"]
    assert dplen == plen
    dids = _gen_ids(cfg, seed + 24000, steps)
    ls32 = np.stack([decode_step(Ps, top, cfg, t, st32, f64=False)
                     for t in dids])
    ls64 = np.stack([decode_step(Ps, top, cfg, t, st64, f64=True)
                     for t in dids])
    d = os.path.join(gdir, dname)
    _save_npy(os.path.join(d, "prompt_ids.npy"), pids)
    _save_npy(os.path.join(d, "decode_ids.npy"), dids)
    _save_npy(os.path.join(d, "logits_f32.npy"), ls32.astype(np.float32))
    _save_npy(os.path.join(d, "logits_f64.npy"), ls64.astype(np.float64))
    manifest["sequences"][dname] = {"kind": "decode", "prompt_len": dplen,
                                    "steps": steps}
    manifest["digests"][dname] = {
        "logits_f32": array_digest(ls32.astype(np.float32)),
        "logits_f64": array_digest(ls64.astype(np.float64)),
        "state_f32": state_digest(st32),
    }
    print(f"  {dname} done")

    # KDA ordering-contract evidence: extend the prompt by one token via a
    # fresh one-shot CHUNKED prefill vs one RECURRENT decode step. Same math,
    # different fp32 orderings — f64 must agree to ~1e-9, f32 is reported.
    xids = np.concatenate([pids, dids[:1]])
    lx32, _, _ = prefill(Ps, top, cfg, xids, f64=False)
    lx64, _, _ = prefill(Ps, top, cfg, xids, f64=True)
    step32 = decode_step(Ps, top, cfg, dids[0],
                         prefill(Ps, top, cfg, pids, f64=False)[2],
                         f64=False)
    step64 = decode_step(Ps, top, cfg, dids[0],
                         prefill(Ps, top, cfg, pids, f64=True)[2],
                         f64=True)
    manifest["ordering_contract"] = {
        "f64_max_abs_diff": float(np.abs(lx64[-1] - step64).max()),
        "f32_max_abs_diff": float(np.abs(lx32[-1] - step32).max()),
        "f32_bitwise": bool(np.array_equal(lx32[-1], step32)),
    }
    print(f"  ordering contract: f64 diff "
          f"{manifest['ordering_contract']['f64_max_abs_diff']:.3g}, "
          f"f32 diff {manifest['ordering_contract']['f32_max_abs_diff']:.3g}")

    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"fixtures written to {fixtures_dir}")
    return manifest


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="apus-glm-5.3-flash M0 oracle fixture generator")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "m0",
                                                  "fixtures"))
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    args = ap.parse_args(argv)
    generate(args.out, clean=not args.no_clean, seed=args.seed)


if __name__ == "__main__":
    main()
