#!/usr/bin/env python3
"""tests/m0/check_vs_hf.py — the EXTERNAL ANCHOR for the verification
chain: our numpy oracle (tools/oracle.py) vs the TRUE HuggingFace
transformers glm5_next implementation (transformers @ git main —
PR #48342, the ONLY native implementation; torch CPU).

Every project gate is C==oracle; the oracle was ported from
reference/inference/modeling_glm5_next.py but had never been run against
actual torch output. This script closes that loop at synthetic scale
(decisive for PORT FIDELITY — the compute graph is scale-independent):

  1. oracle.make_tiny_config() + oracle.write_weights (the m0 fixture
     generator — FP8 128x128 grid intact) at a pinned seed.
  2. The oracle's prefill (f32-faithful mode) on those weights.
  3. The same tensors rewritten as an HF-layout checkpoint
     (model.language_model.* prefix, lm_head top-level): the FP8 pairs
     are DEQUANTIZED to BF16 with the oracle's own fp8_dequant + bf16
     rounding, so HF sees exactly the weights the oracle computes with
     (FP8 storage fidelity is gated bitwise elsewhere — tests/m3g; what
     this anchor validates is the architecture's compute graph). BF16
     rather than FP8 storage because HF's fp8 CPU load path is not the
     reference compute path — SAID HERE per the task instructions.
  4. transformers' Glm5NextForConditionalGeneration forward on the same
     16-token id sequence (no vision input — the text path only; the tiny
     vision tower is built shrunken and left unused).
  5. Compare: max abs/rel logit error, per-position argmax agreement.
     BF16 compute in two different evaluation orders is NOT bitwise — the
     criterion is tight numerical agreement (rel err <= 2e-2, argmax
     identical at every position except the margin-excused tie class).
     KNOWN FRAGILE CLASS (measured): the indexer's selection-boundary
     near-ties (cross-engine bf16 noise flips which pool fills the last
     slot — the m4h fragile-query class, here cross-engine) move
     downstream logits O(1). The gate REDRAWS the fixture seed when any
     DSA layer's selection-boundary margin falls below 0.05 (the
     m4g/m5g router-margin policy applied to the indexer; the measured
     cross-engine index-score noise is ~0.006). A clear-margin argmax
     flip or a large rel error on a redrawn fixture means the oracle port
     has a real discrepancy: rerun with --dump to write both sides'
     per-layer streams (oracle layer_interm block_out_h vs HF
     output_hidden_states) for the sublayer bisect.

Usage: .venv/bin/python tests/m0/check_vs_hf.py [--dump] [--seed N]
Opt-in gate: `make check-hf` (needs torch + transformers @ git main in
.venv — NOT part of the default battery; CI runners don't carry torch).
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402  (tools/oracle.py — the M0 numpy oracle)

SEED = 20260905
N_TOK = 16


def indexer_min_margin(cfg, lim):
    """Min selection-boundary margin over every DSA layer/position: the gap
    between the last selected and the first unselected VALID pool score
    (pre-mask idx_scores + the causal pool-end validity rule). Positions
    where every valid pool is selected have no boundary (margin inf).
    This gate's prefill is always fresh from position 0, so the query
    position of row t is t."""
    kpool = cfg["index_kpool"]
    select_max = cfg["index_topk"] // kpool
    best = np.inf
    for l, im in enumerate(lim):
        if "idx_scores" not in im or im["idx_scores"].shape[1] == 0:
            continue
        sc = im["idx_scores"]                      # [s, n_full] pre-mask
        s, n_full = sc.shape
        pool_end = np.arange(n_full) * kpool + (kpool - 1)
        for t in range(s):
            valid = sc[t][pool_end <= t]
            if valid.size <= 1:
                continue
            k = min(select_max, valid.size)
            if k >= valid.size:
                continue                    # no boundary
            v = np.sort(valid)[::-1]
            best = min(best, float(v[k - 1] - v[k]))
    return best


def build_hf_checkpoint(cfg, stripped_dir, hf_dir):
    """Rewrite the oracle's tiny weights as an HF-layout checkpoint:
    model.language_model.* prefix (lm_head.weight stays top-level), FP8
    pairs dequantized to BF16 with the oracle's own dequant (so HF sees
    exactly the weights the oracle computes with), config.json in the
    REAL glm5_next shape at tiny dims (no quantization_config — BF16
    storage; no MTP block)."""
    os.makedirs(hf_dir, exist_ok=True)
    shards = oracle.ShardSet(stripped_dir)
    recs = []
    weight_map = {}
    for name in sorted(shards.weight_map):
        if name.endswith(".weight_scale_inv"):
            continue
        dt, shape = shards.meta(name)
        if dt == "F8_E4M3":
            codes, scales = shards.fp8(name[:-len(".weight")])
            w = oracle._B(oracle.fp8_dequant(codes, scales, False), False)
            payload = oracle.f32_to_bf16_bytes(w)
            ndt = "BF16"
        else:
            payload = shards.raw(name)
            ndt = dt
        new = (name if name == "lm_head.weight"
               else "model.language_model." + name)
        recs.append((new, ndt, shape, payload))
        weight_map[new] = "hf-00001.safetensors"
    oracle.write_shard(os.path.join(hf_dir, "hf-00001.safetensors"), recs)
    with open(os.path.join(hf_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {"total_size": 0},
                   "weight_map": weight_map}, f, indent=1)

    # config.json: the real shape (reference/config.json) at tiny dims
    ref = json.load(open(os.path.join(ROOT, "reference", "config.json")))
    ref.pop("quantization_config", None)
    L = cfg["num_hidden_layers"]
    tc = ref["text_config"]
    tc.update({
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "num_hidden_layers": L,
        "rms_norm_eps": cfg["rms_norm_eps"],
        "hc_mult": cfg["hc_mult"],
        "hc_sinkhorn_iters": cfg["hc_sinkhorn_iters"],
        "hc_eps": cfg["hc_eps"],
        "first_k_dense_replace": cfg["first_k_dense_replace"],
        "intermediate_size": cfg["intermediate_size"],
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "n_routed_experts": cfg["n_routed_experts"],
        "n_shared_experts": cfg["n_shared_experts"],
        "num_experts_per_tok": cfg["num_experts_per_tok"],
        "routed_scaling_factor": cfg["routed_scaling_factor"],
        "norm_topk_prob": cfg["norm_topk_prob"],
        "n_group": cfg["n_group"],
        "topk_group": cfg["topk_group"],
        "swiglu_limit": cfg["swiglu_limit"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_attention_heads"],
        "q_lora_rank": cfg["q_lora_rank"],
        "kv_lora_rank": cfg["kv_lora_rank"],
        "qk_rope_head_dim": cfg["qk_rope_head_dim"],
        "qk_nope_head_dim": cfg["qk_nope_head_dim"],
        "qk_head_dim": cfg["qk_nope_head_dim"],
        "v_head_dim": cfg["v_head_dim"],
        "index_n_heads": cfg["index_n_heads"],
        "index_head_dim": cfg["index_head_dim"],
        "index_topk": cfg["index_topk"],
        "index_kpool": cfg["index_kpool"],
        "index_kpool_always_select_tail":
            cfg["index_kpool_always_select_tail"],
        "layer_types": list(cfg["layer_types"]),
        "mlp_layer_types": list(cfg["mlp_layer_types"]),
        "indexer_types": list(cfg["indexer_types"]),
        "num_nextn_predict_layers": 0,
        "max_position_embeddings": 4096,
        "eos_token_id": [cfg["vocab_size"] - 1],
        "pad_token_id": cfg["vocab_size"] - 1,
    })
    la = tc["linear_attn_config"]
    la["num_heads"] = cfg["linear_num_heads"]
    la["head_dim"] = cfg["linear_head_dim"]
    la["short_conv_kernel_size"] = cfg["linear_conv_kernel_dim"]
    la["gate_lower_bound"] = cfg["linear_lower_bound"]
    la["kda_layers"] = [i for i, t in enumerate(cfg["layer_types"])
                        if t == "linear_attention"]
    la["full_attn_layers"] = [i for i, t in enumerate(cfg["layer_types"])
                              if t == "deepseek_sparse_attention"]
    # the vision tower is built eagerly but never exercised (text-only
    # input); shrink it to keep the tiny checkpoint tiny
    vc = ref["vision_config"]
    vc.update({"depth": 2, "hidden_size": 64, "intermediate_size": 128,
               "num_heads": 4, "projection_intermediate_size": 128,
               "out_hidden_size": cfg["hidden_size"]})
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(ref, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dump", action="store_true",
                    help="write both sides' per-layer streams (npz) for a "
                         "sublayer bisect on divergence")
    args = ap.parse_args()

    import torch  # deferred: the Makefile target checks deps first
    from transformers import Glm5NextForConditionalGeneration

    cfg = oracle.make_tiny_config()
    ids = oracle._gen_ids(cfg, args.seed, N_TOK)

    with tempfile.TemporaryDirectory(prefix="m0_hf_anchor_") as td:
        stripped = os.path.join(td, "stripped")
        hfdir = os.path.join(td, "hf")

        # --- oracle side (f32-faithful mode — the mode every C gate pins),
        # with the indexer-selection margin redraw (the m4g/m5g router
        # policy applied to the indexer's fragile boundary; see the module
        # docstring). The HF forward runs only for the accepted seed.
        seed = args.seed
        for _ in range(64):
            ids = oracle._gen_ids(cfg, seed, N_TOK)
            oracle.write_weights(cfg, stripped, seed)
            shards = oracle.ShardSet(stripped)
            Ps, top = oracle.load_model_params(shards, cfg)
            lim = []
            logits_o, h_o, _states = oracle.prefill(Ps, top, cfg, ids,
                                                    f64=False,
                                                    layer_interm=lim)
            margin = indexer_min_margin(cfg, lim)
            if margin >= 0.05:
                break
            print(f"  seed {seed}: indexer selection margin "
                  f"{margin:.4g} < 0.05 — redrawing")
            seed += 1
        else:
            print("FAIL: no seed with a safe indexer margin in 64 draws")
            return 1
        print(f"seed {seed}: indexer selection min margin {margin:.4g}")

        # --- HF side (the TRUE implementation)
        build_hf_checkpoint(cfg, stripped, hfdir)
        model = Glm5NextForConditionalGeneration.from_pretrained(
            hfdir, dtype=torch.bfloat16)
        model.eval()
        with torch.no_grad():
            out = model(input_ids=torch.from_numpy(ids)[None, :],
                        output_hidden_states=True)
        logits_h = out.logits[0].float().numpy()
        hidden_h = [hh[0].float().numpy() for hh in out.hidden_states]

    V = cfg["vocab_size"]
    assert logits_o.shape == logits_h.shape == (N_TOK, V), \
        (logits_o.shape, logits_h.shape)

    d = np.abs(logits_h - logits_o)
    scale = np.abs(logits_o).max()
    rel = d.max() / scale
    # argmax agreement, with the margin policy: a flip is excused iff EITHER
    # engine's top1-top2 gap is within the cross-engine noise floor (bf16
    # evaluation-order noise measured at <= 0.06 abs here; 0.1 carries
    # headroom) — the m4h fragile-tie class. A flip with clear margins on
    # BOTH sides is a real divergence. (Random synthetic weights make exact
    # bf16 ties frequent — with real weights ties are rare.)
    MARGIN = 0.1
    flips = 0
    unexcused = 0
    for t in range(N_TOK):
        a_o, a_h = int(np.argmax(logits_o[t])), int(np.argmax(logits_h[t]))
        if a_o == a_h:
            continue
        flips += 1
        g_o = np.sort(logits_o[t])[-1] - np.sort(logits_o[t])[-2]
        g_h = np.sort(logits_h[t])[-1] - np.sort(logits_h[t])[-2]
        excused = g_o <= MARGIN or g_h <= MARGIN
        unexcused += 0 if excused else 1
        print(f"  pos {t}: argmax oracle {a_o} != HF {a_h} "
              f"(gaps oracle {g_o:.4g}, HF {g_h:.4g})"
              + (" [tie class — excused]" if excused else
                 " [CLEAR-MARGIN DIVERGENCE]"))
    print(f"oracle-vs-HF logits: max abs {d.max():.4g}, rel {rel:.4g} "
          f"(scale {scale:.4g})")
    print(f"argmax agreement: {N_TOK - flips}/{N_TOK} exact, "
          f"{flips} tie-class flips excused, {unexcused} clear-margin")
    if args.dump:
        out_npz = os.path.join(HERE, "check_vs_hf_dump.npz")
        np.savez(out_npz, ids=ids, logits_oracle=logits_o,
                 logits_hf=logits_h,
                 oracle_h=np.stack([im["block_out_h"] for im in lim]),
                 **{f"hf_hidden_{i}": hh for i, hh in
                    enumerate(hidden_h)})
        print(f"dump written to {out_npz}")
    if unexcused == 0 and rel <= 2e-2:
        print("MATCH: the numpy oracle is numerically faithful to true-HF "
              "glm5_next (torch CPU, bf16) at synthetic scale")
        return 0
    print("DIVERGE: the oracle port has a real discrepancy — rerun with "
          "--dump and bisect per sublayer; this is a gate-7 finding")
    return 1


if __name__ == "__main__":
    sys.exit(main())
