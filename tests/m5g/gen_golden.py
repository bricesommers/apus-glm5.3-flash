#!/usr/bin/env python3
"""tests/m5g/gen_golden.py — M5 (GLM full-model forward) golden generator.

Pipeline (all deterministic from MASTER_SEED; router-margin re-draws bump
it and regenerate EVERYTHING):

  1. Synthetic tiny-config weights via the M0 oracle's generator
     (tools/oracle.py write_weights), rewritten with the
     "model.language_model." checkpoint prefix as a fake HF source
     checkpoint (config.json + model.safetensors.index.json).
  2. tools/convert.py convert -> the M1 CONTAINER (format v2 manifest +
     coalesced expert slabs) — the C loader under test consumes exactly
     this, including one-pread-per-expert slab reads.
  3. The oracle loads the model BACK FROM THE CONTAINER (ShardSet over
     model.safetensors.index.json — the cross-load M1 promises) and runs
     prefill (68 tokens, 2 KDA chunks) + an 8-step decode chain in
     f32-faithful mode, collecting per-layer interms via the additive
     layer_interm hook (block_out_h, router_idx/biased, idx_topk/scores).
  4. Goldens: per-layer block-output streams h (bf16-valued f32), logits
     per call, final KDA/DSA states, the teacher-forcing selections
     (router idx per MoE layer, indexer top-k per DSA layer — the
     TOLERANCE tier runs the model with these forced; tests/m5g/README.md),
     and the exp probe (the m4g/m4h two-tier host detection).

Near-tie policy: router selections are margin-protected (the whole
fixture is re-drawn until every biased-score top-k boundary gap across
every MoE layer/token/step exceeds ROUTER_MARGIN_MIN = 1e-4 — the m4g
policy). Indexer near-ties are STRUCTURAL (relu-clip zero ties; m4h
showed no re-draw escapes them): the manifest records fragile-query
counts (m4h's SEL_MARGIN rule) and the tolerance tier teacher-forces the
indexer top-k from these goldens instead of skipping — at model level a
flipped selection poisons the whole downstream chain, so skipping is not
an option. The BITWISE tier compares everything (exact ties are
deterministic: stable descending, lower index first, in both numpy and
the C top-k).

Fixtures (tests/m5g/golden/, gitignored):
  manifest.json / manifest.txt   dims, margins, fragile counts
  probe_x.bin / probe_y.bin      f32 exp probe (np.exp values)
  config.json                    flat oracle fixture config (the C parser)
  container/                     the M1 v2 container under test
  src_ckpt/                      (removed after conversion)
  prompt_ids/decodem_*.bin, prefill_logits, decode_logits,
  pre_h_l{L}/dec_h_l{L}, forced_*_{pre,dec}_l{L}, kda_l{L}_*, dsa_l{L}_*
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402  (tools/oracle.py — the M0 numpy oracle)

fnv1a = oracle.fnv1a
bf16_round = oracle.bf16_round

MASTER_SEED = 20260830
PREFILL_LEN = 68           # 2 KDA chunks (64 + 4 tail)
DECODE_STEPS = 8
ROUTER_MARGIN_MIN = 1e-4   # m4g policy (redraw below this)
SEL_MARGIN = 0.02          # m4h fragile-query threshold (informational —
                           # the tolerance tier teacher-forces selections)

DIGESTS = {}


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(OUT, name + ".bin"))
    DIGESTS[name] = fnv1a([arr.tobytes()])


def bf16_codes(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


# ---------------------------------------------------------------------------
# 1+2. source checkpoint + conversion
# ---------------------------------------------------------------------------

def build_source_checkpoint(cfg, seed, src_dir):
    """Oracle-generated weights (stripped names) rewritten with the
    "model.language_model." prefix — a minimal fake HF checkpoint that
    tools/convert.py accepts."""
    tmp = src_dir + ".stripped"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    oracle.write_weights(cfg, tmp, seed)
    os.makedirs(src_dir, exist_ok=True)
    weight_map = {}
    for fname in sorted(os.listdir(tmp)):
        if not fname.endswith(".safetensors"):
            continue
        header, data_start = oracle.read_shard(os.path.join(tmp, fname))
        recs = []
        with open(os.path.join(tmp, fname), "rb") as f:
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                f.seek(data_start + meta["data_offsets"][0])
                payload = f.read(meta["data_offsets"][1]
                                 - meta["data_offsets"][0])
                new = (name if name == "lm_head.weight"
                       else "model.language_model." + name)
                recs.append((new, meta["dtype"], meta["shape"], payload))
                weight_map[new] = fname
        oracle.write_shard(os.path.join(src_dir, fname), recs)
    with open(os.path.join(src_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {"total_size": 0},
                   "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(src_dir, "config.json"), "w") as f:
        json.dump({"num_hidden_layers": cfg["num_hidden_layers"]}, f)
    shutil.rmtree(tmp)


def convert(src_dir, dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "convert.py"),
         "convert", src_dir, dst_dir],
        check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 3. oracle run with interm collection
# ---------------------------------------------------------------------------

def run_oracle(Ps, top, cfg, pids, dids):
    """prefill + decode chain with per-layer interms. Returns a dict with
    logits, per-layer h, states, and the collected interm lists."""
    lim_pre = []
    logits_pre, _, states = oracle.prefill(Ps, top, cfg, pids, f64=False,
                                           layer_interm=lim_pre)
    step_logits = []
    lim_dec = []
    for t in dids:
        lst = []
        lg = oracle.decode_step(Ps, top, cfg, int(t), states, f64=False,
                                layer_interm=lst)
        step_logits.append(lg)
        lim_dec.append(lst)
    return {"logits_pre": logits_pre,
            "logits_dec": np.stack(step_logits),
            "states": states, "lim_pre": lim_pre, "lim_dec": lim_dec}


def router_min_margin(run, cfg):
    """Min biased-score top-k boundary gap over every MoE layer, token,
    and call (prefill + every decode step)."""
    topk = cfg["num_experts_per_tok"]
    best = np.inf
    for call in [run["lim_pre"]] + [lst for lst in run["lim_dec"]]:
        for im in call:
            if "router_scores_biased" not in im:
                continue
            b = np.sort(-im["router_scores_biased"], axis=-1)
            gap = (-b)[:, topk - 1] - (-b)[:, topk]
            best = min(best, float(gap.min()))
    return best


def indexer_fragile(run, cfg, base0=0):
    """m4h fragile-query rule per DSA call (informational — the tolerance
    tier teacher-forces selections). Returns {call_index: n_fragile}."""
    kpool = cfg["index_kpool"]
    itopk = cfg["index_topk"]
    finfo_min = np.finfo(np.float32).min
    out = {}
    calls = [("pre", run["lim_pre"], PREFILL_LEN, base0)]
    calls += [("dec", lst, 1, base0 + PREFILL_LEN + i)
              for i, lst in enumerate(run["lim_dec"])]
    for tag, lim, slen, base in calls:
        for li, im in enumerate(lim):
            if "idx_scores" not in im:
                continue
            scores = im["idx_scores"]
            n = base + slen
            n_full = n // kpool
            if not n_full:
                out[f"{tag}_l{li}"] = 0
                continue
            q_pos = base + np.arange(slen)
            pool_end = np.arange(n_full) * kpool + (kpool - 1)
            valid = pool_end[None, :] <= q_pos[:, None]
            sm = np.where(valid, scores, finfo_min)
            select_k = min(itopk // kpool, n_full)
            nfrag = 0
            for t in range(slen):
                row = -np.sort(-sm[t])
                for j in range(min(select_k, n_full - 1)):
                    if row[j + 1] <= finfo_min:
                        continue
                    if float(row[j] - row[j + 1]) < SEL_MARGIN:
                        nfrag += 1
                        break
            out[f"{tag}_l{li}"] = nfrag
    return out


# ---------------------------------------------------------------------------

def attempt(seed):
    """One full generation attempt; returns (run, cfg, pids, dids)."""
    cfg = oracle.make_tiny_config()
    src = os.path.join(OUT, "src_ckpt")
    if os.path.isdir(src):
        shutil.rmtree(src)
    build_source_checkpoint(cfg, seed, src)
    container = os.path.join(OUT, "container")
    convert(src, container)
    shutil.rmtree(src)

    shards = oracle.ShardSet(container)
    Ps, top = oracle.load_model_params(shards, cfg)
    pids = oracle._gen_ids(cfg, seed + 17 * PREFILL_LEN, PREFILL_LEN)
    dids = oracle._gen_ids(cfg, seed + 24000, DECODE_STEPS)
    run = run_oracle(Ps, top, cfg, pids, dids)
    # determinism (the m0 rerun check at model level)
    l2, _, _ = oracle.prefill(Ps, top, cfg, pids, f64=False)
    assert np.array_equal(l2, run["logits_pre"]), "prefill not deterministic"
    return run, cfg, pids, dids


def main():
    os.makedirs(OUT, exist_ok=True)
    seed = MASTER_SEED
    for _ in range(64):
        run, cfg, pids, dids = attempt(seed)
        margin = router_min_margin(run, cfg)
        if margin >= ROUTER_MARGIN_MIN:
            break
        print(f"  seed {seed}: router margin {margin:.3g} < "
              f"{ROUTER_MARGIN_MIN} — redrawing")
        seed += 1
    else:
        raise RuntimeError("no seed with a safe router margin")
    print(f"  seed {seed}: router min margin {margin:.6g}")
    fragile = indexer_fragile(run, cfg)
    print(f"  indexer fragile queries (informational): {fragile}")

    L = cfg["num_hidden_layers"]
    hc = cfg["hc_mult"]
    dim = cfg["hidden_size"]
    V = cfg["vocab_size"]
    Hk, Dk = cfg["linear_num_heads"], cfg["linear_head_dim"]
    qkv = Hk * Dk
    Hd = cfg["num_attention_heads"]
    qd, vd = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    ID = cfg["index_head_dim"]
    topk = cfg["num_experts_per_tok"]
    width = cfg["index_topk"] + cfg["index_kpool"] - 1
    n_total = PREFILL_LEN + DECODE_STEPS

    # exp probe (host-transcendental detection in the C test)
    px = np.concatenate([
        np.linspace(-90.0, 10.0, 1021, dtype=np.float32),
        np.array([0.0, -0.0, 1e-30], dtype=np.float32),
    ])
    save("probe_x", px)
    save("probe_y", np.exp(px))

    # flat fixture config for the C parser (the oracle's own cfg dict)
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=1)

    save("prompt_ids", pids.astype(np.int32))
    save("decode_ids", dids.astype(np.int32))
    save("prefill_logits", run["logits_pre"].astype(np.float32))
    save("decode_logits", run["logits_dec"].astype(np.float32))

    dsa_layers = [l for l in range(L)
                  if cfg["layer_types"][l] == "deepseek_sparse_attention"]
    moe_layers = [l for l in range(L)
                  if cfg["mlp_layer_types"][l] == "sparse"]
    kda_layers = [l for l in range(L) if l not in dsa_layers]

    for l in range(L):
        save(f"pre_h_l{l}", run["lim_pre"][l]["block_out_h"]
             .astype(np.float32))
        save(f"dec_h_l{l}", np.stack([lst[l]["block_out_h"][0]
                                      for lst in run["lim_dec"]])
             .astype(np.float32))
    for l in dsa_layers:
        save(f"forced_topk_pre_l{l}", run["lim_pre"][l]["idx_topk"]
             .astype(np.int32))
        save(f"forced_topk_dec_l{l}",
             np.stack([lst[l]["idx_topk"][0] for lst in run["lim_dec"]])
             .astype(np.int32))
    for l in moe_layers:
        save(f"forced_idx_pre_l{l}", run["lim_pre"][l]["router_idx"]
             .astype(np.int32))
        save(f"forced_idx_dec_l{l}",
             np.stack([lst[l]["router_idx"][0] for lst in run["lim_dec"]])
             .astype(np.int32))

    # final states (after prefill + the full decode chain), oracle layout
    states = run["states"]
    for l in range(L):
        st = states[l]
        if st.kind == "kda":
            save(f"kda_l{l}_conv", st.conv_state.astype(np.float32))
            save(f"kda_l{l}_rec", st.rec_state.astype(np.float32))
        else:
            save(f"dsa_l{l}_k_cache", st.k_cache.astype(np.float32))
            save(f"dsa_l{l}_v_cache", st.v_cache.astype(np.float32))
            save(f"dsa_l{l}_idx_k", st.idx_k.astype(np.float32))
            save(f"dsa_l{l}_idx_gate", st.idx_gate.astype(np.float32))

    manifest = {
        "seed": seed,
        "router_min_margin": margin,
        "indexer_fragile": fragile,
        "sel_margin": SEL_MARGIN,
        "probe_n": int(px.size),
        "config": {
            "L": L, "dim": dim, "V": V, "hc": hc,
            "Hk": Hk, "Dk": Dk, "qkv": qkv, "Hd": Hd, "qd": qd, "vd": vd,
            "ID": ID, "topk": topk, "width": width,
            "dsa_layers": dsa_layers, "moe_layers": moe_layers,
            "kda_layers": kda_layers,
        },
        "prefill_len": PREFILL_LEN,
        "decode_steps": DECODE_STEPS,
        "n_total": n_total,
        "digests": DIGESTS,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # key=value copy for the C test (no JSON parser needed there)
    mc = manifest["config"]
    lines = [f"probe_n={manifest['probe_n']}",
             f"seed={seed}",
             f"router_min_margin={margin:.9g}",
             f"prefill_len={PREFILL_LEN}",
             f"decode_steps={DECODE_STEPS}",
             f"n_total={n_total}"]
    for k, v in mc.items():
        if isinstance(v, list):
            lines.append(f"cfg_{k}=" + ",".join(str(x) for x in v))
        else:
            lines.append(f"cfg_{k}={v}")
    for k, v in fragile.items():
        lines.append(f"fragile_{k}={v}")
    with open(os.path.join(OUT, "manifest.txt"), "w", newline="") as f:
        f.write("\n".join(lines) + "\n")
    print(f"m5g goldens: {L} layers (KDA {kda_layers}, DSA {dsa_layers}, "
          f"MoE {moe_layers}), prefill {PREFILL_LEN} + {DECODE_STEPS} "
          f"decode steps -> {OUT}")


if __name__ == "__main__":
    main()
