#!/usr/bin/env python3
"""tests/m8g/gen_fixtures.py — M8a (GLM MTP / classic NextN) fixture generator.

Pipeline (all deterministic from MASTER_SEED; router-margin re-draws bump
it and regenerate EVERYTHING — the m5g policy):

  1. Synthetic tiny-config weights via the M0 oracle's generator
     (tools/oracle.py write_weights with mtp=True — the NextN block lands
     at layers.<L>.* with plain residuals, no hc_*), rewritten with the
     "model.language_model." checkpoint prefix as a fake HF source
     checkpoint (config.json num_hidden_layers = L, so convert.py routes
     layers.L.* to the "mtp" shard group).
  2. tools/convert.py convert -> the M1 CONTAINER (format v2 manifest +
     coalesced expert slabs, apus-mtp-* shard group) — the M8b C loader
     under test consumes exactly this.
  3. The oracle loads the model + MTP block BACK FROM THE CONTAINER and
     runs: prefill (12 tokens) + a 2-step decode chain (f32-faithful
     mode), capturing the per-position PINNED hnorm input
     (oracle.MTP_HNORM_INPUT / oracle.MTP_PAIR_LAG — the tools/mtp_pin.py
     verdict on the real container).
  4. MTP goldens: (a) batched true-pair replay via mtp_forward ->
     logits + out_h per replay position; (b) the draft chain seeded from
     the replay's last position (d1 = argmax of the replay's last logits
     row, then mtp_chain over the replay-built state) -> draft ids +
     per-step logits + per-step out_h. Router/indexer selections are
     saved alongside for the M8b tolerance tier (teacher-forcing), like
     m5g's forced_* files.

Two cases (the draft seed comes from a main-model decode step whose top
layer is KDA vs DSA — the C engine's h-out surface differs):
  kda_top: L=2 main layers [KDA, KDA], mlp [dense, sparse] + MTP block
  dsa_top: L=2 main layers [KDA, DSA], mlp [dense, sparse] + MTP block

Fixtures (tests/m8g/golden/<case>/, gitignored):
  config.json            flat oracle fixture config (the C parser)
  container/             the M1 v2 container under test (mtp shard group)
  prompt_ids / decode_ids
  pre_logits / dec_logits
  pair_ids / pair_h      the batched mtp_forward replay input ([s] / [s,dim])
  replay_logits / replay_out_h
  chain_drafts / chain_logits / chain_out_h
  forced_idx_* / forced_topk_*   teacher-forcing selections (tolerance tier)
  probe_x / probe_y      f32 exp probe (host-transcendental detection)
  manifest.json / manifest.txt
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

MASTER_SEED = 20260830
PREFILL_LEN = 12           # 3 full kpool-4 indexer pools
DECODE_STEPS = 2           # the chain seeds from the last decode position
CHAIN_DEPTH = 3            # drafts per chain
ROUTER_MARGIN_MIN = 1e-4   # m4g policy (redraw below this)

CASES = {
    "kda_top": ["linear_attention", "linear_attention"],
    "dsa_top": ["linear_attention", "deepseek_sparse_attention"],
}


def save(gdir, digests, name, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(gdir, name + ".bin"))
    digests[name] = oracle.fnv1a([arr.tobytes()])


# ---------------------------------------------------------------------------
# 1+2. source checkpoint + conversion (m5g conventions)
# ---------------------------------------------------------------------------

def build_source_checkpoint(cfg, seed, src_dir):
    """Oracle-generated weights (stripped names, MTP block included)
    rewritten with the "model.language_model." prefix — a minimal fake HF
    checkpoint that tools/convert.py accepts (num_hidden_layers = L routes
    layers.L.* to the mtp shard group)."""
    tmp = src_dir + ".stripped"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    oracle.write_weights(cfg, tmp, seed, mtp=True)
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
# 3+4. oracle runs + goldens
# ---------------------------------------------------------------------------

def run_case(cfg, seed, case_dir):
    """One case: convert + oracle runs; returns the collected data dict."""
    src = os.path.join(case_dir, "src_ckpt")
    if os.path.isdir(src):
        shutil.rmtree(src)
    build_source_checkpoint(cfg, seed, src)
    container = os.path.join(case_dir, "container")
    convert(src, container)
    shutil.rmtree(src)

    shards = oracle.ShardSet(container)
    Ps, top = oracle.load_model_params(shards, cfg)
    Pm = oracle.load_mtp_params(shards, cfg)

    pids = oracle._gen_ids(cfg, seed + 17 * PREFILL_LEN, PREFILL_LEN)
    dids = oracle._gen_ids(cfg, seed + 24000, DECODE_STEPS)

    # main model: prefill + decode, capturing h per position
    lim_pre = []
    logits_pre, h, states = oracle.prefill(Ps, top, cfg, pids, f64=False,
                                           layer_interm=lim_pre)
    h_rows = [h[i].astype(np.float32) for i in range(PREFILL_LEN)]
    dec_logits = []
    lim_dec = []
    for t in dids:
        lst = []
        lg, h = oracle.model_forward(Ps, top, cfg, [int(t)], states, False,
                                     decode=True, layer_interm=lst)
        for st in states:
            st.pos += 1
        dec_logits.append(lg[0])
        h_rows.append(h[0].astype(np.float32))
        lim_dec.append(lst)
    h_all = np.stack(h_rows)                          # [T, hc, dim]
    tokens = np.concatenate([pids, dids])
    T = tokens.shape[0]

    # pinned hnorm input + pairing (tools/mtp_pin.py verdict)
    cand = oracle.mtp_hnorm_input(h_all, top, cfg, False)
    lag = oracle.MTP_PAIR_LAG

    # (a) batched true-pair replay over positions lo..T-1
    lo = 0 if lag == 0 else 1
    pair_ids = tokens[lo:]
    pair_h = cand[lo - lag:T - lag] if lag else cand[:T]
    mst = oracle.new_mtp_state(cfg)
    im_replay = {}
    replay_logits, replay_out_h = oracle.mtp_forward(
        Pm, top, cfg, pair_ids, pair_h, mst, False, interm=im_replay)

    # (b) draft chain from the replay's last position (the engine flow:
    # d1 = argmax of the replay's last logits row, then mtp_chain over the
    # replay-built state)
    d1 = int(np.argmax(replay_logits[-1]))
    im_chain = []
    drafts, chain_logits, chain_out_h = oracle.mtp_chain(
        Pm, top, cfg, d1, replay_out_h[-1], mst, CHAIN_DEPTH - 1, False,
        interm=im_chain)
    chain_drafts = np.concatenate([[d1], drafts])
    chain_logits = np.concatenate([replay_logits[-1:], chain_logits])
    chain_out_h = np.concatenate([replay_out_h[-1:], chain_out_h])

    # determinism: fresh states, identical results (the m0 rerun check)
    mst2 = oracle.new_mtp_state(cfg)
    lg2, oh2 = oracle.mtp_forward(Pm, top, cfg, pair_ids, pair_h, mst2,
                                  False)
    assert np.array_equal(lg2, replay_logits)
    assert np.array_equal(oh2, replay_out_h)

    return {"tokens": tokens, "h_all": h_all, "cand": cand, "lag": lag,
            "logits_pre": logits_pre, "logits_dec": np.stack(dec_logits),
            "lim_pre": lim_pre, "lim_dec": lim_dec,
            "pair_ids": pair_ids, "pair_h": pair_h,
            "replay_logits": replay_logits, "replay_out_h": replay_out_h,
            "im_replay": im_replay, "im_chain": im_chain,
            "chain_drafts": chain_drafts, "chain_logits": chain_logits,
            "chain_out_h": chain_out_h}


def min_router_margin(run, cfg):
    """Min biased-score top-k boundary gap over EVERY MoE call (main MoE
    layers prefill + decode, the MTP replay rows, the MTP chain steps) —
    the m4g/m5g re-draw policy."""
    topk = cfg["num_experts_per_tok"]
    best = np.inf
    dicts = [im for im in run["lim_pre"]]
    for lst in run["lim_dec"]:
        dicts.extend(lst)
    dicts.append(run["im_replay"])
    dicts.extend(run["im_chain"] or [])
    for im in dicts:
        if im is None or "router_scores_biased" not in im:
            continue
        b = np.sort(-im["router_scores_biased"], axis=-1)
        gap = (-b)[:, topk - 1] - (-b)[:, topk]
        best = min(best, float(gap.min()))
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    for case, layer_types in CASES.items():
        seed = MASTER_SEED
        for _ in range(64):
            cfg = oracle.make_tiny_config(
                num_hidden_layers=len(layer_types),
                layer_types=layer_types,
                mlp_layer_types=["dense"] + ["sparse"]
                * (len(layer_types) - 1))
            case_dir = os.path.join(OUT, case)
            if os.path.isdir(case_dir):
                shutil.rmtree(case_dir)
            os.makedirs(case_dir)
            run = run_case(cfg, seed, case_dir)
            margin = min_router_margin(run, cfg)
            if margin >= ROUTER_MARGIN_MIN:
                break
            print(f"  {case} seed {seed}: router margin {margin:.3g} < "
                  f"{ROUTER_MARGIN_MIN} — redrawing")
            seed += 1
        else:
            raise RuntimeError(f"{case}: no seed with a safe router margin")
        print(f"  {case} seed {seed}: router min margin {margin:.6g}")

        L = cfg["num_hidden_layers"]
        dim = cfg["hidden_size"]
        V = cfg["vocab_size"]
        hc = cfg["hc_mult"]
        topk = cfg["num_experts_per_tok"]
        width = cfg["index_topk"] + cfg["index_kpool"] - 1
        T = PREFILL_LEN + DECODE_STEPS
        s = run["pair_ids"].shape[0]

        digests = {}
        with open(os.path.join(case_dir, "config.json"), "w") as f:
            json.dump({k: v for k, v in cfg.items()}, f, indent=1)

        # exp probe (host-transcendental detection in the C test, m5g)
        px = np.concatenate([
            np.linspace(-90.0, 10.0, 1021, dtype=np.float32),
            np.array([0.0, -0.0, 1e-30], dtype=np.float32),
        ])
        save(case_dir, digests, "probe_x", px)
        save(case_dir, digests, "probe_y", np.exp(px))

        save(case_dir, digests, "prompt_ids",
             run["tokens"][:PREFILL_LEN].astype(np.int32))
        save(case_dir, digests, "decode_ids",
             run["tokens"][PREFILL_LEN:].astype(np.int32))
        save(case_dir, digests, "pre_logits",
             run["logits_pre"].astype(np.float32))
        save(case_dir, digests, "dec_logits",
             run["logits_dec"].astype(np.float32))
        save(case_dir, digests, "pair_ids", run["pair_ids"].astype(np.int32))
        save(case_dir, digests, "pair_h", run["pair_h"].astype(np.float32))
        save(case_dir, digests, "replay_logits",
             run["replay_logits"].astype(np.float32))
        save(case_dir, digests, "replay_out_h",
             run["replay_out_h"].astype(np.float32))
        save(case_dir, digests, "chain_drafts",
             run["chain_drafts"].astype(np.int32))
        save(case_dir, digests, "chain_logits",
             run["chain_logits"].astype(np.float32))
        save(case_dir, digests, "chain_out_h",
             run["chain_out_h"].astype(np.float32))

        # teacher-forcing selections (tolerance tier, m5g conventions)
        for l in range(L):
            if cfg["mlp_layer_types"][l] == "sparse":
                save(case_dir, digests, f"forced_idx_pre_l{l}",
                     run["lim_pre"][l]["router_idx"].astype(np.int32))
                save(case_dir, digests, f"forced_idx_dec_l{l}",
                     np.stack([lst[l]["router_idx"][0]
                               for lst in run["lim_dec"]]).astype(np.int32))
            if cfg["layer_types"][l] == "deepseek_sparse_attention":
                save(case_dir, digests, f"forced_topk_pre_l{l}",
                     run["lim_pre"][l]["idx_topk"].astype(np.int32))
                save(case_dir, digests, f"forced_topk_dec_l{l}",
                     np.stack([lst[l]["idx_topk"][0]
                               for lst in run["lim_dec"]]).astype(np.int32))
        save(case_dir, digests, "forced_idx_mtp_replay",
             run["im_replay"]["router_idx"].astype(np.int32))
        save(case_dir, digests, "forced_topk_mtp_replay",
             run["im_replay"]["idx_topk"].astype(np.int32))
        if run["im_chain"]:
            save(case_dir, digests, "forced_idx_mtp_chain",
                 np.stack([im["router_idx"][0]
                           for im in run["im_chain"]]).astype(np.int32))
            save(case_dir, digests, "forced_topk_mtp_chain",
                 np.stack([im["idx_topk"][0]
                           for im in run["im_chain"]]).astype(np.int32))

        manifest = {
            "seed": seed,
            "case": case,
            "hnorm_input": oracle.MTP_HNORM_INPUT,
            "pair_lag": run["lag"],
            "router_min_margin": margin,
            "probe_n": int(px.size),
            "config": {"L": L, "dim": dim, "V": V, "hc": hc,
                       "topk": topk, "width": width},
            "prefill_len": PREFILL_LEN,
            "decode_steps": DECODE_STEPS,
            "n_tokens": T,
            "replay_len": s,
            "chain_depth": CHAIN_DEPTH,
            "digests": digests,
        }
        with open(os.path.join(case_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)

        # key=value copy for the C test (no JSON parser needed there, m5g)
        lines = [f"case={case}",
                 f"seed={seed}",
                 f"hnorm_input={oracle.MTP_HNORM_INPUT}",
                 f"pair_lag={run['lag']}",
                 f"router_min_margin={margin:.9g}",
                 f"probe_n={int(px.size)}",
                 f"prefill_len={PREFILL_LEN}",
                 f"decode_steps={DECODE_STEPS}",
                 f"n_tokens={T}",
                 f"replay_len={s}",
                 f"chain_depth={CHAIN_DEPTH}"]
        for k, v in manifest["config"].items():
            lines.append(f"cfg_{k}={v}")
        with open(os.path.join(case_dir, "manifest.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  {case}: L={L} ({layer_types}), prefill {PREFILL_LEN} + "
              f"{DECODE_STEPS} decode, replay {s}, chain {CHAIN_DEPTH} -> "
              f"{case_dir}")


if __name__ == "__main__":
    main()
