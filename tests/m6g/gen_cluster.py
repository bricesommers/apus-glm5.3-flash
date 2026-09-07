#!/usr/bin/env python3
"""tests/m6g/gen_cluster.py — M6g locality fixtures: two tiny-config E=32
containers (no oracle goldens — the m6g gate measures cache/pilot
MACHINERY on them: hit rates, prefetch coverage, pilot recall; numerics
are gated separately on the m5g fixture).

  cluster_container/   router gates + embeddings constructed with cluster
                       structure (4 clusters x 8 experts): embeddings and
                       every sparse layer's router gate rows share the same
                       64-dim block basis, so hidden states stay cluster-
                       flavored across the stack — dL=1 pilot predictions
                       land in the right cluster and consecutive same-
                       cluster tokens reuse experts (temporal locality).
  random_container/    the SAME config with pure random oracle weights —
                       the machinery-validation control (expected pilot
                       recall ~ pilot_k/E; see the base project's
                       tools/measure_router_locality.py caveat: random
                       weights have near-uniform routing).
  workload.bin         prompt (16 tokens, 4 cluster runs of 4) + decode
                       (64 tokens, 8 cluster runs of 8) ids
  manifest.txt         dims for the C gate

Pipeline (deterministic from SEED): oracle.write_weights -> (cluster only)
rewrite embed_tokens.weight + layers.*.mlp.gate.weight with the cluster
structure -> m5g-style fake HF source checkpoint (prefix re-added) ->
tools/convert.py convert (the REAL converter, v2 manifest + slabs).
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

SEED = 20260830
N_EXPERTS = 32
N_CLUSTERS = 4
EMBED_SCALE = 8.0
GATE_SCALE = 0.10
GATE_NOISE = 0.05
PROMPT_RUNS = [(0, 4), (1, 4), (2, 4), (3, 4)]        # 16 tokens
DECODE_RUNS = [(c, 8) for c in (0, 1, 2, 3) * 2]      # 64 tokens


def bf16_codes(x):
    return (oracle.bf16_round(x).view(np.uint32)
            >> np.uint32(16)).astype(np.uint16)


def cluster_basis(c, dim):
    b = np.zeros(dim, dtype=np.float32)
    per = dim // N_CLUSTERS
    b[c * per:(c + 1) * per] = 1.0
    return b


def build(cfg, clustered, src_dir):
    tmp = src_dir + ".stripped"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    oracle.write_weights(cfg, tmp, SEED)
    dim = cfg["hidden_size"]
    V = cfg["vocab_size"]
    E = cfg["n_routed_experts"]
    rng = np.random.default_rng(SEED + 777)
    moe_layers = [l for l in range(cfg["num_hidden_layers"])
                  if cfg["mlp_layer_types"][l] == "sparse"]
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
                if clustered and name == "embed_tokens.weight":
                    rows = np.zeros((V, dim), dtype=np.float32)
                    for i in range(V):
                        rows[i] = (EMBED_SCALE
                                   * cluster_basis(i % N_CLUSTERS, dim)
                                   + 0.05 * rng.standard_normal(dim))
                    payload = bf16_codes(rows).tobytes()
                elif clustered and any(
                        name == f"layers.{l}.mlp.gate.weight"
                        for l in moe_layers):
                    rows = np.zeros((E, dim), dtype=np.float32)
                    for e in range(E):
                        u = (rng.standard_normal(dim)
                             * cluster_basis(e % N_CLUSTERS, dim))
                        u /= np.linalg.norm(u)
                        rows[e] = (GATE_SCALE
                                   * cluster_basis(e % N_CLUSTERS, dim)
                                   + GATE_NOISE * u)
                    payload = bf16_codes(rows).tobytes()
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


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = oracle.make_tiny_config(n_routed_experts=N_EXPERTS)
    for name, clustered in (("cluster_container", True),
                            ("random_container", False)):
        src = os.path.join(OUT, name + ".src")
        if os.path.isdir(src):
            shutil.rmtree(src)
        build(cfg, clustered, src)
        convert(src, os.path.join(OUT, name))
        shutil.rmtree(src)
        print(f"  {name}: converted")

    # workload: cluster runs (ids encode their cluster via id % N_CLUSTERS)
    rng = np.random.default_rng(SEED + 555)
    def run_ids(runs):
        ids = []
        for c, n in runs:
            for _ in range(n):
                v = int(rng.integers(0, cfg["vocab_size"] // N_CLUSTERS))
                ids.append(v * N_CLUSTERS + c)
        return np.array(ids, dtype=np.int32)

    pids = run_ids(PROMPT_RUNS)
    dids = run_ids(DECODE_RUNS)
    pids.tofile(os.path.join(OUT, "cluster_prompt.bin"))
    dids.tofile(os.path.join(OUT, "cluster_decode.bin"))

    # flat fixture config for the C parser
    with open(os.path.join(OUT, "cluster_config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=1)

    with open(os.path.join(OUT, "cluster_manifest.txt"), "w", newline="") as f:
        f.write("\n".join([
            f"seed={SEED}",
            f"cfg_L={cfg['num_hidden_layers']}",
            f"cfg_dim={cfg['hidden_size']}",
            f"cfg_V={cfg['vocab_size']}",
            f"cfg_hc={cfg['hc_mult']}",
            f"cfg_E={cfg['n_routed_experts']}",
            f"cfg_topk={cfg['num_experts_per_tok']}",
            f"cfg_clusters={N_CLUSTERS}",
            f"n_prompt={len(pids)}",
            f"n_decode={len(dids)}",
        ]) + "\n")
    print(f"m6g locality fixtures: E={N_EXPERTS}, "
          f"{N_CLUSTERS} clusters, prompt {len(pids)} + decode "
          f"{len(dids)} -> {OUT}")


if __name__ == "__main__":
    main()
