#!/usr/bin/env python3
"""tools/measure_glocality.py — P1 GLM router-locality measurements: the
glm5_next port of the base repo's ARCHITECTURE.md §8 plan
(../Apus/tools/measure_router_locality.py, V4 engine). These numbers set
the P2 tiering defaults (APUS_GEXPERT_CACHE_MB, APUS_GPILOT_K).

Dump production (read-only instrumentation in c/apus.c; the C run is
usually launched by hand as a background job — 30–90 min at real scale):

  APUS_GEXPERT_CACHE_MB=16384 bin/apus_metal run \
      --model weights/glm-5.3-flash --tiered --metal --greedy --quiet \
      --prompt "..." --max-tokens 64 \
      --measure-locality docs/locality/glm_locality_dump.ndjson

Then the analysis (this tool's main job):

  .venv/bin/python tools/measure_glocality.py \
      --model weights/glm-5.3-flash \
      --dump docs/locality/glm_locality_dump.ndjson \
      --out docs/locality/glm_report.json

Dump format (ndjson, same shape as the V4 dump): one {"type":"A"} record
per token per SPARSE layer with the ACTUAL routed top-8 set (the M6 routed
hook), one {"type":"P"} record per token per predictable layer with the
pilot's predicted top-24 set for layer l+1 (apus_gpilot_predict from the
layer-l post-attention stream), plus "ids" (prompt) and "gen" lines.

Differences vs the V4 model that matter here:
  - NO hash layers (V4 measurement 5 is N/A).
  - ALL 42 sparse layers (3..44) sit after EITHER attention kind — the
    base's csa/hca source-type split becomes kda/dsa: layers with
    i % 4 == 3 in 3..43 are DSA (11), the rest KDA (31, of the sparse
    layers; layers 0-2 are dense MLP, no router). Both layer kinds share
    the same MoE block, so the split is by ATTENTION type of the source
    layer (the pilot predicts layer l+1 from layer l's post-attention
    stream) and of the MoE-carrying layer itself.

Measurements:
  1. pilot recall curve: |pred top-N ∩ actual top-8| / 8 for
     N in {6,8,12,16,24} — overall and per source/target attention kind.
  2. temporal reuse: |A(pos) ∩ A(pos+T)| / 8, T in {1,2,4,8}, overall and
     per layer kind.
  3. cross-layer coupling: P(e_{L+d}=j | e_L=i), d in {1,2} (mean top-1
     probability per layer; sparse artifacts coupling_d{d}.json next to
     the report).
  4. per-layer expert frequency + Zipf fit; pin-candidate coverage at
     5/10/20% of E pinned.
  5. cache sizing: per-layer LRU simulation over the recorded decode
     sequence (the gcache policy class), S in {2,4,8,16,32} payloads per
     layer -> hit rate; MB = S x 42 layers x payload_bytes. This is the
     direct P2 cache-budget input.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

RECALL_N = [6, 8, 12, 16, 24]
REUSE_T = [1, 2, 4, 8]
COUPLE_D = [1, 2]
PIN_FRAC = [0.05, 0.10, 0.20]
CACHE_S = [2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# model config (glm5_next: text_config nest or the flat oracle fixture)
# ---------------------------------------------------------------------------

def load_model_cfg(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    n_layers = tc["num_hidden_layers"]
    n_dense = tc.get("first_k_dense_replace", 3)
    # attention kind per layer: the engine's i%4==3 rule (c/gmodel.h
    # config parser reproduces the explicit 45-layer DSA list this way)
    layer_kind = []
    for i in range(n_layers):
        if i < n_dense:
            layer_kind.append("dense")
        elif i % 4 == 3:
            layer_kind.append("dsa")
        else:
            layer_kind.append("kda")
    return {
        "n_layers": n_layers,
        "E": tc["n_routed_experts"],
        "topk": tc["num_experts_per_tok"],
        "n_dense": n_dense,
        "layer_kind": layer_kind,     # attention kind; MoE on all non-dense
        "sparse_layers": [i for i in range(n_layers)
                          if layer_kind[i] != "dense"],
        "raw": tc,
    }


# ---------------------------------------------------------------------------
# dump loading
# ---------------------------------------------------------------------------

def load_dump(path):
    ids_at = {}
    A = defaultdict(dict)      # layer -> pos -> eids (actual)
    P = defaultdict(dict)      # layer -> pos -> eids (pilot top-24)
    for line in open(path):
        e = json.loads(line)
        t = e["type"]
        if t == "ids":
            for i, tok in enumerate(e["ids"]):
                ids_at[e["pos0"] + i] = tok
        elif t == "gen":
            ids_at[e["pos"]] = e["id"]
        elif t == "A":
            A[e["layer"]][e["pos"]] = e["eids"]
        elif t == "P":
            P[e["layer"]][e["pos"]] = e["eids"]
    return ids_at, A, P


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def m1_recall(mcfg, A, P):
    """Pilot recall |pred[:N] ∩ actual| / topk — overall, per source-layer
    attention kind (the layer whose post-attention stream the prediction
    was computed from) and per MoE-layer attention kind."""
    topk = mcfg["topk"]
    by_src = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_tgt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    overall = defaultdict(lambda: [0, 0])
    for layer, per_pos in A.items():
        if layer not in P:
            continue
        src = mcfg["layer_kind"][layer - 1]
        tgt = mcfg["layer_kind"][layer]
        for pos, a in per_pos.items():
            pred = P[layer].get(pos)
            if not pred:
                continue
            for N in RECALL_N:
                if N > len(pred):
                    continue
                h = sum(1 for e in a if e in pred[:N])
                by_src[src][N][0] += h
                by_src[src][N][1] += len(a)
                by_tgt[tgt][N][0] += h
                by_tgt[tgt][N][1] += len(a)
                overall[N][0] += h
                overall[N][1] += len(a)

    def ratios(d):
        return {N: (v[0] / v[1] if v[1] else None)
                for N, v in sorted(d.items())}

    return {
        "topk": topk,
        "overall": ratios(overall),
        "per_source_kind": {k: ratios(v) for k, v in by_src.items()},
        "per_target_kind": {k: ratios(v) for k, v in by_tgt.items()},
    }


def m2_temporal_reuse(mcfg, A):
    """|A(pos) ∩ A(pos+T)| / topk — overall and per layer kind."""
    topk = mcfg["topk"]
    per_T = {T: [] for T in REUSE_T}
    per_kind_T = defaultdict(lambda: {T: [] for T in REUSE_T})
    for layer, per_pos in A.items():
        kind = mcfg["layer_kind"][layer]
        pos_set = set(per_pos)
        for p in pos_set:
            for T in REUSE_T:
                if p + T in pos_set:
                    r = len(set(per_pos[p]) & set(per_pos[p + T])) / topk
                    per_T[T].append(r)
                    per_kind_T[kind][T].append(r)
    return {
        "overall": {T: (float(np.mean(v)) if v else None)
                    for T, v in per_T.items()},
        "per_kind": {k: {T: (float(np.mean(v)) if v else None)
                         for T, v in d.items()}
                     for k, d in per_kind_T.items()},
    }


def m3_coupling(mcfg, A, out_dir):
    """P(e_{L+d}=j | e_L=i) co-occurrence; per-layer mean top-1 probability
    is the summary, the full top-8 successor lists are the artifacts."""
    summary = {}
    artifacts = []
    for d in COUPLE_D:
        per_layer = []
        for L in mcfg["sparse_layers"]:
            if L + d not in A or L not in A:
                continue
            counts = defaultdict(int)
            for pos, a in A[L].items():
                b = A[L + d].get(pos)
                if not b:
                    continue
                for i in a:
                    for j in b:
                        counts[(i, j)] += 1
            succ = defaultdict(list)
            tot_i = defaultdict(int)
            for (i, j), c in counts.items():
                succ[i].append((j, c))
                tot_i[i] += c
            succ = {i: sorted(v, key=lambda x: (-x[1], x[0]))[:8]
                    for i, v in succ.items()}
            p1 = [v[0][1] / tot_i[i] for i, v in succ.items() if v]
            per_layer.append({
                "layer": L, "d": d,
                "kind": mcfg["layer_kind"][L],
                "mean_p_top1": float(np.mean(p1)) if p1 else None,
                "top_j_given_i": {str(i): v for i, v in succ.items()},
            })
        path = os.path.join(out_dir, f"glm_coupling_d{d}.json")
        with open(path, "w") as f:
            json.dump({"d": d, "E": mcfg["E"], "layers": per_layer}, f)
        artifacts.append(path)
        vals = [x["mean_p_top1"] for x in per_layer
                if x["mean_p_top1"] is not None]
        summary[d] = {
            "per_layer": [x["mean_p_top1"] for x in per_layer],
            "mean": float(np.mean(vals)) if vals else None,
        }
    return summary, artifacts


def m4_frequency_zipf(mcfg, A):
    """Per-layer expert frequency, Zipf slope, pin coverage at 5/10/20%."""
    E = mcfg["E"]
    per_layer = {}
    for layer, per_pos in A.items():
        cnt = np.zeros(E)
        for a in per_pos.values():
            for e in a:
                cnt[e] += 1
        total = cnt.sum()
        nz = np.sort(cnt[cnt > 0])[::-1]
        ranks = np.arange(1, len(nz) + 1)
        slope = None
        if len(nz) > 2:
            s, _ = np.polyfit(np.log(ranks), np.log(nz), 1)
            slope = float(-s)
        pins = {}
        for frac in PIN_FRAC:
            npin = max(1, round(E * frac))
            pins[f"{int(frac*100)}%"] = (
                float(np.sort(cnt)[::-1][:npin].sum() / total)
                if total else None)
        per_layer[str(layer)] = {
            "kind": mcfg["layer_kind"][layer],
            "zipf_slope": slope,
            "pin_coverage": pins,
            "top_counts": [int(x) for x in np.sort(cnt)[::-1][:16]],
        }
    return per_layer


# ---------------------------------------------------------------------------

def m5_cache_sim(mcfg, A, P, n_prompt):
    """Per-layer LRU simulation over the DECODE sequence (positions >=
    n_prompt) — the gcache policy class (per-layer LRU; the global byte
    budget split evenly across the 42 sparse layers). For S payloads per
    layer: plain-LRU hit rate, and LRU+pilot (a touch also counts as a hit
    when the expert is in the pilot's top-12 prediction for that layer and
    token — the optimistic prefetch-coverage bound)."""
    tc = mcfg["raw"]
    payload_bytes = 3 * tc["moe_intermediate_size"] * tc["hidden_size"] * 2
    n_sparse = len(mcfg["sparse_layers"])
    rows = {}
    for S in CACHE_S:
        hits = misses = phits = 0
        for layer in mcfg["sparse_layers"]:
            per_pos = A.get(layer)
            if not per_pos:
                continue
            preds = P.get(layer, {})
            lru = []
            for pos in sorted(per_pos):
                if pos < n_prompt:
                    # prefill warms the LRU but is not scored
                    for e in per_pos[pos]:
                        if e in lru:
                            lru.remove(e)
                        lru.append(e)
                        del lru[:-S]
                    continue
                pred12 = set(preds.get(pos, [])[:12])
                for e in per_pos[pos]:
                    if e in lru:
                        hits += 1
                        lru.remove(e)
                        lru.append(e)
                        phits += 1
                    else:
                        misses += 1
                        lru.append(e)
                        del lru[:-S]
                        if e in pred12:
                            phits += 1
                del lru[:-S]
        tot = hits + misses
        mb = S * n_sparse * payload_bytes / 1048576.0
        rows[str(S)] = {
            "payloads_per_layer": S,
            "cache_mb": round(mb),
            "lru_hit_rate": (hits / tot) if tot else None,
            "lru_plus_pilot12_hit_rate": (phits / tot) if tot else None,
            "decode_touches": tot,
        }
    return {"payload_bytes": payload_bytes,
            "n_sparse_layers": n_sparse,
            "note": "per-layer LRU over the decode sequence (prefill warms "
                    "but is not scored); LRU+pilot12 = a missed touch that "
                    "is in the pilot top-12 prediction counts as a hit "
                    "(optimistic prefetch bound)",
            "sizes": rows}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="model dir (config.json is read for the layer map)")
    ap.add_argument("--dump", required=True,
                    help="the --measure-locality ndjson dump")
    ap.add_argument("--out", required=True, help="report JSON path")
    args = ap.parse_args()

    mcfg = load_model_cfg(args.model)
    ids_at, A, P = load_dump(args.dump)
    n_a = sum(len(v) for v in A.values())
    n_p = sum(len(v) for v in P.values())
    # prompt/decode split from the ids/gen records (self-contained dump)
    ids_rec = None
    n_gen = 0
    for line in open(args.dump):
        e = json.loads(line)
        if e["type"] == "ids":
            ids_rec = len(e["ids"])
        elif e["type"] == "gen":
            n_gen += 1

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    recall = m1_recall(mcfg, A, P)
    reuse = m2_temporal_reuse(mcfg, A)
    coupling, coupling_files = m3_coupling(mcfg, A, out_dir)
    zipf = m4_frequency_zipf(mcfg, A)
    cachesim = m5_cache_sim(mcfg, A, P, ids_rec or 0)

    report = {
        "model": args.model,
        "model_type": "glm5_next",
        "prompt_tokens": ids_rec,
        "decode_tokens": n_gen,
        "dump": {"path": args.dump, "A_sets": n_a, "P_sets": n_p},
        "layer_map": {"n_layers": mcfg["n_layers"],
                      "sparse": mcfg["sparse_layers"],
                      "dsa": [i for i, k in enumerate(mcfg["layer_kind"])
                              if k == "dsa"],
                      "note": "all sparse layers carry the same MoE; the "
                              "kda/dsa split is the ATTENTION kind"},
        "m1_pilot_recall": recall,
        "m2_temporal_reuse": reuse,
        "m3_coupling": coupling,
        "m4_frequency_zipf": zipf,
        "m5_cache_sim": cachesim,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    # --- console summary
    print("=" * 72)
    print(f"GLM router-locality report: {args.model}")
    print(f"dump: {n_a} actual sets, {n_p} predicted sets "
          f"({ids_rec} prompt + {n_gen} decode tokens)")
    print("=" * 72)
    print(f"\n[1] pilot recall (top-{recall['topk']}):")
    hdr = "  source     " + "".join(f"  N={N:<3}" for N in RECALL_N)
    print(hdr)

    def row(name, d):
        print(f"  {name:<10} " + "".join(
            f"  {d.get(N):.3f}" if d.get(N) is not None else "     -"
            for N in RECALL_N))

    for k, d in sorted(recall["per_source_kind"].items()):
        row(f"src-{k}", d)
    for k, d in sorted(recall["per_target_kind"].items()):
        row(f"tgt-{k}", d)
    row("overall", recall["overall"])
    print("\n[2] temporal reuse |A(pos) ∩ A(pos+T)|/topk:")
    for k, d in [("overall", reuse["overall"])] + \
                sorted(reuse["per_kind"].items()):
        print(f"  {k:<8} " + "  ".join(
            f"T={T}: {v:.3f}" if v is not None else f"T={T}: -"
            for T, v in d.items()))
    print("\n[3] cross-layer coupling (mean P(top-1 j | i)):")
    for d, s in sorted(coupling.items()):
        print(f"  d={d}: mean {s['mean']:.3f}" if s["mean"] is not None
              else f"  d={d}: -")
    print(f"  artifacts: {', '.join(coupling_files)}")
    print("\n[4] frequency/Zipf (slope; pin coverage 5/10/20%):")
    for layer, z in sorted(zipf.items(), key=lambda x: int(x[0])):
        cov = z["pin_coverage"]
        if z["zipf_slope"] is not None:
            print(f"  L{layer:>3} ({z['kind']}): slope={z['zipf_slope']:.3f} "
                  f"cov={cov['5%']:.3f}/{cov['10%']:.3f}/{cov['20%']:.3f}")
    print("\n[5] cache sizing (per-layer LRU over decode; MB = S x "
          f"{cachesim['n_sparse_layers']} x "
          f"{cachesim['payload_bytes'] // 1048576} MiB):")
    for s, r in sorted(cachesim["sizes"].items(), key=lambda x: int(x[0])):
        lru = r["lru_hit_rate"]
        lp = r["lru_plus_pilot12_hit_rate"]
        print(f"  S={r['payloads_per_layer']:>2} ({r['cache_mb']:>6} MB): "
              f"lru {lru:.3f}" if lru is not None else f"  S={s}: -",
              end="")
        print(f"  lru+pilot12 {lp:.3f}" if lp is not None else "")
    print(f"\nreport: {args.out}")

if __name__ == "__main__":
    main()
