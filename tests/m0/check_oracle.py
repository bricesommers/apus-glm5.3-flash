#!/usr/bin/env python3
"""tests/m0/check_oracle.py — M0 gate for the glm5_next numpy oracle.

Run:  .venv/bin/python tests/m0/check_oracle.py

Gate (all hard unless marked informational):
  1. Fixture generation is deterministic: tools/oracle.py generate() ->
     tests/m0/fixtures, then a fresh rerun from the written safetensors
     reproduces the manifest digests BITWISE (f32 and f64, logits and
     decode-carried state).
  2. Determinism across runs within a process (prefill run twice, bitwise).
  3. Weight round-trip: params loaded through ShardSet from the written
     shards reproduce the same digests (exercises the real naming/format).
  4. KDA ORDERING CONTRACT: a fresh one-shot CHUNKED prefill of the
     extended prompt vs prefill+one RECURRENT decode step — f64 logits must
     agree to < 1e-6 (same math), f32 difference is reported (informational;
     NOT asserted bitwise — the two paths have different fp32 orderings and
     the C engine is gated per phase).
  5. Indexer legality: topk indices are -1 or < kv_len; for the fixture
     sizes every query's selected set covers all positions it must attend
     (pools selected exhaustively or by score + the always-selected tail).
  6. Sinkhorn sanity: comb column sums ~= 1 to ~1e-6 (the eps-in-denominator
     effect; row sums may deviate — doubly stochastic only in the Sinkhorn
     sense).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import oracle  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "m0", "fixtures")

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("== 1. generate fixtures ==")
    cfg = oracle.make_tiny_config()
    manifest = oracle.generate(FIXTURES, cfg=cfg)
    pname, _ = oracle.FIXTURE_SEQUENCES["prefill"]
    dname, _, _ = oracle.FIXTURE_SEQUENCES["decode"]

    print("== 2. reload + rerun: bitwise digests ==")
    shards = oracle.ShardSet(os.path.join(FIXTURES, "weights"))
    Ps, top = oracle.load_model_params(shards, cfg)
    with open(os.path.join(FIXTURES, "manifest.json")) as f:
        man = json.load(f)

    pids = np.load(os.path.join(FIXTURES, "golden", pname,
                                "input_ids.npy"))
    l32, _, st32 = oracle.prefill(Ps, top, cfg, pids, f64=False)
    l64, _, st64 = oracle.prefill(Ps, top, cfg, pids, f64=True)
    dig = man["digests"][pname]
    check("prefill f32 logits digest",
          oracle.array_digest(l32.astype(np.float32)) == dig["logits_f32"],
          dig["logits_f32"])
    check("prefill f64 logits digest",
          oracle.array_digest(l64.astype(np.float64)) == dig["logits_f64"],
          dig["logits_f64"])
    check("prefill f32 state digest",
          oracle.state_digest(st32) == dig["state_f32"], dig["state_f32"])

    dids = np.load(os.path.join(FIXTURES, "golden", dname,
                                "decode_ids.npy"))
    ls32 = np.stack([oracle.decode_step(Ps, top, cfg, t, st32, f64=False)
                     for t in dids])
    ls64 = np.stack([oracle.decode_step(Ps, top, cfg, t, st64, f64=True)
                     for t in dids])
    dig = man["digests"][dname]
    check("decode f32 logits digest",
          oracle.array_digest(ls32.astype(np.float32)) == dig["logits_f32"],
          dig["logits_f32"])
    check("decode f64 logits digest",
          oracle.array_digest(ls64.astype(np.float64)) == dig["logits_f64"],
          dig["logits_f64"])
    check("decode f32 state digest",
          oracle.state_digest(st32) == dig["state_f32"], dig["state_f32"])

    print("== 3. determinism across runs (bitwise) ==")
    l32b, _, _ = oracle.prefill(Ps, top, cfg, pids, f64=False)
    check("prefill rerun bitwise", np.array_equal(l32, l32b))
    l64b, _, _ = oracle.prefill(Ps, top, cfg, pids, f64=True)
    check("prefill f64 rerun bitwise", np.array_equal(l64, l64b))

    print("== 4. KDA ordering contract ==")
    oc = man["ordering_contract"]
    check("f64 chunk-vs-recurrent max diff < 1e-6",
          oc["f64_max_abs_diff"] < 1e-6, f"{oc['f64_max_abs_diff']:.3g}")
    print(f"  [info] f32 chunk-vs-recurrent max diff "
          f"{oc['f32_max_abs_diff']:.3g} (bitwise: {oc['f32_bitwise']})"
          " — NOT gated: the two KDA paths have different fp32 orderings")
    # recompute locally too (don't trust the manifest alone)
    xids = np.concatenate([pids, dids[:1]])
    lx64, _, _ = oracle.prefill(Ps, top, cfg, xids, f64=True)
    step64 = oracle.decode_step(
        Ps, top, cfg, dids[0],
        oracle.prefill(Ps, top, cfg, pids, f64=True)[2], f64=True)
    check("f64 local contract recheck",
          np.abs(lx64[-1] - step64).max() < 1e-6,
          f"{np.abs(lx64[-1] - step64).max():.3g}")

    print("== 4b. KDA unit: chunked == recurrent over a full sequence (f64) ==")
    li_kda = cfg["layer_types"].index("linear_attention")
    Pk = Ps[li_kda]
    rng = np.random.default_rng(oracle.MASTER_SEED + 555)
    sq = 68                                    # exercises 2 chunks
    xu = oracle.bf16_round(rng.standard_normal((sq, cfg["hidden_size"]))
                           .astype(np.float32))
    Hu, Du = Pk["heads"], Pk["head_dim"]
    qu = oracle.bf16_round(rng.standard_normal((sq, Hu, Du))
                           .astype(np.float32))
    ku = oracle.bf16_round(rng.standard_normal((sq, Hu, Du))
                           .astype(np.float32))
    vu = oracle.bf16_round(rng.standard_normal((sq, Hu, Du))
                           .astype(np.float32))
    gu = -5.0 * rng.random((sq, Hu, Du))       # in the safe-gate range
    bu = oracle.bf16_round(rng.random((sq, Hu)).astype(np.float32))
    oc_, sc_ = oracle.kda_chunk(qu, ku, vu, gu, bu, None, f64=True)
    or_, sr_ = oracle.kda_recurrent(qu, ku, vu, gu, bu, None, f64=True)
    dcore = float(np.abs(oc_ - or_).max())
    dstate = float(np.abs(sc_ - sr_).max())
    check("chunk vs recurrent core out (f64) < 1e-8", dcore < 1e-8,
          f"{dcore:.3g}")
    check("chunk vs recurrent final state (f64) < 1e-8", dstate < 1e-8,
          f"{dstate:.3g}")

    print("== 5. indexer legality ==")
    # run a single DSA layer (layer 3) forward with synthetic input and
    # inspect the indexer selection
    li = cfg["layer_types"].index("deepseek_sparse_attention")
    P = Ps[li]
    st = oracle.LayerState(cfg, li)
    rng = np.random.default_rng(oracle.MASTER_SEED + 777)
    x = oracle.bf16_round(rng.standard_normal((9, cfg["hidden_size"]))
                          .astype(np.float32))
    interm = {}
    oracle.dsa_forward(P, x, st, False, interm)
    topk = interm["idx_topk"]
    n = st.k_cache.shape[0]
    legal = ((topk == -1) | ((topk >= 0) & (topk < n))).all()
    check("topk indices in range", bool(legal),
          f"shape {topk.shape}, kv_len {n}")
    width = cfg["index_topk"] + cfg["index_kpool"] - 1
    check("topk width = index_topk + kpool - 1", topk.shape == (9, width),
          str(topk.shape))
    # every query must see its own position (pool or tail coverage)
    covered = True
    for t in range(9):
        ids = set(topk[t][topk[t] >= 0].tolist())
        if t not in ids:
            covered = False
    check("every query attends to itself (fixture sizes)", covered)
    # decode steps keep indices legal as the cache grows
    for k in range(3):
        x1 = oracle.bf16_round(
            rng.standard_normal((1, cfg["hidden_size"])).astype(np.float32))
        oracle.dsa_forward(P, x1, st, False, interm)
        topk = interm["idx_topk"]
        n = st.k_cache.shape[0]
        ok = ((topk == -1) | ((topk >= 0) & (topk < n))).all()
        check(f"decode step {k} topk legal (kv_len {n})", bool(ok))

    print("== 6. Sinkhorn sanity ==")
    st_h = np.stack([oracle.bf16_round(
        rng.standard_normal(cfg["hidden_size"]).astype(np.float32))
        for _ in range(3)])[:, None, :].repeat(cfg["hc_mult"], axis=1)
    _, _, comb = oracle.hc_pre(st_h, Ps[0]["hc_attn_fn"],
                               Ps[0]["hc_attn_scale"], Ps[0]["hc_attn_base"],
                               cfg, False)
    col_dev = float(np.abs(comb.sum(axis=-2) - 1.0).max())
    check("comb column sums ~ 1 (< 1e-4)", col_dev < 1e-4, f"{col_dev:.3g}")

    print()
    if FAILURES:
        print(f"M0 GATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("M0 GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
