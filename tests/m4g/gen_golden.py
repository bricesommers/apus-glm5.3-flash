#!/usr/bin/env python3
"""M4G golden generator — GLM-5.3-Flash sublayer fixtures (c/gmhc.h, c/gmoe.h).

Uses the M0 oracle's own sublayer functions (tools/oracle.py: hc_pre,
hc_post, hc_split_sinkhorn, router_forward, moe_forward, swiglu_mlp,
fp8_linear, unweighted_rms_norm, f32_linear) in f32-faithful mode, so the
goldens ARE the normative semantics. Every golden is stored as f32
(pre-round values where the reference bf16-rounds — the C side derives
the expected BF16 codes from them with its own m3g-gated RNE narrow) plus
the router selections as i32.

Host-transcendental note: the C gate is BITWISE on hosts where libm expf
== numpy's float32 exp (macOS arm64 always; Linux/x86_64 when numpy is
pinned to its baseline exp kernel — the Makefile's golden-m4g recipe sets
NPY_DISABLE_CPU_FEATURES for exactly this). The probe_* fixtures let the
C test detect the host property at runtime and fall back to the
documented tolerance class (tests/m4g/README.md).

Near-tie policy: random fixtures are RE-DRAWN until every token's biased-
score gap across the top-k boundary exceeds 1e-4 (recorded per case), so
the bitwise selection gate is robust to ±few-ulp host exp differences.
An exact-tie case (bitwise-identical gate rows) pins the stable
descending / lower-index-first tie-break with zero flip risk.

Fixtures (tests/m4g/golden/, gitignored):
  manifest.json / manifest.txt   shapes, margins, FNV-1a digests
  probe_x.bin / probe_y.bin      f32 exp probe (np.exp values)
  mhc{i}_*.bin                   h, fn(codes), scale, base, mixes, pre,
                                 post, comb, y (collapse, f32 bf16-valued)
  exp{i}_*.bin                   x(codes), res(codes), post, comb, y4
  head{i}_*.bin                  x4(codes), y
  rtr{i}_*.bin                   x(codes), w(codes), bias, scores, biased,
                                 idx(i32), wgt
  xp{i}_*.bin                    x(codes), g/u/d(_codes/_scales), go, uo,
                                 ho, out
  moe{i}_*.bin                   x(codes), gate_w(codes), gate_bias,
                                 e{e}_{g,u,d}_*, s{g,u,d}_*, scores,
                                 biased, idx(i32), wgt, routed, shared,
                                 out
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402  (tools/oracle.py — the M0 numpy oracle)

fnv1a = oracle.fnv1a
bf16_round = oracle.bf16_round

N = 4               # hc_mult
ITERS = 20          # hc_sinkhorn_iters
HC_EPS = 1e-6       # hc_eps
NORM_EPS = 1e-5     # rms_norm_eps
LIMIT = 10.0        # swiglu_limit
MARGIN = 1e-4       # min biased-score gap across the top-k boundary

DIGESTS = {}


def bf16_codes(x):
    """f32 array -> raw uint16 BF16 codes (the oracle's bf16_round)."""
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def codes_to_f32(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(OUT, name + ".bin"))
    DIGESTS[name] = fnv1a([arr.tobytes()])


def mhc_cfg(d):
    return {"hc_mult": N, "hidden_size": d, "rms_norm_eps": NORM_EPS,
            "hc_sinkhorn_iters": ITERS, "hc_eps": HC_EPS}


def topk_margin(biased, k):
    """Min over tokens of (k-th - (k+1)-th) biased score (descending)."""
    b = -np.sort(-biased, axis=-1)
    return float((b[:, k - 1] - b[:, k]).min())


def rnd(rng, *shape, std=None):
    k = shape[-1]
    return (rng.standard_normal(shape)
            * (std if std is not None else 1.0 / math.sqrt(k))
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# mHC maps + collapse (hc_pre)
# ---------------------------------------------------------------------------

def gen_mhc(rng, tag, s, d, h_scale=1.0, edge=False):
    h = (rng.standard_normal((s, N, d)) * h_scale).astype(np.float32)
    base = (0.25 * rng.standard_normal((2 + N) * N)).astype(np.float32)
    if edge:
        h[0] = 0.0
        h[1] = h[1] * 100.0
        h[2] = h[2] * 1e-3
        base[0] = 30.0            # sigmoid-saturating bases
        base[N] = -30.0
        base[2 * N] = 30.0
    h = bf16_round(h)             # the stream is bf16-valued
    fn = bf16_codes(rnd(rng, (2 + N) * N, N * d))
    scale = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
    cfg = mhc_cfg(d)
    fn_f32 = codes_to_f32(fn)
    flat = oracle.unweighted_rms_norm(h.reshape(s, N * d), NORM_EPS, False)
    mixes = oracle.f32_linear(flat, fn_f32, False)
    interm = {}
    y, post, comb = oracle.hc_pre(h, fn_f32, scale, base, cfg, False,
                                  interm, "t")
    pre, comb = interm["t_pre"], interm["t_comb"]
    # cross-check the split helper against hc_pre's internals
    pre2, post2, comb2 = oracle.hc_split_sinkhorn(mixes, scale, base, N,
                                                  ITERS, HC_EPS, False)
    assert np.array_equal(pre, pre2) and np.array_equal(post, post2)
    assert np.array_equal(comb, comb2)
    save(f"{tag}_h", h)
    save(f"{tag}_fn", fn)
    save(f"{tag}_scale", scale)
    save(f"{tag}_base", base)
    save(f"{tag}_mixes", mixes)
    save(f"{tag}_pre", pre)
    save(f"{tag}_post", post)
    save(f"{tag}_comb", comb)
    save(f"{tag}_y", y)           # bf16-valued f32 (the collapse output)
    return {"s": s, "d": d, "n": N}


def gen_expand(rng, tag, s, d):
    x = bf16_codes(rng.standard_normal((s, d)).astype(np.float32))
    res = bf16_codes(rng.standard_normal((s, N, d)).astype(np.float32))
    post = (2.0 * rng.random((s, N))).astype(np.float32)
    comb = oracle.softmax(rng.standard_normal((s, N, N))
                          .astype(np.float32), axis=-1)
    y4 = oracle.hc_post(codes_to_f32(x), codes_to_f32(res), post, comb,
                        False)
    save(f"{tag}_x", x)
    save(f"{tag}_res", res)
    save(f"{tag}_post", post)
    save(f"{tag}_comb", comb)
    save(f"{tag}_y4", y4)         # bf16-valued f32
    return {"s": s, "d": d, "n": N}


def gen_head(rng, tag, s, d):
    x4 = bf16_codes(rng.standard_normal((s, N, d)).astype(np.float32))
    y = oracle._B(codes_to_f32(x4).mean(axis=1), False)
    save(f"{tag}_x4", x4)
    save(f"{tag}_y", y)
    return {"s": s, "d": d, "n": N}


# ---------------------------------------------------------------------------
# MoE router
# ---------------------------------------------------------------------------

def router_P(gate_f32, bias, topk):
    return {"gate_w": gate_f32, "gate_bias": bias, "top_k": topk,
            "route_scale": 2.5, "norm_topk_prob": True,
            "n_group": 1, "topk_group": 1}


def gen_router(rng, tag, s, E, d, topk):
    """Random router case, re-drawn until the top-k boundary margin holds."""
    for attempt in range(100):
        r = np.random.RandomState(rng.randint(0, 2**31) + attempt)
        x = bf16_codes(r.standard_normal((s, d)).astype(np.float32))
        gw = bf16_codes((0.05 * r.standard_normal((E, d)))
                        .astype(np.float32))
        bias = (0.1 * r.standard_normal(E)).astype(np.float32)
        interm = {}
        w, idx = oracle.router_forward(router_P(codes_to_f32(gw), bias,
                                                topk),
                                       codes_to_f32(x), False, interm)
        margin = topk_margin(interm["router_scores_biased"], topk)
        if margin > MARGIN:
            break
    else:
        raise RuntimeError(f"{tag}: no draw with margin > {MARGIN}")
    save(f"{tag}_x", x)
    save(f"{tag}_w", gw)
    save(f"{tag}_bias", bias)
    save(f"{tag}_scores", interm["router_scores"])
    save(f"{tag}_biased", interm["router_scores_biased"])
    save(f"{tag}_idx", idx.astype(np.int32))
    save(f"{tag}_wgt", w)
    return {"s": s, "E": E, "d": d, "topk": topk, "margin": margin}


def gen_router_tie(rng, tag, d=256, E=8, topk=3):
    """Crafted EXACT tie (duplicate gate rows + equal bias -> identical
    biased scores, zero flip risk): the stable descending tie-break must
    emit the LOWER index first."""
    s = 2
    r = np.random.RandomState(0x71E)
    x = bf16_codes(r.standard_normal((s, d)).astype(np.float32))
    gw_f = (0.05 * r.standard_normal((E, d))).astype(np.float32)
    # make rows 2 and 5 identical and strongly aligned with x[0]
    strong = np.abs(codes_to_f32(x[0])) * 0.05
    gw_f[2] = strong
    gw_f[5] = strong
    bias = (0.1 * r.standard_normal(E)).astype(np.float32)
    bias[5] = bias[2]
    gw = bf16_codes(gw_f)
    interm = {}
    P = router_P(codes_to_f32(gw), bias, topk)
    w, idx = oracle.router_forward(P, codes_to_f32(x), False, interm)
    # pin: the EXACT tie (bitwise-identical rows + bias -> identical
    # biased scores, zero flip risk) breaks to the LOWER index first
    assert interm["router_scores_biased"][0][2] == \
        interm["router_scores_biased"][0][5]
    assert idx[0][0] == 2 and idx[0][1] == 5, idx[0]
    save(f"{tag}_x", x)
    save(f"{tag}_w", gw)
    save(f"{tag}_bias", bias)
    save(f"{tag}_scores", interm["router_scores"])
    save(f"{tag}_biased", interm["router_scores_biased"])
    save(f"{tag}_idx", idx.astype(np.int32))
    save(f"{tag}_wgt", w)
    return {"s": s, "E": E, "d": d, "topk": topk, "margin": 0.0,
            "crafted": "exact-tie"}


def gen_router_selbias(rng, tag, d=256, E=8, topk=3):
    """Crafted selection-only-bias case: the correction bias reorders the
    top-k selection vs the UNBIASED sigmoid scores, and the weights must
    still be gathered from the unbiased ones (noaux_tc semantics)."""
    s = 2
    r = np.random.RandomState(0xB1A5)
    x = bf16_codes(r.standard_normal((s, d)).astype(np.float32))
    gw = bf16_codes((0.05 * r.standard_normal((E, d))).astype(np.float32))
    bias = np.zeros(E, dtype=np.float32)
    interm = {}
    P = router_P(codes_to_f32(gw), bias, topk)
    oracle.router_forward(P, codes_to_f32(x), False, interm)
    # token 1: boost the unbiased-weakest expert past the top-k boundary
    unb = interm["router_scores"][1]
    weak = int(np.argsort(-unb, kind="stable")[-1])
    b_sorted = -np.sort(-unb)
    bias[weak] = (b_sorted[topk - 1] - unb[weak]) + 0.05
    w, idx = oracle.router_forward(P, codes_to_f32(x), False, interm)
    # pins: the biased selection differs from the unbiased one, the weak
    # expert IS selected, and its weight is its UNBIASED score
    sel_biased = set(idx[1].tolist())
    sel_unb = set(np.argsort(-unb, kind="stable")[:topk].tolist())
    assert weak in sel_biased and sel_biased != sel_unb
    sc = interm["router_scores"][1]
    wg = sc[idx[1]]
    wg = wg / (wg.sum() + np.float32(1e-20)) * np.float32(2.5)
    assert np.array_equal(w[1], wg)
    # token 0 untouched by the bias (all-zero except `weak`); the boundary
    # margin must be strictly positive everywhere for the selection gate
    margin = topk_margin(interm["router_scores_biased"], topk)
    assert margin > 0.0
    save(f"{tag}_x", x)
    save(f"{tag}_w", gw)
    save(f"{tag}_bias", bias)
    save(f"{tag}_scores", interm["router_scores"])
    save(f"{tag}_biased", interm["router_scores_biased"])
    save(f"{tag}_idx", idx.astype(np.int32))
    save(f"{tag}_wgt", w)
    return {"s": s, "E": E, "d": d, "topk": topk, "margin": margin,
            "crafted": "selection-only-bias"}


# ---------------------------------------------------------------------------
# Expert / dense MLP (swiglu_mlp)
# ---------------------------------------------------------------------------

def fp8_pair(rng, O, K):
    codes_b, scales_b, csh, ssh = oracle.fp8_store(rnd(rng, O, K))
    codes = np.frombuffer(codes_b, dtype=np.uint8).reshape(csh)
    scales = np.frombuffer(scales_b, dtype=np.float32).reshape(ssh)
    return codes, scales


def gen_expert(rng, tag, s, d, inter, x_std=1.0):
    x = bf16_codes((rng.standard_normal((s, d)) * x_std)
                   .astype(np.float32))
    gc, gs = fp8_pair(rng, inter, d)
    uc, us = fp8_pair(rng, inter, d)
    dc, ds = fp8_pair(rng, d, inter)
    xf = codes_to_f32(x)
    g = oracle.fp8_linear(xf, gc, gs, False)
    u = oracle.fp8_linear(xf, uc, us, False)
    gclamp = np.minimum(g, LIMIT)
    uclamp = np.clip(u, -LIMIT, LIMIT)
    n_gclamp = int((g != gclamp).sum())
    n_uclamp = int((u != uclamp).sum())
    h = oracle._B(oracle.silu(gclamp), False)
    h = oracle._B(h * oracle._B(uclamp, False), False)
    out = oracle.fp8_linear(h, dc, ds, False)
    # cross-check against the oracle's swiglu_mlp (same composition)
    ref = oracle.swiglu_mlp(xf, (gc, gs), (uc, us), (dc, ds), LIMIT, False)
    assert np.array_equal(out, ref)
    save(f"{tag}_x", x)
    save(f"{tag}_g_codes", gc)
    save(f"{tag}_g_scales", gs)
    save(f"{tag}_u_codes", uc)
    save(f"{tag}_u_scales", us)
    save(f"{tag}_d_codes", dc)
    save(f"{tag}_d_scales", ds)
    save(f"{tag}_go", g)          # post-GEMM, pre-clamp (bf16-valued)
    save(f"{tag}_uo", u)
    save(f"{tag}_ho", h)
    save(f"{tag}_out", out)
    return {"s": s, "d": d, "inter": inter, "limit": LIMIT,
            "g_clamped": n_gclamp, "u_clamped": n_uclamp}


# ---------------------------------------------------------------------------
# Full MoE forward
# ---------------------------------------------------------------------------

def gen_moe(rng, tag, s, E, d, topk, inter, sinter):
    for attempt in range(100):
        r = np.random.RandomState(rng.randint(0, 2**31) + attempt)
        x = bf16_codes(r.standard_normal((s, d)).astype(np.float32))
        gw = bf16_codes((0.05 * r.standard_normal((E, d)))
                        .astype(np.float32))
        bias = (0.1 * r.standard_normal(E)).astype(np.float32)
        experts = []
        for e in range(E):
            eg = fp8_pair(r, inter, d)
            eu = fp8_pair(r, inter, d)
            ed = fp8_pair(r, d, inter)
            experts.append((eg, eu, ed))
        sg = fp8_pair(r, sinter, d)
        su = fp8_pair(r, sinter, d)
        sd = fp8_pair(r, d, sinter)
        P = dict(router_P(codes_to_f32(gw), bias, topk))
        P.update({"n_routed": E, "limit": LIMIT,
                  "experts": experts,
                  "shared": (sg, su, sd)})
        interm = {}
        y = oracle.moe_forward(P, codes_to_f32(x), False, interm)
        margin = topk_margin(interm["router_scores_biased"], topk)
        if margin > MARGIN:
            break
    else:
        raise RuntimeError(f"{tag}: no draw with margin > {MARGIN}")
    save(f"{tag}_x", x)
    save(f"{tag}_gate_w", gw)
    save(f"{tag}_gate_bias", bias)
    for e in range(E):
        for nm, pair in (("g", experts[e][0]), ("u", experts[e][1]),
                         ("d", experts[e][2])):
            save(f"{tag}_e{e}_{nm}_codes", pair[0])
            save(f"{tag}_e{e}_{nm}_scales", pair[1])
    for nm, pair in (("g", sg), ("u", su), ("d", sd)):
        save(f"{tag}_s{nm}_codes", pair[0])
        save(f"{tag}_s{nm}_scales", pair[1])
    save(f"{tag}_scores", interm["router_scores"])
    save(f"{tag}_biased", interm["router_scores_biased"])
    save(f"{tag}_idx", interm["router_idx"].astype(np.int32))
    save(f"{tag}_wgt", interm["router_w"])
    save(f"{tag}_routed", interm["moe_routed"])
    save(f"{tag}_shared", interm["moe_shared"])
    save(f"{tag}_out", y)
    return {"s": s, "E": E, "d": d, "topk": topk, "inter": inter,
            "sinter": sinter, "limit": LIMIT, "margin": margin}


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(0x4467)
    manifest = {"mhc": [], "expand": [], "head": [], "router": [],
                "expert": [], "moe": []}

    # exp probe (host-transcendental detection in the C test)
    px = np.concatenate([
        np.linspace(-90.0, 10.0, 1021, dtype=np.float32),
        np.array([0.0, -0.0, 1e-30], dtype=np.float32),
    ])
    save("probe_x", px)
    save("probe_y", np.exp(px))
    manifest["probe_n"] = int(px.size)

    manifest["mhc"].append(gen_mhc(rng, "mhc0", 6, 256))
    manifest["mhc"].append(gen_mhc(rng, "mhc1", 6, 256, edge=True))
    manifest["mhc"].append(gen_mhc(rng, "mhc2", 3, 128))
    manifest["mhc"].append(gen_mhc(rng, "mhc3", 1, 256))

    manifest["expand"].append(gen_expand(rng, "exp0", 6, 256))
    manifest["expand"].append(gen_expand(rng, "exp1", 1, 128))

    manifest["head"].append(gen_head(rng, "head0", 6, 256))
    manifest["head"].append(gen_head(rng, "head1", 1, 128))

    manifest["router"].append(gen_router(rng, "rtr0", 6, 8, 256, 3))
    manifest["router"].append(gen_router_tie(rng, "rtr1"))
    manifest["router"].append(gen_router_selbias(rng, "rtr2"))
    manifest["router"].append(gen_router(rng, "rtr3", 4, 16, 256, 8))

    manifest["expert"].append(gen_expert(rng, "xp0", 4, 256, 128))
    manifest["expert"].append(gen_expert(rng, "xp1", 4, 256, 128,
                                         x_std=8.0))
    manifest["expert"].append(gen_expert(rng, "xp2", 2, 256, 256))

    manifest["moe"].append(gen_moe(rng, "moe0", 6, 8, 256, 3, 128, 128))
    manifest["moe"].append(gen_moe(rng, "moe1", 4, 16, 256, 8, 128, 128))

    manifest["digests"] = DIGESTS
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # key=value copy for the C test (no JSON parser needed)
    lines = [f"probe_n={manifest['probe_n']}"]
    for sec, prefix in (("mhc", "mhc"), ("expand", "exp"), ("head", "head"),
                        ("router", "rtr"), ("expert", "xp"), ("moe", "moe")):
        cases = manifest[sec]
        lines.append(f"n{sec}={len(cases)}")
        for i, c in enumerate(cases):
            for k, v in c.items():
                lines.append(f"{prefix}{i}_{k}={v}")
    for name, dg in DIGESTS.items():
        lines.append(f"dg_{name}={dg}")
    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"m4g goldens: {sum(len(v) for k, v in manifest.items()
                             if isinstance(v, list))} cases -> {OUT}")


if __name__ == "__main__":
    main()
