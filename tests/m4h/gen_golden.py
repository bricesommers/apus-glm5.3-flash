#!/usr/bin/env python3
"""M4H golden generator — GLM-5.3-Flash KDA + DSA/indexer sublayer
fixtures (c/gkda.h, c/gdsa.h).

Uses the M0 oracle's own sublayer functions (tools/oracle.py: kda_forward,
dsa_forward and their pieces) in f32-faithful mode, so the goldens ARE the
normative semantics. Both KDA ORDERINGS are generated separately (the M0
KDA ORDERING CONTRACT): prefill through the CHUNKED path
(kda_forward decode=False), decode through the RECURRENT path
(decode=True), state-carrying prefill continuation included (§4.4).

Every golden is stored as f32 (pre-round values where the reference keeps
fp32; bf16-valued where the reference bf16-rounds — the C side compares
its BF16 codes widened) plus the indexer top-k selections as i32.

Host-transcendental note: same tier design as m4g — the C gate is BITWISE
where libm expf == numpy float32 exp (macOS arm64; Linux x86_64 with
numpy pinned to its baseline exp kernel — the Makefile's golden-m4h
recipe sets NPY_DISABLE_CPU_FEATURES). The probe_* fixtures let the C
test detect the host property at runtime and fall back to the documented
tolerance class (tests/m4h/README.md).

Near-tie policy: the indexer score ties are STRUCTURAL with random
weights (relu-clipped pools tie at exactly 0.0, and ~25 % of queries have
all-negative scores so the zero block tops the ranking; masked pools tie
at exactly finfo.min) — no redraw can escape them. Instead the generator
records per-call FRAGILE query lists: a query is fragile when any
relevant consecutive gap in its sorted masked pool scores (pairs that
decide the emitted pool set/order; masked-pair ties excluded — they are
deterministic) is below SEL_MARGIN (0.02 — sized for the tolerance tier's
bf16-amplified host-exp perturbations, ~20x the measured class). The
BITWISE tier compares EVERYTHING (exact ties are deterministic: stable
descending, lower index first, in both numpy and the C top-k). The
TOLERANCE tier skips fragile queries' selections and compares the rest
bitwise; the score/output goldens use documented tolerance classes
(tests/m4h/README.md). A crafted EXACT-tie case (bitwise-identical token
rows -> bitwise-identical pool scores, zero flip risk) pins the stable
descending / lower-index-first tie-break in the bitwise tier.

Fixtures (tests/m4h/golden/, gitignored):
  manifest.json / manifest.txt   shapes, margins, FNV-1a digests
  probe_x.bin / probe_y.bin      f32 exp probe (np.exp values)
  kda_*                          the shared KDA weight set
  k{i}_*                         KDA case inputs/goldens/states
  dsa_*                          the shared DSA weight set
  d{i}_*                         DSA case inputs/goldens/states
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

DIM = 256          # hidden size (tiny fixture)
H = 2              # KDA heads / DSA heads
D = 128            # KDA head_dim / DSA qk/v head dim
DR = 128           # KDA low-rank width (f_a/g_a output)
QL = 128           # q_lora_rank
KL = 128           # kv_lora_rank
IH = 2             # indexer heads
ID = 128           # indexer head dim
ITOPK = 8          # index_topk (select_k = 2 pools)
KPOOL = 4          # index_kpool
WIDTH = ITOPK + KPOOL - 1
EPS = 1e-5         # rms_norm_eps
LB = -5.0          # gate_lower_bound
SEL_MARGIN = 0.02  # fragile-query threshold on relevant score gaps
                   # (tolerance-tier protection; ~20x the measured
                   # bf16-amplified host-exp perturbation class: a 1-ulp
                   # exp difference essentially never moves a pool score
                   # by more than ~1e-3 on these fixture magnitudes)

DIGESTS = {}


def bf16_codes(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def codes_to_f32(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(OUT, name + ".bin"))
    DIGESTS[name] = fnv1a([arr.tobytes()])


def rnd(rng, *shape, std=None):
    k = shape[-1]
    return (rng.standard_normal(shape)
            * (std if std is not None else 1.0 / math.sqrt(k))
            ).astype(np.float32)


def fp8_pair(rng, O, K):
    codes_b, scales_b, csh, ssh = oracle.fp8_store(rnd(rng, O, K))
    return (np.frombuffer(codes_b, dtype=np.uint8).reshape(csh),
            np.frombuffer(scales_b, dtype=np.float32).reshape(ssh))


# ---------------------------------------------------------------------------
# Shared weight sets
# ---------------------------------------------------------------------------

def gen_kda_weights(rng):
    qkv = H * D
    W = {
        "q_w": rnd(rng, qkv, DIM), "k_w": rnd(rng, qkv, DIM),
        "v_w": rnd(rng, qkv, DIM),
        "conv_w": (0.25 * rng.standard_normal((3 * qkv, 4))
                   ).astype(np.float32),
        "f_a": rnd(rng, DR, DIM), "f_b": rnd(rng, qkv, DR),
        "A_log": (0.5 * rng.standard_normal(H)).astype(np.float32),
        "dt_bias": (0.5 * rng.standard_normal(qkv)).astype(np.float32),
        "b_w": rnd(rng, H, DIM),
        "g_a": rnd(rng, DR, DIM), "g_b": rnd(rng, qkv, DR),
        "o_norm": (1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32),
        "o_w": rnd(rng, DIM, qkv),
    }
    for name in ("q_w", "k_w", "v_w", "conv_w", "f_a", "f_b", "b_w",
                 "g_a", "g_b", "o_norm", "o_w"):
        save(f"kda_{name}", bf16_codes(W[name]))
        W[name] = codes_to_f32(bf16_codes(W[name]))
    save("kda_A_log", W["A_log"])
    save("kda_dt_bias", W["dt_bias"])
    P = dict(W)
    P.update({"heads": H, "head_dim": D, "qkv_dim": qkv,
              "lower_bound": LB, "eps": EPS})
    return P


def gen_dsa_weights(rng):
    qac, qas = fp8_pair(rng, QL, DIM)
    qbc, qbs = fp8_pair(rng, H * D, QL)
    kvc, kvs = fp8_pair(rng, KL, DIM)
    oc, osc = fp8_pair(rng, DIM, H * D)
    W = {
        "q_a": (qac, qas), "q_b": (qbc, qbs), "kv_a": (kvc, kvs),
        "o_proj": (oc, osc),
        "q_a_norm": (1.0 + 0.05 * rng.standard_normal(QL)
                     ).astype(np.float32),
        "kv_a_norm": (1.0 + 0.05 * rng.standard_normal(KL)
                      ).astype(np.float32),
        "kv_b": rnd(rng, H * (D + D), KL),
        "idx_wq_b": rnd(rng, IH * ID, QL),
        "idx_wk": rnd(rng, ID, DIM),
        "idx_knorm_w": (1.0 + 0.05 * rng.standard_normal(ID)
                        ).astype(np.float32),
        "idx_knorm_b": (0.05 * rng.standard_normal(ID)).astype(np.float32),
        "idx_wproj": rnd(rng, IH, DIM),
        "idx_ape": (0.05 * rng.standard_normal((KPOOL, ID))
                    ).astype(np.float32),
        "idx_gate": rnd(rng, ID, DIM),
    }
    save("dsa_q_a_codes", qac)
    save("dsa_q_a_scales", qas)
    save("dsa_q_b_codes", qbc)
    save("dsa_q_b_scales", qbs)
    save("dsa_kv_a_codes", kvc)
    save("dsa_kv_a_scales", kvs)
    save("dsa_o_codes", oc)
    save("dsa_o_scales", osc)
    for name in ("q_a_norm", "kv_a_norm", "kv_b", "idx_wq_b", "idx_wk",
                 "idx_knorm_w", "idx_knorm_b", "idx_wproj", "idx_ape",
                 "idx_gate"):
        save(f"dsa_{name}", bf16_codes(W[name]))
        W[name] = codes_to_f32(bf16_codes(W[name]))
    P = dict(W)
    P.update({"heads": H, "qk_head_dim": D, "v_head_dim": D,
              "attn_scale": D ** -0.5, "eps": EPS,
              "idx_heads": IH, "idx_dim": ID, "idx_topk": ITOPK,
              "idx_kpool": KPOOL, "idx_tail": True,
              "idx_scale": ID ** -0.5})
    return P


# ---------------------------------------------------------------------------
# KDA cases
# ---------------------------------------------------------------------------

class KSt:
    def __init__(self):
        self.conv_state = np.zeros((3, 3 * H * D), dtype=np.float32)
        self.rec_state = None


def gen_kda_case(tag, P, rng, pre_lens, dec_steps):
    """pre_lens: list of chunked-prefill call lengths (state carried
    between them); dec_steps: recurrent steps after the last prefill."""
    st = KSt()
    calls = []
    for ci, spre in enumerate(pre_lens):
        x = bf16_codes(rng.standard_normal((spre, DIM)).astype(np.float32))
        save(f"{tag}_c{ci}_x", x)
        interm = {}
        out = oracle.kda_forward(P, codes_to_f32(x).reshape(spre, DIM),
                                 st, False, decode=False, interm=interm)
        calls.append((spre, x, interm, out))
    for di in range(dec_steps):
        x = bf16_codes(rng.standard_normal((1, DIM)).astype(np.float32))
        ci = len(pre_lens) + di
        save(f"{tag}_c{ci}_x", x)
        interm = {}
        out = oracle.kda_forward(P, codes_to_f32(x).reshape(1, DIM),
                                 st, False, decode=True, interm=interm)
        calls.append((1, x, interm, out))
    for ci, (spre, x, interm, out) in enumerate(calls):
        xf = codes_to_f32(x).reshape(spre, DIM)
        gate = oracle.bf16_linear(oracle.bf16_linear(xf, P["g_a"], False),
                                  P["g_b"], False)
        onorm = oracle.rms_norm_gated(interm["kda_core"], P["o_norm"],
                                      gate.reshape(spre, H, D), EPS, False)
        save(f"{tag}_c{ci}_mixed", interm["kda_mixed"])
        save(f"{tag}_c{ci}_g", interm["kda_g"])
        save(f"{tag}_c{ci}_beta", interm["kda_beta"])
        save(f"{tag}_c{ci}_core", interm["kda_core"])
        save(f"{tag}_c{ci}_gate", gate)
        save(f"{tag}_c{ci}_onorm", onorm.reshape(spre, -1))
        save(f"{tag}_c{ci}_out", out)
    save(f"{tag}_conv_state", st.conv_state)
    save(f"{tag}_rec_state", st.rec_state)
    return {"calls": len(calls), "pre_lens": pre_lens,
            "dec_steps": dec_steps}


# ---------------------------------------------------------------------------
# DSA cases
# ---------------------------------------------------------------------------

class DSt:
    k_cache = None
    v_cache = None
    idx_k = None
    idx_gate = None


def dsa_fragile(calls):
    """Per-call list of FRAGILE query indices: a query is fragile when any
    relevant consecutive gap in its sorted masked pool scores is below
    SEL_MARGIN. Relevant pairs are the consecutive sorted entries (j,
    j+1), j < select_k, that decide the emitted pool set/order; pairs
    whose lower entry is masked (finfo.min) are EXCLUDED (deterministic:
    a masked pool can never outrank a valid one, and masked selections
    all become -1). Exact 0.0 relu-clip ties ARE fragile (a bf16-amplified
    host-exp perturbation can un-clip them)."""
    out = []
    finfo_min = np.finfo(np.float32).min
    for ci, slen, base, x, interm, outp in calls:
        scores = interm["idx_scores"]
        n = base + slen
        n_full = n // KPOOL
        fragile = []
        if n_full:
            q_pos = base + np.arange(slen)
            pool_end = np.arange(n_full) * KPOOL + (KPOOL - 1)
            valid = pool_end[None, :] <= q_pos[:, None]
            sm = np.where(valid, scores, finfo_min)
            select_k = min(ITOPK // KPOOL, n_full)
            for t in range(slen):
                row = -np.sort(-sm[t])
                for j in range(min(select_k, n_full - 1)):
                    if row[j + 1] <= finfo_min:
                        continue
                    if float(row[j] - row[j + 1]) < SEL_MARGIN:
                        fragile.append(t)
                        break
        out.append(fragile)
    return out


def gen_dsa_case(tag, P, rng, call_lens, crafted=None):
    """call_lens: prefill length then per-step decode lengths (1).
    crafted="identical-rows": every x row is the same vector (the
    exact-tie pin; every selection is a bitwise exact tie)."""
    r = np.random.RandomState(rng.randint(0, 2**31))
    st = DSt()
    calls = []
    base = 0
    for ci, slen in enumerate(call_lens):
        if crafted == "identical-rows":
            row = r.standard_normal((1, DIM)).astype(np.float32)
            x = np.repeat(row, slen, axis=0)
        else:
            x = r.standard_normal((slen, DIM)).astype(np.float32)
        x = bf16_codes(x)
        interm = {}
        out = oracle.dsa_forward(P, codes_to_f32(x).reshape(slen, DIM),
                                 st, False, interm)
        calls.append((ci, slen, base, x, interm, out))
        base += slen
    fragile = dsa_fragile(calls)
    for ci, slen, b, x, interm, out in calls:
        n = b + slen
        save(f"{tag}_c{ci}_x", x)
        save(f"{tag}_c{ci}_q_resid", interm["dsa_q_resid"])
        save(f"{tag}_c{ci}_q", interm["dsa_q"])
        save(f"{tag}_c{ci}_k_new", interm["dsa_k_new"])
        save(f"{tag}_c{ci}_v_new", interm["dsa_v_new"])
        save(f"{tag}_c{ci}_idx_knew", interm["idx_k_new"])
        save(f"{tag}_c{ci}_idx_gnew", interm["idx_gate_new"])
        save(f"{tag}_c{ci}_idx_scores", interm["idx_scores"])
        save(f"{tag}_c{ci}_idx_topk", interm["idx_topk"].astype(np.int32))
        save(f"{tag}_c{ci}_probs", interm["attn_probs"])
        save(f"{tag}_c{ci}_attn", interm["dsa_attn"])
        save(f"{tag}_c{ci}_out", out)
    n = sum(call_lens)
    save(f"{tag}_k_cache", st.k_cache)
    save(f"{tag}_v_cache", st.v_cache)
    save(f"{tag}_idx_k", st.idx_k)
    save(f"{tag}_idx_gate", st.idx_gate)
    return {"calls": len(calls), "call_lens": call_lens, "n": n,
            "fragile": fragile,
            "crafted": crafted or ""}


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(0x4B4A)
    manifest = {"kda": [], "dsa": []}

    # exp probe (host-transcendental detection in the C test)
    px = np.concatenate([
        np.linspace(-90.0, 10.0, 1021, dtype=np.float32),
        np.array([0.0, -0.0, 1e-30], dtype=np.float32),
    ])
    save("probe_x", px)
    save("probe_y", np.exp(px))
    manifest["probe_n"] = int(px.size)

    kdaP = gen_kda_weights(np.random.RandomState(0x11))
    dsaP = gen_dsa_weights(np.random.RandomState(0x22))
    manifest["dims"] = {
        "dim": DIM, "H": H, "D": D, "Dr": DR, "ql": QL, "kl": KL,
        "IH": IH, "ID": ID, "itopk": ITOPK, "kpool": KPOOL,
        "width": WIDTH, "eps": EPS, "lb": LB,
    }

    # KDA cases: chunked prefill orderings + recurrent decode chains.
    manifest["kda"].append(gen_kda_case("k0", kdaP, rng, [68], 0))
    manifest["kda"].append(gen_kda_case("k1", kdaP, rng, [64], 0))
    manifest["kda"].append(gen_kda_case("k2", kdaP, rng, [1], 0))
    manifest["kda"].append(gen_kda_case("k3", kdaP, rng, [130], 0))
    manifest["kda"].append(gen_kda_case("k4", kdaP, rng, [68, 70], 0))
    manifest["kda"].append(gen_kda_case("k5", kdaP, rng, [68], 6))
    manifest["kda"].append(gen_kda_case("k6", kdaP, rng, [], 4))

    # DSA cases: prefill + decode chains with the pool-rebuild semantics.
    manifest["dsa"].append(gen_dsa_case("d0", dsaP, rng, [68, 1, 1, 1, 1]))
    manifest["dsa"].append(gen_dsa_case("d1", dsaP, rng, [3]))
    manifest["dsa"].append(gen_dsa_case("d2", dsaP, rng, [4, 1, 1, 1, 1, 1]))
    manifest["dsa"].append(gen_dsa_case("d3", dsaP, rng, [8]))
    manifest["dsa"].append(gen_dsa_case("d4", dsaP, rng, [12],
                                        crafted="identical-rows"))

    manifest["digests"] = DIGESTS
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # key=value copy for the C test (no JSON parser needed)
    dm = manifest["dims"]
    lines = [f"probe_n={manifest['probe_n']}"]
    for k, v in dm.items():
        lines.append(f"dim_{k}={v}")
    lines.append(f"nkda={len(manifest['kda'])}")
    for i, c in enumerate(manifest["kda"]):
        lines.append(f"k{i}_calls={c['calls']}")
        for j, slen in enumerate(c["pre_lens"]):
            lines.append(f"k{i}_c{j}_s={slen}")
        for j in range(c["dec_steps"]):
            lines.append(f"k{i}_c{len(c['pre_lens']) + j}_s=1")
        lines.append(f"k{i}_npre={len(c['pre_lens'])}")
    lines.append(f"ndsa={len(manifest['dsa'])}")
    for i, c in enumerate(manifest["dsa"]):
        lines.append(f"d{i}_calls={c['calls']}")
        for j, slen in enumerate(c["call_lens"]):
            lines.append(f"d{i}_c{j}_s={slen}")
            fr = c["fragile"][j]
            lines.append(f"d{i}_c{j}_fragile="
                         + (",".join(str(t) for t in fr) if fr else "-"))
        lines.append(f"d{i}_n={c['n']}")
        if c["crafted"]:
            lines.append(f"d{i}_crafted={c['crafted']}")
    for name, dg in DIGESTS.items():
        lines.append(f"dg_{name}={dg}")
    # newline="" pins LF: native-Windows Python text mode would otherwise
    # write CRLF, and test_m4h.c reads the manifest with "rb" — the CRLF
    # left parse_fragile stranded on '\r' (strtol no-conversion, no p
    # advance): the 2026-09-06 windows-latest 88-min CI hang. (The C
    # parser is hardened too; the m2 generator has pinned LF since M15.)
    with open(os.path.join(OUT, "manifest.txt"), "w", newline="") as f:
        f.write("\n".join(lines) + "\n")
    print(f"m4h goldens: {len(manifest['kda'])} KDA + "
          f"{len(manifest['dsa'])} DSA cases -> {OUT}")


if __name__ == "__main__":
    main()
