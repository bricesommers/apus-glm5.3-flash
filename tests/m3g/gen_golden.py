#!/usr/bin/env python3
"""M3G golden generator — GLM-5.3-Flash kernel fixtures (c/fp8blk.h, c/bf16.h).

Uses the M0 oracle's own codec (tools/oracle.py: E4M3_TABLE, bf16_round, _mm,
fp8_dequant, fp8_linear) so the goldens ARE the normative semantics:

  * dequant:  out_bf16[o,k] = bf16_rne(E4M3(w[o,k]) * ws[o/128, k/128])
              — one f32 multiply, then RNE narrow (oracle f32 mode).
  * bf16 GEMV/GEMM:  y[m,o] = bf16_rne(sum_k f32(x[m,k]) * f32(w[o,k]))
              — fp32 sequential ascending-k accumulate (oracle _mm), the
              order the C scalar anchor implements, so goldens are BITWISE.
  * fp8_linear composition: x @ dequant(W).T per the oracle's fp8_linear.

Fixtures (tests/m3g/golden/, gitignored):
  manifest.json       shapes + FNV-1a digests of every golden output
  deq_<i>_codes.bin   uint8 [O,K]      dequant case i input
  deq_<i>_scales.bin  float32 [nbo,nkb]
  deq_<i>_out.bin     uint16 [O,K]     expected BF16 codes
  gem_<i>_w.bin       uint16 [O,K]     GEMV/GEMM case i weights (BF16 codes)
  gem_<i>_x.bin       uint16 [M,K]
  gem_<i>_y.bin       uint16 [M,O]     expected output codes
  lin_<i>_*.bin       composition cases (codes/scales/x/y)

Shapes deliberately include non-multiples of the 128 block grid and of the
SIMD chunk/row-group tails (K % 32 != 0, O % 8 != 0, M % 4 != 0).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402  (tools/oracle.py — the M0 numpy oracle)

E4M3_TABLE = oracle.E4M3_TABLE
bf16_round = oracle.bf16_round
fnv1a = oracle.fnv1a


def bf16_codes(x):
    """f32 array -> raw uint16 BF16 codes (RNE, the oracle's bf16_round)."""
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def codes_to_f32(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def seq_mm_bf16(x_codes, w_codes):
    """y[m,o] = bf16_rne(sum_k f32(x)*f32(w)) with the oracle's _mm (fp32
    sequential ascending-k accumulate, two roundings per element)."""
    x = codes_to_f32(x_codes)
    w = codes_to_f32(w_codes)
    y = oracle._mm(x, w.T)                      # fp32, deterministic order
    return (bf16_round(y).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def rand_fp8_codes(rng, shape):
    """Random E4M3 codes, NaN codes 0x7F/0xFF excluded (converter-refused)."""
    c = rng.randint(0, 256, size=shape).astype(np.uint8)
    c[c == 0x7F] = 0x7E
    c[c == 0xFF] = 0xFE
    return c


def rand_scales(rng, shape):
    """Log-spread positive F32 scales + the corners the block grid can see."""
    s = np.exp2(rng.uniform(-20.0, 20.0, size=shape)).astype(np.float32)
    flat = s.reshape(-1)
    corners = [np.float32(1.0), np.float32(0.0),
               np.float32(2.0 ** -126),           # min normal
               np.float32(2.0 ** -127),           # subnormal
               np.float32(2.0 ** 40), np.float32(0.5)]
    for i, c in enumerate(corners):
        if i < flat.size:
            flat[i] = c
    return s


def fp8_dequant_ceil(codes, scales):
    """E4M3(codes) * scales[o//128, k//128] in f32 (ONE f32 multiply per
    element — the oracle's fp8_dequant semantics, generalized to ceil
    shapes; oracle.fp8_dequant's np.repeat requires 128-multiples)."""
    O, K = codes.shape
    w = E4M3_TABLE[codes].astype(np.float32)
    oi = np.arange(O) // 128
    ki = np.arange(K) // 128
    return w * scales[oi][:, ki].astype(np.float32)


def dequant_golden(codes, scales):
    """out[o,k] = bf16_rne(E4M3(codes[o,k]) * scales[o/128, k/128]) — the
    oracle's f32-mode dequant (one f32 mul) + bf16_round, as codes."""
    return bf16_codes(fp8_dequant_ceil(codes, scales))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(0x33)  # fixed seed
    manifest = {"dequant": [], "gemm": [], "linear": []}

    # --- dequant cases: (O, K) incl. non-multiples of 128 -----------------
    deq_shapes = [(128, 128), (256, 384), (200, 136), (1, 128), (384, 128),
                  (160, 200), (128, 127), (300, 264), (5, 129)]
    for i, (O, K) in enumerate(deq_shapes):
        codes = rand_fp8_codes(rng, (O, K))
        # inject edge codes into the first rows/columns
        edge = np.array([0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0x7E, 0xFE,
                         0xC0, 0x81, 0x77, 0x6F], np.uint8)
        codes.reshape(-1)[:edge.size] = edge
        nbo, nbk = (O + 127) // 128, (K + 127) // 128
        scales = rand_scales(rng, (nbo, nbk))
        out = dequant_golden(codes, scales)
        codes.tofile(f"{OUT}/deq_{i}_codes.bin")
        scales.tofile(f"{OUT}/deq_{i}_scales.bin")
        out.tofile(f"{OUT}/deq_{i}_out.bin")
        manifest["dequant"].append({
            "O": O, "K": K,
            "digest": fnv1a([out.tobytes()]),
        })

    # --- GEMV/GEMM cases: tails of 32-chunk, 8/4-row groups, M groups -----
    gem_shapes = [  # (M, O, K)
        (1, 1, 1), (1, 3, 5), (2, 8, 32), (1, 7, 33), (4, 16, 128),
        (5, 17, 129), (8, 64, 256), (3, 13, 200), (1, 128, 384),
        (7, 200, 160), (2, 33, 31), (6, 40, 100), (1, 256, 128),
    ]
    for i, (M, O, K) in enumerate(gem_shapes):
        w = bf16_codes(rng.uniform(-2.0, 2.0, size=(O, K)).astype(np.float32))
        x = bf16_codes(rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float32))
        # exact-value corners: zeros, inf-free extrema, subnormals
        w.reshape(-1)[0] = 0x0000
        if O * K > 1:
            w.reshape(-1)[1] = 0x8000        # -0.0
        if O * K > 4:
            w.reshape(-1)[2] = 0x0001    # bf16 min subnormal
        x.reshape(-1)[0] = 0x3F80        # 1.0
        y = seq_mm_bf16(x, w)
        w.tofile(f"{OUT}/gem_{i}_w.bin")
        x.tofile(f"{OUT}/gem_{i}_x.bin")
        y.tofile(f"{OUT}/gem_{i}_y.bin")
        manifest["gemm"].append({
            "M": M, "O": O, "K": K,
            "digest": fnv1a([y.tobytes()]),
        })

    # --- fp8_linear composition cases --------------------------------------
    lin_shapes = [(1, 256, 384), (5, 200, 264), (8, 384, 128), (3, 160, 200)]
    for i, (M, O, K) in enumerate(lin_shapes):
        codes = rand_fp8_codes(rng, (O, K))
        nbo, nbk = (O + 127) // 128, (K + 127) // 128
        scales = rand_scales(rng, (nbo, nbk))
        x = bf16_codes(rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float32))
        # oracle fp8_linear semantics, f32-faithful mode: dequant (one f32
        # mul) -> bf16 -> _mm fp32 sequential -> bf16 (fp8_dequant_ceil is
        # the ceil-shape generalization of oracle.fp8_dequant)
        w_deq = bf16_round(fp8_dequant_ceil(codes, scales))
        y = oracle._mm(codes_to_f32(x), w_deq.T)
        y_codes = (bf16_round(y).view(np.uint32)
                   >> np.uint32(16)).astype(np.uint16)
        codes.tofile(f"{OUT}/lin_{i}_codes.bin")
        scales.tofile(f"{OUT}/lin_{i}_scales.bin")
        x.tofile(f"{OUT}/lin_{i}_x.bin")
        y_codes.tofile(f"{OUT}/lin_{i}_y.bin")
        manifest["linear"].append({
            "M": M, "O": O, "K": K,
            "digest": fnv1a([y_codes.tobytes()]),
        })

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    # key=value copy for the C test (no JSON parser needed)
    with open(f"{OUT}/manifest.txt", "w") as f:
        f.write(f"ndeq={len(manifest['dequant'])}\n")
        for i, c in enumerate(manifest["dequant"]):
            f.write(f"deq_{i}_O={c['O']}\ndeq_{i}_K={c['K']}\n"
                    f"deq_{i}_digest={c['digest']}\n")
        f.write(f"ngem={len(manifest['gemm'])}\n")
        for i, c in enumerate(manifest["gemm"]):
            f.write(f"gem_{i}_M={c['M']}\ngem_{i}_O={c['O']}\n"
                    f"gem_{i}_K={c['K']}\ngem_{i}_digest={c['digest']}\n")
        f.write(f"nlin={len(manifest['linear'])}\n")
        for i, c in enumerate(manifest["linear"]):
            f.write(f"lin_{i}_M={c['M']}\nlin_{i}_O={c['O']}\n"
                    f"lin_{i}_K={c['K']}\nlin_{i}_digest={c['digest']}\n")
    print(f"m3g goldens: {len(deq_shapes)} dequant + {len(gem_shapes)} gemm"
          f" + {len(lin_shapes)} linear cases -> {OUT}")


if __name__ == "__main__":
    main()
