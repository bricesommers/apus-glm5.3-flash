#!/usr/bin/env python3
"""Compare C trace_real per-layer digests vs the oracle reference (ref.npz).
Usage: .venv/bin/python tools/ref_compare.py [c_trace_output] [ref_dir]
The C side digests BF16 codes of the last-position block stream; the oracle
side digests f32_to_bf16_bytes of the same values — identical bytes iff the
implementations agree bitwise. Prints the first divergent layer.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import oracle  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "tests", "smoke", "ref")


def fnv1a64(b: bytes) -> str:
    h = 14695981039346656037
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main():
    c_out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REF, "c_trace.txt")
    ref_dir = sys.argv[2] if len(sys.argv) > 2 else REF
    z = np.load(os.path.join(ref_dir, "ref.npz"))
    blk = z["block_out_last"]  # [L, hc, dim] f32, bf16-valued

    c_digs = {}
    with open(c_out) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0].startswith("layer"):
                c_digs[parts[0]] = parts[1]

    first_bad = None
    for l in range(blk.shape[0]):
        key = f"layer{l:02d}"
        ref_d = fnv1a64(oracle.f32_to_bf16_bytes(
            np.ascontiguousarray(blk[l]).ravel()))
        c_d = c_digs.get(key, "<missing>")
        ok = "OK " if c_d == ref_d else "BAD"
        if c_d != ref_d and first_bad is None:
            first_bad = l
        print(f"{key} C={c_d} ref={ref_d} {ok}")
    if first_bad is None:
        print("ALL LAYERS MATCH — divergence is in the head (norm/lm_head)")
    else:
        print(f"\nfirst divergence: layer {first_bad:02d}")


if __name__ == "__main__":
    main()
