#!/usr/bin/env python3
"""tests/m8g/bin2cap.py — convert the C-engine h dump (tests/m8g/
dump_h_real.c output: 32-byte "APM8HCAP" header + s i32 token ids + s x
hc*dim f32 rows) into the mtp pin/probe capture schema (*.capture.npz:
tokens int64 [N], h_all float32 [N, hc, dim], plen scalar int).

Usage: .venv/bin/python tests/m8g/bin2cap.py [dump.bin] [out.npz] [plen]
Defaults: tests/m8g/mtp_h_c.bin -> tests/m8g/mtp_pin_c.capture.npz,
plen 31 (the Paris tf capture's chat-prompt length).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "mtp_h_c.bin")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, "mtp_pin_c.capture.npz")
    plen = int(sys.argv[3]) if len(sys.argv) > 3 else 31

    with open(src, "rb") as f:
        hdr = f.read(32)
        assert hdr[:8] == b"APM8HCAP", "bad magic"
        version, s, hc, dim = np.frombuffer(hdr[8:24], dtype=np.uint32)
        assert version == 1, version
        tokens = np.frombuffer(f.read(4 * s), dtype=np.int32)
        assert tokens.shape[0] == s
        h = np.frombuffer(f.read(), dtype=np.float32)
        assert h.size == s * hc * dim, (h.size, s * hc * dim)
        h_all = h.reshape(int(s), int(hc), int(dim)).copy()

    # sanity: finite + bf16-quantized values (f32 with a bf16 mantissa)
    assert np.isfinite(h_all).all(), "non-finite values in h_all"
    u = h_all.view(np.uint32)
    assert (u & np.uint32(0xFFFF) == 0).all(), "h_all not bf16-valued"
    np.savez(dst, tokens=tokens.astype(np.int64), h_all=h_all,
             plen=np.asarray(plen))
    print(f"{dst}: tokens [{len(tokens)}] (plen {plen}), h_all "
          f"{h_all.shape} float32, |h| max {np.abs(h_all).max():.4g}")


if __name__ == "__main__":
    main()
