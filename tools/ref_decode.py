#!/usr/bin/env python3
"""Oracle side of the teacher-forced decode bisect: prefill the prompt, then
decode a FIXED chain (default: the C engine's 4 tokens), printing per-step
logits + per-layer digests in the same format as tests/smoke/trace_real.c
(BF16 bytes of the last-position block stream).
Usage: .venv/bin/python tools/ref_decode.py [out_txt]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import oracle  # noqa: E402
from ref_check import LazyShardSet, WEIGHTS  # noqa: E402

PROMPT = [785, 6722, 315, 9621, 374]
CHAIN = [12089, 13, 758, 8584]  # teacher-forced: C's own decode tokens
                                # (post-gate7-fix greedy 2026-09-05;
                                # pre-fix was [279, 9103, 369, 279])


def fnv1a64(b: bytes) -> str:
    h = 14695981039346656037
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main():
    out = (sys.argv[1] if len(sys.argv) > 1 else
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "tests", "smoke", "ref", "ref_decode.txt"))
    cfg = oracle.load_text_config(os.path.join(WEIGHTS, "config.json"))
    shards = LazyShardSet(WEIGHTS)
    Ps, top = oracle.load_model_params(shards, cfg)
    print("params loaded", flush=True)
    ids = np.asarray(PROMPT, dtype=np.int64)
    lines = []
    logits, h, states = oracle.prefill(Ps, top, cfg, ids, f64=False)
    for step, tok in enumerate(CHAIN):
        interm = []
        logits = oracle.decode_step(Ps, top, cfg, tok, states, f64=False,
                                    layer_interm=interm)
        lines.append("step%d logits %s" %
                     (step, fnv1a64(oracle.f32_to_bf16_bytes(logits))))
        for l, im in enumerate(interm):
            row = np.ascontiguousarray(im["block_out_h"][-1])
            lines.append("step%d layer%02d %s" %
                         (step, l,
                          fnv1a64(oracle.f32_to_bf16_bytes(row.ravel()))))
        print(f"step {step} done (argmax {int(np.argmax(logits))})",
              flush=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("written:", os.path.normpath(out), flush=True)


if __name__ == "__main__":
    main()
