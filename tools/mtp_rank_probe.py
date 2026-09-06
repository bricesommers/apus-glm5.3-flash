#!/usr/bin/env python3
"""tools/mtp_rank_probe.py — M8a diagnostic: teacher-forced rank probe.

Companion to tools/mtp_pin.py for the all-zero-acceptance case: instead of
exact-match acceptance (what the pin sweep measures), replay the TRUE
(h, id) pairs from the checkpointed capture (mtp_pin_results.capture.npz)
and report the RANK of the true next token in the MTP d1 logits per hnorm
candidate. Interpretation:

  - rank ~1-10 for some candidate  → the MTP block computes correctly and
    the pin is a semantics choice (that candidate wins; acceptance follows).
  - rank ~vocab/2 for ALL          → the block/loading itself is broken
    (bug hunt, not a semantics verdict).

Only the MTP params + top are needed (the capture checkpoint carries the
main model's tokens + h_all), so this is a few minutes, not hours.

Usage:
  .venv/bin/python tools/mtp_rank_probe.py [--weights DIR] [--positions N]
      [--cap PATH] [--lag 0]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import oracle  # noqa: E402
from ref_check import LazyShardSet, WEIGHTS  # noqa: E402
from mtp_pin import CANDIDATES, load_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--cap", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tests", "m8g", "mtp_pin_results.capture.npz"))
    args = ap.parse_args()

    cfg = load_cfg(args.weights)
    shards = LazyShardSet(args.weights)
    t0 = time.time()
    _Ps, top = oracle.load_model_params(shards, cfg)
    del _Ps  # the probe never runs the main model
    Pm = oracle.load_mtp_params(shards, cfg)
    print(f"params loaded in {time.time() - t0:.1f}s", flush=True)

    cap = np.load(args.cap)
    tokens, h_all, plen = cap["tokens"], cap["h_all"], int(cap["plen"])
    print(f"capture: {plen} prompt + {len(tokens) - plen} decode tokens",
          flush=True)

    lag = args.lag
    for cname in CANDIDATES:
        cand = oracle.mtp_hnorm_input(h_all, top, cfg, False, cname)
        st = oracle.new_mtp_state(cfg)
        lo = 0 if lag == 0 else 1
        rids = tokens[lo:plen]
        rh = cand[lo - lag:plen - lag] if lag else cand[:plen]
        logits, _ = oracle.mtp_forward(Pm, top, cfg, rids, rh, st, False)
        d1 = logits[-1]
        ranks = []
        for q in range(plen, min(plen + args.positions, len(tokens))):
            true_tok = int(tokens[q])
            rank = int(1 + np.sum(d1 > d1[true_tok]))
            ranks.append(rank)
            # commit the true pair for position q -> next d1
            pid = true_tok
            ph = cand[q] if lag == 0 else cand[q - 1]
            lg, _ = oracle.mtp_forward(Pm, top, cfg, [pid],
                                       ph[None, :], st, False)
            d1 = lg[0]
        med = float(np.median(ranks))
        print(f"{cname}/lag{lag}: median rank {med:8.1f}  "
              f"ranks {ranks}", flush=True)


if __name__ == "__main__":
    main()
