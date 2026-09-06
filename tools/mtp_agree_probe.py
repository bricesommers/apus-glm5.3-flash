#!/usr/bin/env python3
"""tools/mtp_agree_probe.py — M8a diagnostic: draft-vs-MAIN agreement probe.

mtp_rank_probe.py ranks the TRUE text token in the MTP draft logits. On a
teacher-forced capture that is the wrong question: acceptance is whether the
draft's pick equals the MAIN MODEL's OWN pick, and the main model may
disagree with the natural text. This probe computes, per position:

  - main_logits  = head(norm(mean(h_all[p-1])))  (cheap from the capture;
      exact — the capture stores the trunk stream and the HyperHead/final
      norm/head are position-local)
  - main_argmax  = argmax main_logits  (what greedy WOULD pick here)
  - text_rank    = rank of the text token in main_logits (is the text even
      natural TO THE MODEL?)
  - d1 logits    = MTP draft distribution after replaying true pairs
  - agree        = (argmax d1 == main_argmax)   ← the acceptance criterion
  - agree_rank   = rank of main_argmax in d1

per hnorm candidate × lag. CLEAR VERDICT: some (candidate, lag) with
agree-rate ~60%+ (TR: accept length ~2.76 at depth 4) on text the main
model itself ranks highly.

Usage:
  .venv/bin/python tools/mtp_agree_probe.py [--cap PATH] [--positions N]
      [--lag 0] [--weights DIR]
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
    ap.add_argument("--cap", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tests", "m8g", "mtp_pin_c.capture.npz"))
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--lag", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.weights)
    shards = LazyShardSet(args.weights)
    t0 = time.time()
    _Ps, top = oracle.load_model_params(shards, cfg)
    del _Ps
    Pm = oracle.load_mtp_params(shards, cfg)
    print(f"params loaded in {time.time() - t0:.1f}s", flush=True)

    cap = np.load(args.cap)
    tokens, h_all, plen = cap["tokens"], cap["h_all"], int(cap["plen"])
    n_pos = min(args.positions, len(tokens) - plen - 1)
    print(f"capture: {plen} prompt + {len(tokens) - plen} tf tokens; "
          f"probing {n_pos} positions at lag {args.lag}", flush=True)

    # main-model logits per probed position (HyperHead mean -> norm -> head)
    eps = cfg["rms_norm_eps"]
    main_am, text_rank = [], []
    for q in range(plen, plen + n_pos):
        y = h_all[q - 1].astype(np.float32).mean(axis=0)
        yn = oracle.rms_norm(y, top["norm"], eps, False)
        lg = oracle.bf16_linear(yn, top["head"], False)
        am = int(np.argmax(lg))
        main_am.append(am)
        tt = int(tokens[q])
        text_rank.append(int(1 + np.sum(lg > lg[tt])))
    print(f"main-model view of the text: median text-token rank "
          f"{float(np.median(text_rank)):.1f} (ranks {text_rank})",
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
        agrees, agree_ranks = [], []
        for i, q in enumerate(range(plen, plen + n_pos)):
            da = int(np.argmax(d1))
            agrees.append(1 if da == main_am[i] else 0)
            agree_ranks.append(int(1 + np.sum(d1 > d1[main_am[i]])))
            pid = int(tokens[q])
            ph = cand[q] if lag == 0 else cand[q - 1]
            lg, _ = oracle.mtp_forward(Pm, top, cfg, [pid],
                                       ph[None, :], st, False)
            d1 = lg[0]
        top5 = sum(1 for r in agree_ranks if r <= 5)
        print(f"{cname}/lag{lag}: d1==main_argmax at "
              f"{sum(agrees)}/{n_pos} positions; median rank of main's "
              f"pick in d1: {float(np.median(agree_ranks)):.1f}; top-5 "
              f"mass {top5}/{n_pos} "
              f"(ranks {agree_ranks})", flush=True)


if __name__ == "__main__":
    main()
