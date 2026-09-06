#!/usr/bin/env python3
"""tools/mtp_pin.py — M8a empirical pin for the MTP (NextN) draft head.

Resolves the two open semantics questions from the 2026-09-04 M8 KICKOFF
STATUS entry on the REAL 306 GiB container (weights/glm-5.3-flash):

  1. hnorm input: the trunk stream is mHC [s, 4, 4096] but hnorm is
     4096-wide. Candidates (oracle.mtp_hnorm_input): "prenorm_mean" (the
     HyperHead mean `y`), "postnorm" (post-final-norm `yn`), "hc0".."hc3"
     (individual mHC slots).
  2. (h, id) pairing lag: 0 = pair (h_p, tok_p) predicts tok_{p+1} (the
     parent engine's convention, ../Apus/c/mtp.h); 1 = pair (h_{p-1},
     tok_p) predicts tok_{p+1} (DeepSeek-V3/SGLang EAGLE bookkeeping).

Method: greedy main-model decode on a fixed chat-formatted prompt while
capturing every candidate hidden per position; then per (candidate, lag)
config, replay the draft chain exactly as the engine will — MTP state
holds TRUE pairs only (prompt replay batched, one true pair per committed
position, snapshot/restore around the D-deep draft chain) — and measure
the acceptance of the D=4 argmax drafts against the actual greedy
continuation. Expected: the right choice lands near the technical-report
figure (~2.76 accept length at 4 draft steps, arXiv:2602.15763); a wrong
choice gives ~1.0 (SGLang issue #36829).

Oracle-on-real-weights is SLOW (numpy, lazy 306 GiB mmap): the main decode
dominates (~1-2 min/token). Run as a background task. Results are written
to tests/m8g/mtp_pin_results.{json,txt}.

Usage:
  .venv/bin/python tools/mtp_pin.py [--weights DIR] [--positions N]
                                    [--depth D] [--out PREFIX]
                                    [--ids 1,2,3,...]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import oracle  # noqa: E402
from ref_check import LazyShardSet, WEIGHTS  # noqa: E402

CANDIDATES = ["prenorm_mean", "postnorm", "hc0", "hc1", "hc2", "hc3"]
LAGS = [0, 1]

PROMPT_TEXT = ("Write a short travel guide for Paris: three paragraphs "
               "covering landmarks, food, and getting around.")


def chat_prompt_ids(weights_dir):
    """Render the fixed prompt through the GLM chat template (jinja2, the
    same reference/chat_template.jinja the M2 encoding implements) and
    tokenize with the in-container tokenizer.json."""
    from tokenizers import Tokenizer
    import jinja2
    tmpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "reference", "chat_template.jinja")
    with open(tmpl_path) as f:
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        tmpl = env.from_string(f.read())
    text = tmpl.render(messages=[{"role": "user", "content": PROMPT_TEXT}],
                       add_generation_prompt=True)
    tok = Tokenizer.from_file(os.path.join(weights_dir, "tokenizer.json"))
    return np.asarray(tok.encode(text, add_special_tokens=False).ids,
                      dtype=np.int64)


def encode_text(weights_dir, text):
    """Tokenize raw text with the in-container tokenizer (teacher-forced
    natural-continuation captures, --tf_file)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(weights_dir, "tokenizer.json"))
    return np.asarray(tok.encode(text, add_special_tokens=False).ids,
                      dtype=np.int64)


def load_cfg(weights_dir):
    path = os.path.join(weights_dir, "config.json")
    with open(path) as f:
        raw = json.load(f)
    if "text_config" in raw:
        return oracle.load_text_config(path)
    return raw  # flat oracle fixture config (synthetic smoke runs)


def capture_greedy(Ps, top, cfg, ids, n_decode):
    """Greedy main-model run: prefill + n_decode decode steps. Returns
    (tokens [plen+n_decode], h_all [plen+n_decode, hc, dim]) — h_all[p] is
    the final mHC stream at position p (the MTP hnorm candidates derive
    from it)."""
    plen = len(ids)
    t0 = time.time()
    logits, h, states = oracle.prefill(Ps, top, cfg, ids, f64=False)
    print(f"prefill {plen} tokens in {time.time() - t0:.1f}s", flush=True)
    h_rows = [h[i].astype(np.float32) for i in range(plen)]
    tokens = list(int(t) for t in ids)
    for i in range(n_decode):
        nxt = int(np.argmax(logits[-1]))
        t0 = time.time()
        logits, h = oracle.model_forward(Ps, top, cfg, [nxt], states,
                                         False, decode=True)
        for st in states:
            st.pos += 1
        tokens.append(nxt)
        h_rows.append(h[0].astype(np.float32))
        print(f"  decode {i + 1}/{n_decode} tok={nxt} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return np.asarray(tokens, dtype=np.int64), np.stack(h_rows)


def snap_state(st):
    return {"pos": st.pos,
            **{n: (None if getattr(st, n) is None else getattr(st, n).copy())
               for n in ("k_cache", "v_cache", "idx_k", "idx_gate")}}


def restore_state(st, snap):
    st.pos = snap["pos"]
    for n in ("k_cache", "v_cache", "idx_k", "idx_gate"):
        setattr(st, n, snap[n])


def eval_config(Pm, top, cfg, tokens, cand, lag, plen, n_pos, depth):
    """One (candidate, lag) config: true-pair replay over the prompt, then
    per position draft `depth` tokens and count consecutive accepts against
    the greedy continuation. Returns (accepts, drafts) per-position lists
    (drafts are logged for diagnosis — an all-zero sweep needs to see WHAT
    the draft head predicts)."""
    st = oracle.new_mtp_state(cfg)
    # prompt replay (batched): lag 0 pairs (h_p, x_p) p = 0..plen-1;
    # lag 1 pairs (h_{p-1}, x_p) p = 1..plen-1. The last row's logits are
    # the first draft distribution (position plen).
    lo = 0 if lag == 0 else 1
    rids = tokens[lo:plen]
    rh = cand[lo - lag:plen - lag] if lag else cand[:plen]
    logits, out_h = oracle.mtp_forward(Pm, top, cfg, rids, rh, st, False)
    d1_logits, d1_h = logits[-1], out_h[-1]
    accepts = []
    all_drafts = []
    for q in range(plen, plen + n_pos):
        drafts = [int(np.argmax(d1_logits))]
        cur_h = d1_h
        snap = snap_state(st)
        for _ in range(depth - 1):                       # chain d2..dD
            lg, oh = oracle.mtp_forward(Pm, top, cfg, [drafts[-1]],
                                        cur_h[None, :], st, False)
            drafts.append(int(np.argmax(lg[0])))
            cur_h = oh[0]
        restore_state(st, snap)
        all_drafts.append(drafts)
        n_acc = 0
        for j in range(depth):
            if q + j < len(tokens) and drafts[j] == int(tokens[q + j]):
                n_acc += 1
            else:
                break
        accepts.append(n_acc)
        # commit the true pair for position q -> next draft distribution
        if lag == 0:
            pid, ph = tokens[q], cand[q]
        else:
            pid, ph = tokens[q], cand[q - 1]
        lg, oh = oracle.mtp_forward(Pm, top, cfg, [pid], ph[None, :], st,
                                    False)
        d1_logits, d1_h = lg[0], oh[0]
    return accepts, all_drafts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--positions", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tests", "m8g", "mtp_pin_results"))
    ap.add_argument("--ids", default=None,
                    help="comma-separated prompt ids (skips the tokenizer)")
    ap.add_argument("--tf_file", default=None,
                    help="natural text appended after the chat prompt and "
                         "teacher-forced (no greedy decode); the sweep "
                         "evaluates acceptance against the natural tokens")
    ap.add_argument("--resume", action="store_true",
                    help="load tokens/h_all from OUT.capture.npz instead of "
                         "re-running the greedy main-model capture")
    args = ap.parse_args()

    cfg = load_cfg(args.weights)
    print(f"config: {cfg['num_hidden_layers']} layers, dim "
          f"{cfg['hidden_size']}, vocab {cfg['vocab_size']}", flush=True)
    shards = LazyShardSet(args.weights)
    t0 = time.time()
    Ps, top = oracle.load_model_params(shards, cfg)
    Pm = oracle.load_mtp_params(shards, cfg)
    print(f"params loaded in {time.time() - t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cap_path = args.out + ".capture.npz"
    if args.resume and os.path.isfile(cap_path):
        cap = np.load(cap_path)
        tokens, h_all = cap["tokens"], cap["h_all"]
        plen = int(cap["plen"])
        print(f"resumed capture: {plen} prompt + "
              f"{len(tokens) - plen} decode tokens", flush=True)
    else:
        if args.tf_file is not None:
            # Teacher-forced natural-continuation capture: chat prompt +
            # natural text appended; the sweep's "true continuation" is the
            # natural text itself, so NO greedy decode steps are run (the
            # degenerate-greedy regime biases acceptance downward — M8a).
            chat_ids = chat_prompt_ids(args.weights)
            with open(args.tf_file) as f:
                tf_ids = encode_text(args.weights, f.read())
            ids = np.concatenate([chat_ids, tf_ids])
            plen = len(chat_ids)
            n_decode = 0
        else:
            if args.ids is not None:
                ids = np.asarray([int(x) for x in args.ids.split(",")],
                                 dtype=np.int64)
            else:
                ids = chat_prompt_ids(args.weights)
            plen = len(ids)
            n_decode = args.positions + args.depth
        print(f"prompt: {plen} tokens (+ {len(ids) - plen} tf/decode "
              f"tokens)", flush=True)
        tokens, h_all = capture_greedy(Ps, top, cfg, ids, n_decode)
        np.savez(cap_path, tokens=tokens, h_all=h_all,
                 plen=np.asarray(plen))
        print(f"capture checkpointed to {cap_path}", flush=True)
    del Ps  # free the main-model param references before the sweep

    cand_map = {c: oracle.mtp_hnorm_input(h_all, top, cfg, False, c)
                for c in CANDIDATES}

    results = {"weights": os.path.normpath(args.weights),
               "positions": args.positions, "depth": args.depth,
               "prompt_ids": [int(t) for t in tokens[:plen]],
               "greedy_tokens": [int(t) for t in tokens],
               "configs": {}}
    if args.resume and os.path.isfile(args.out + ".json"):
        with open(args.out + ".json") as f:
            results["configs"] = json.load(f)["configs"]
        print(f"resuming with {len(results['configs'])} configs done",
              flush=True)
    for cname in CANDIDATES:
        for lag in LAGS:
            key = f"{cname}/lag{lag}"
            if key in results["configs"]:
                continue
            t0 = time.time()
            acc, drafts = eval_config(Pm, top, cfg, tokens,
                                      cand_map[cname], lag, plen,
                                      args.positions, args.depth)
            mean_drafts = float(np.mean(acc))
            # accept length = accepted drafts + the always-emitted seed
            results["configs"][key] = {
                "accepts_per_position": acc,
                "drafts_per_position": drafts,
                "mean_accepted_drafts": mean_drafts,
                "mean_accept_length": mean_drafts + 1.0,
                "d1_hit_rate": float(np.mean([1 if a > 0 else 0
                                              for a in acc])),
            }
            # incremental save: a crash mid-sweep keeps finished configs
            with open(args.out + ".json", "w") as f:
                json.dump(results, f, indent=1)
            print(f"{key}: mean accepted drafts {mean_drafts:.3f} "
                  f"(accept length {mean_drafts + 1.0:.3f}) "
                  f"[{time.time() - t0:.1f}s]", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(results, f, indent=1)
    lines = ["M8a MTP hnorm-input / pairing pin",
             f"weights: {results['weights']}",
             f"positions: {args.positions}  depth: {args.depth}  "
             f"prompt: {plen} tokens",
             "",
             f"{'config':<24}{'accepted drafts':>16}{'accept length':>16}"
             f"{'d1 hit rate':>13}"]
    for key, r in results["configs"].items():
        lines.append(f"{key:<24}{r['mean_accepted_drafts']:>16.3f}"
                     f"{r['mean_accept_length']:>16.3f}"
                     f"{r['d1_hit_rate']:>13.3f}")
    lines.append("")
    lines.append("technical-report reference: ~2.76 accept length at 4 "
                 "draft steps (arXiv:2602.15763)")
    with open(args.out + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("written:", os.path.normpath(args.out) + ".{json,txt}",
          flush=True)


if __name__ == "__main__":
    main()
