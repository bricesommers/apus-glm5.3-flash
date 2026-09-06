#!/usr/bin/env python3
"""tests/m7a/gen_fixtures.py — M7a fixture generator: scripted GLM-5.3-Flash
"parrot" containers for the server battery.

The server milestone needs a glm5_next v2 container whose generated token
stream is a fixed, known script (GLM tool-call output, EOS handling, stop
strings, exact usage counts, SSE), asserted end-to-end through the real C
engine AND independently reproduced by the M0 oracle from the same
container. Random weights cannot do that. The parrot trick (same idea as
the V4 m7a fixtures, rebuilt for the GLM stack):

  * ALL layer weights are ZERO (KDA/DSA/MoE/dense-MLP + mHC fn/base/scale):
    every sublayer output F(x) is 0, the mHC comb is the uniform hc×hc
    doubly-stochastic matrix (Sinkhorn of zeros) with post = 2·sigmoid(0)
    = 1, so the hc-replicated residual stream is preserved exactly. The
    final hidden state therefore depends ONLY on the last embedded token:
    the unweighted-mean HyperHead averages the identical copies, the final
    RMSNorm (weight 1) rescales. Logits = lm_head @ f(embed[last]).
  * Norm weights (input/post layernorm, o_norm, q/kv_a_layernorm, indexer
    k_norm, top-level norm) are 1.0; embed rows are random N(0,1) (bf16).
  * lm_head is all zero except: for each scripted transition a -> b,
    head[b] = 8·embed[a]. Greedy decoding then follows the scripted Markov
    chain with a huge logit margin (checked against the ORACLE's logits at
    gen time, > 100), so token equality is host-exp independent.
  * Chain tokens with multi-character content are ADDED TOKENS (ids 268+),
    each used at exactly one chain position so every transition source is
    unique. The GLM specials ([gMASK] <sop> <|system|> <|user|>
    <|assistant|> <|observation|> <|endoftext|> <think> </think>) are
    added tokens 256-267 (compact ids mirroring the real layout).
  * The byte-level vocab (ids 0-255, GPT-2 bytes_to_unicode alphabet, no
    merges) tokenizes arbitrary prompt text one byte per token — with no
    merges the Qwen2-style Split pre-tokenizer in c/tok.h cannot change
    the byte-id sequence.

Pipeline (deterministic from SEED):
  1. oracle.make_tiny_config() (L=5: KDA 0,1,2,4 + DSA 3; dense MLP 0-2,
     MoE 3,4 — every layer kind exercised) + oracle.write_weights, then
     payload surgery (zero / norms=1 / scripted head) and the
     "model.language_model." prefix, as a fake HF source checkpoint.
  2. tools/convert.py convert -> the M1 v2 container (one pread per
     expert slab — the same path the m5g/m6g fixtures take).
  3. config.json (flat fixture config for the C parser), tokenizer.json
     and generation_config.json (eos stop set) are placed INSIDE the
     container dir — the layout `bin/apus run/serve --model DIR` expects.
  4. Oracle gate: the oracle loads the model BACK FROM THE CONTAINER and
     greedy-generates from the one-token prompt [THINK]; the stream must
     equal the script exactly (this verifies the parrot mechanism) and
     every step's top1-top2 margin must exceed 100. Because the parrot is
     Markov in the last token, that one stream is the golden for EVERY
     prompt ending in <think> (all chat-template prompts do).

Two variants are written:
  fixtures/model_chat/    reasoning + "The answer is STOP right here." + EOS
  fixtures/model_tools/   reasoning + GLM <tool_call> get_weather + EOS

Regenerate: `make golden-m7a` (or run this file directly). Deterministic.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402  (tools/oracle.py — the M0 numpy oracle)

SEED = 20260830
FIX = os.path.join(HERE, "fixtures")
MARGIN_MIN = 100.0

# ---- tokenizer layout ------------------------------------------------------
# ids 0-255: byte alphabet. ids 256+: added tokens (GLM specials, then the
# scripted chain chunks). Ids mirror the real model's special-token roles.

EOS = 256        # <|endoftext|>
GMASK = 258      # [gMASK]
SOP = 260        # <sop>
SYSTEM = 262     # <|system|>
USER = 263       # <|user|>
ASSISTANT = 264  # <|assistant|>
OBSERVATION = 265
THINK = 266      # <think>
THINK_END = 267  # </think>

SPECIALS = [
    (EOS, "<|endoftext|>"),
    (257, "[MASK]"),
    (GMASK, "[gMASK]"),
    (259, "[sMASK]"),
    (SOP, "<sop>"),
    (261, "<eop>"),
    (SYSTEM, "<|system|>"),
    (USER, "<|user|>"),
    (ASSISTANT, "<|assistant|>"),
    (OBSERVATION, "<|observation|>"),
    (THINK, "<think>"),
    (THINK_END, "</think>"),
]

# ---- scripted chains -------------------------------------------------------
# Full texts (the constants tests assert):
#   model_chat:  "reasoning: thinking it over.</think>"
#                "The answer is STOP right here." + EOS
#   model_tools: "I should check the weather.</think>"
#                "<tool_call>get_weather<arg_key>location</arg_key>"
#                "<arg_value>Beijing</arg_value></tool_call>" + EOS

CHAT_CHUNKS = {
    268: "reasoning: thinking it over.",
    269: "The answer is ",
    270: "STOP",
    271: " right here.",
}
CHAT_CHAIN = [268, THINK_END, 269, 270, 271, EOS]

TOOL_CALL_TEXT = ("<tool_call>get_weather<arg_key>location</arg_key>"
                  "<arg_value>Beijing</arg_value></tool_call>")
TOOLS_CHUNKS = {
    268: "I should check the weather.",
    269: TOOL_CALL_TEXT,
}
TOOLS_CHAIN = [268, THINK_END, 269, EOS]

VARIANTS = {
    "model_chat": (CHAT_CHUNKS, CHAT_CHAIN),
    "model_tools": (TOOLS_CHUNKS, TOOLS_CHAIN),
}


def bytes_to_unicode():
    """GPT-2 byte-level alphabet (the GLM/Qwen2 alphabet c/tok.h expects)."""
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD))
          + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()


def write_tokenizer(path, chunks):
    vocab = {B2U[b]: b for b in range(256)}
    added = [
        {"id": i, "content": c, "special": True, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False}
        for i, c in SPECIALS
    ]
    added += [
        {"id": i, "content": c, "special": False, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False}
        for i, c in sorted(chunks.items())
    ]
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=1)


# ---- weight surgery ----------------------------------------------------------

_NORM_SUFFIXES = ("layernorm.weight", "o_norm.weight", "k_norm.weight")


def is_norm(name):
    return name == "norm.weight" or name.endswith(_NORM_SUFFIXES)


def build_source_checkpoint(cfg, seed, src_dir, transitions, embed_out):
    """oracle.write_weights -> payload surgery (zero everything except
    embed + norms; script the lm_head transitions) -> fake HF checkpoint
    with the "model.language_model." prefix (the m5g pattern)."""
    tmp = src_dir + ".stripped"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    oracle.write_weights(cfg, tmp, seed)
    os.makedirs(src_dir, exist_ok=True)
    weight_map = {}
    embed = None
    for fname in sorted(os.listdir(tmp)):
        if not fname.endswith(".safetensors"):
            continue
        header, data_start = oracle.read_shard(os.path.join(tmp, fname))
        recs = []
        with open(os.path.join(tmp, fname), "rb") as f:
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                f.seek(data_start + meta["data_offsets"][0])
                payload = f.read(meta["data_offsets"][1]
                                 - meta["data_offsets"][0])
                if name == "embed_tokens.weight":
                    embed = oracle.bf16_bytes_to_f32(payload).reshape(
                        *meta["shape"]).copy()
                elif name == "lm_head.weight":
                    V, d = meta["shape"]
                    head = np.zeros((V, d), dtype=np.float32)
                    for a, b in transitions.items():
                        head[b] = np.float32(8.0) * embed[a]
                    payload = oracle.f32_to_bf16_bytes(head)
                elif is_norm(name):
                    n = len(payload) // (2 if meta["dtype"] == "BF16" else 4)
                    ones = np.ones(n, dtype=np.float32)
                    payload = (oracle.f32_to_bf16_bytes(ones)
                               if meta["dtype"] == "BF16"
                               else ones.tobytes())
                else:
                    payload = b"\x00" * len(payload)
                new = (name if name == "lm_head.weight"
                       else "model.language_model." + name)
                recs.append((new, meta["dtype"], meta["shape"], payload))
                weight_map[new] = fname
        oracle.write_shard(os.path.join(src_dir, fname), recs)
    with open(os.path.join(src_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {"total_size": 0},
                   "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(src_dir, "config.json"), "w") as f:
        json.dump({"num_hidden_layers": cfg["num_hidden_layers"]}, f)
    shutil.rmtree(tmp)
    embed_out.append(embed)


def convert(src_dir, dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "convert.py"),
         "convert", src_dir, dst_dir],
        check=True, cwd=ROOT,
        stdout=subprocess.DEVNULL if os.environ.get("QUIET") else None)


# ---- oracle greedy gate ------------------------------------------------------


def oracle_greedy(Ps, top, cfg, pids, eos_ids, max_steps=16):
    """prefill + greedy decode chain on the oracle (f32-faithful mode).
    Returns (ids incl. the terminating eos, min top1-top2 margin)."""
    logits_pre, _, states = oracle.prefill(Ps, top, cfg, pids, f64=False)
    cur = np.asarray(logits_pre, dtype=np.float64)[-1]
    out, worst = [], 1e30
    for _ in range(max_steps):
        t = int(np.argmax(cur))
        worst = min(worst, cur[t] - np.max(np.delete(cur, t)))
        if t in eos_ids:
            out.append(t)
            break
        out.append(t)
        cur = np.asarray(
            oracle.decode_step(Ps, top, cfg, t, states, f64=False),
            dtype=np.float64)
    return out, worst


def build_variant(name, chunks, chain):
    cfg = oracle.make_tiny_config()
    V = cfg["vocab_size"]
    assert max(max(chunks), 267) < V, "chain ids exceed the tiny vocab"

    transitions = {THINK: chain[0]}
    for a, b in zip(chain, chain[1:]):
        assert a not in transitions, f"duplicate chain source {a}"
        transitions[a] = b

    vdir = os.path.join(FIX, name)
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
    src = vdir + ".src"
    embed_holder = []
    build_source_checkpoint(cfg, SEED, src, transitions, embed_holder)
    convert(src, vdir)
    shutil.rmtree(src)

    # the engine's expected in-container layout
    with open(os.path.join(vdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    write_tokenizer(os.path.join(vdir, "tokenizer.json"), chunks)
    eos_ids = [EOS, USER, OBSERVATION]   # the reference stop-set shape
    with open(os.path.join(vdir, "generation_config.json"), "w") as f:
        json.dump({"eos_token_id": eos_ids}, f, indent=1)

    # oracle gate: greedy from [THINK] must reproduce the script exactly,
    # with a comfortable logit margin at every step
    shards = oracle.ShardSet(vdir)
    Ps, top = oracle.load_model_params(shards, cfg)
    stream, margin = oracle_greedy(Ps, top, cfg, [THINK], set(eos_ids))
    print(f"  {name}: oracle stream {stream}, min margin {margin:.1f}")
    assert stream == chain, f"{name}: oracle {stream} != script {chain}"
    assert margin > MARGIN_MIN, f"{name}: margin {margin} too small"
    return {
        "chain": chain,
        "chain_text": "".join(chunks.get(t, "") for t in chain),
        "oracle_stream": stream,
        "min_margin": margin,
        "eos_ids": eos_ids,
        "specials": {c: i for i, c in SPECIALS},
        "transitions": {str(a): b for a, b in transitions.items()},
    }


def main():
    os.makedirs(FIX, exist_ok=True)
    golden = {"seed": SEED, "variants": {}}
    for name, (chunks, chain) in VARIANTS.items():
        golden["variants"][name] = build_variant(name, chunks, chain)
    with open(os.path.join(FIX, "golden.json"), "w") as f:
        json.dump(golden, f, indent=1)
    print("m7a fixtures done")


if __name__ == "__main__":
    main()
