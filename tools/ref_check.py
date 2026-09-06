#!/usr/bin/env python3
"""Real-weight reference check: run the M0 oracle (HF-semantics port) on the
converted GLM-5.3-Flash container and dump reference logits + per-layer
block outputs for bisecting the C engine at real scale.

Usage: .venv/bin/python tools/ref_check.py [ids_csv] [out_dir]
Defaults: "The capital of France is" ids, tests/smoke/ref/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import oracle  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "weights", "glm-5.3-flash")
IDS = [785, 6722, 315, 9621, 374]  # "The capital of France is"


class LazyShardSet:
    """mmap-backed ShardSet for the real 306 GiB container: never pins a
    whole shard, never copies expert codes (the fixture ShardSet reads full
    shards into RAM — OOM at real scale). Same raw/meta/f32/fp8 semantics;
    fp8 returns mmap views (read-only) instead of copies."""

    def __init__(self, weights_dir):
        import mmap
        self._mmap_mod = mmap
        with open(os.path.join(weights_dir,
                               "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.dir = weights_dir
        self._shards = {}  # fname -> (mmap, header, data_start)

    def _shard(self, fname):
        if fname not in self._shards:
            header, data_start = oracle.read_shard(
                os.path.join(self.dir, fname))
            f = open(os.path.join(self.dir, fname), "rb")
            mm = self._mmap_mod.mmap(f.fileno(), 0,
                                     prot=self._mmap_mod.PROT_READ)
            self._shards[fname] = (mm, header, data_start)
        return self._shards[fname]

    def raw(self, name):
        fname = self.weight_map[name]
        mm, header, data_start = self._shard(fname)
        begin, end = header[name]["data_offsets"]
        return memoryview(mm)[data_start + begin:data_start + end]

    def meta(self, name):
        fname = self.weight_map[name]
        _, header, _ = self._shard(fname)
        return header[name]["dtype"], header[name]["shape"]

    def f32(self, name):
        dtype, shape = self.meta(name)
        b = self.raw(name)
        if dtype == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        if dtype == "BF16":
            return oracle.bf16_bytes_to_f32(b).reshape(shape)
        raise ValueError(f"{name}: {dtype}")

    def fp8(self, name):
        dt, csh = self.meta(name + ".weight")
        assert dt == "F8_E4M3", (name, dt)
        codes = np.frombuffer(self.raw(name + ".weight"),
                              np.uint8).reshape(csh)
        dt, ssh = self.meta(name + ".weight_scale_inv")
        assert dt == "F32", (name, dt)
        scales = np.frombuffer(self.raw(name + ".weight_scale_inv"),
                               np.float32).reshape(ssh)
        return codes, scales  # mmap views — no copy (RAM is the point)


def main():
    ids = ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1
           else IDS)
    out = (sys.argv[2] if len(sys.argv) > 2 else
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "tests", "smoke", "ref"))
    os.makedirs(out, exist_ok=True)

    cfg = oracle.load_text_config(os.path.join(WEIGHTS, "config.json"))
    print(f"config: {cfg['num_hidden_layers']} layers, dim "
          f"{cfg['hidden_size']}, vocab {cfg['vocab_size']}", flush=True)
    shards = LazyShardSet(WEIGHTS)
    Ps, top = oracle.load_model_params(shards, cfg)
    print("params loaded", flush=True)

    interm = []
    ids = np.asarray(ids, dtype=np.int64)
    logits, h, states = oracle.prefill(Ps, top, cfg, ids, f64=False,
                                       layer_interm=interm)
    # Per-layer block output at the LAST position (what drives the next
    # token) — [L, hc, dim] f32 — for C-side bisection.
    blk = np.stack([im["block_out_h"][-1].astype(np.float32)
                    for im in interm])
    np.savez(os.path.join(out, "ref.npz"),
             ids=ids,
             logits_last=logits[-1].astype(np.float32),
             block_out_last=blk,
             h_last=h[-1].astype(np.float32))
    digs = {f"layer{i:02d}": oracle.array_digest(blk[i])
            for i in range(blk.shape[0])}
    digs["logits_last"] = oracle.array_digest(logits[-1])
    with open(os.path.join(out, "digests.json"), "w") as f:
        json.dump(digs, f, indent=1, sort_keys=True)

    last = logits[-1]
    order = np.argsort(-last)[:10]
    print("top-10 next-token ids:", [int(i) for i in order], flush=True)
    print("  Paris id 12089 logit:", float(last[12089]),
          "| rank:", int((last > last[12089]).sum()) + 1, flush=True)
    print("written to", os.path.normpath(out), flush=True)


if __name__ == "__main__":
    main()
