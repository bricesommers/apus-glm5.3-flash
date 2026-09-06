#!/usr/bin/env python3
"""tests/m8g/verify_fixtures.py — M8a fixture integrity check (the C-side
gate is M8b; this is the minimal `make test-m8g` body until then).

Per case (golden/<case>/): every .bin digest in manifest.json is recomputed
from the file on disk, and the chain/replay coupling is sanity-checked
(chain_drafts[0] == argmax of the replay's last logits row — the engine
flow). Prints PASS/FAIL per case.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import oracle  # noqa: E402


def main():
    failures = 0
    for case in sorted(os.listdir(OUT)):
        case_dir = os.path.join(OUT, case)
        mpath = os.path.join(case_dir, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as f:
            manifest = json.load(f)
        n_bad = 0
        for name, want in sorted(manifest["digests"].items()):
            path = os.path.join(case_dir, name + ".bin")
            if not os.path.isfile(path):
                print(f"  {case}: MISSING {name}.bin")
                n_bad += 1
                continue
            got = oracle.fnv1a([open(path, "rb").read()])
            if got != want:
                print(f"  {case}: digest mismatch {name}: {got} != {want}")
                n_bad += 1
        cfg = manifest["config"]
        V, dim = cfg["V"], cfg["dim"]
        rl = np.fromfile(os.path.join(case_dir, "replay_logits.bin"),
                         np.float32).reshape(-1, V)
        drafts = np.fromfile(os.path.join(case_dir, "chain_drafts.bin"),
                             np.int32)
        if int(drafts[0]) != int(np.argmax(rl[-1])):
            print(f"  {case}: chain_drafts[0] != argmax(replay last row)")
            n_bad += 1
        oh = np.fromfile(os.path.join(case_dir, "replay_out_h.bin"),
                         np.float32).reshape(-1, dim)
        if oh.shape[0] != manifest["replay_len"]:
            print(f"  {case}: replay_out_h rows {oh.shape[0]} != "
                  f"{manifest['replay_len']}")
            n_bad += 1
        status = "PASS" if n_bad == 0 else "FAIL"
        failures += n_bad
        print(f"{case}: {status} ({len(manifest['digests'])} files, "
              f"replay_len {manifest['replay_len']}, chain_depth "
              f"{manifest['chain_depth']})")
    if failures:
        print(f"verify_fixtures: {failures} failures")
        return 1
    print("verify_fixtures: all cases PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
