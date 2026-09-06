"""M1 test 5 — converter assumptions vs the REAL checkpoint index.

Parses reference/model.safetensors.index.json (the real 62-shard,
76,108-tensor GLM-5.3-Flash index) and asserts every assumption the
converter relies on:

  * the naming scheme matches the converter's prefixes/regexes (after
    dropping "model.language_model."),
  * every (layer, expert) has exactly the 6-tensor group
    {gate,up,down}_proj.{weight,weight_scale_inv},
  * FP8/scale_inv pairs are never split across input shards,
  * expert splits across input shards are exactly the ONE known case
    (MTP expert (45,197) — the converter's deferral path exists for it),
  * layer inventories: 34 KDA + 11 DSA + MTP(layer 45, DSA+MoE, no hc_*),
    MoE on 3..44+45, dense on 0..2,
  * of-record size arithmetic (pinned in tests/m1/README.md):
    per-expert slab 25,171,968 B, experts total 311,729,651,712 B.

NOTE (documented deviation): the real index maps tensor name -> shard
file only; it carries NO shapes/dtypes. Shapes/dtypes are read from each
shard's safetensors header at conversion time and validated there
(128x128 F32 scale pairing); the slab arithmetic below is derived from
config.json (hidden 4096, moe inter 2048, block [128,128]).
"""

import json
import os
import re
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))
import convert as apus_convert  # noqa: E402 — reuse converter regexes

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "reference")

EXPERT_SRC_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")


class TestIndexRealism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REF_DIR, "model.safetensors.index.json")) as f:
            cls.index = json.load(f)
        cls.wm = cls.index["weight_map"]
        with open(os.path.join(REF_DIR, "config.json")) as f:
            cls.cfg = json.load(f)
        cls.text = cls.cfg["text_config"]

    def test_scale_and_counts(self):
        self.assertEqual(len(set(self.wm.values())), 62)
        self.assertEqual(len(self.wm), 76108)
        self.assertEqual(self.index["metadata"]["total_size"],
                         328326771576)   # 305.78 GiB

    def test_prefix_coverage(self):
        """Every tensor is language_model / visual / lm_head — nothing
        else exists at the top level."""
        for name in self.wm:
            if name.startswith(apus_convert.VISION_PREFIX):
                continue
            if name in apus_convert.TOPLEVEL_KEEP:
                continue
            self.assertTrue(
                name.startswith(apus_convert.LANG_PREFIX), name)

    def test_every_text_name_matches_converter_whitelist(self):
        """The stripped name of every kept tensor must pass the
        converter's structural whitelist — no drift, no surprises."""
        for name in self.wm:
            if name.startswith(apus_convert.VISION_PREFIX):
                continue
            stripped = apus_convert.output_name(name)
            apus_convert.validate_structure(stripped)  # raises on drift

    def test_pattern_count_drift_guard(self):
        pats = {re.sub(r"\d+", "N", n) for n in self.wm}
        self.assertEqual(len(pats), 91)

    def test_vision_inventory(self):
        vis = [n for n in self.wm
               if n.startswith(apus_convert.VISION_PREFIX)]
        self.assertEqual(len(vis), 347)
        blocks = {int(m.group(1)) for n in vis
                  for m in [re.match(r"model\.visual\.blocks\.(\d+)\.", n)]
                  if m}
        self.assertEqual(blocks, set(range(24)))

    def test_expert_groups_complete(self):
        groups = defaultdict(set)
        for name in self.wm:
            m = EXPERT_SRC_RE.match(name)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                groups[key].add(f"{m.group(3)}.{m.group(4)}")
        # 43 MoE layers (3..44 + MTP 45) x 288 experts
        self.assertEqual(len(groups), 43 * 288)
        self.assertEqual(len(groups), 12384)
        expected = set(apus_convert.SLAB_MEMBERS)
        for key, members in groups.items():
            self.assertEqual(members, expected, f"expert {key} incomplete")

    def test_expert_split_inventory(self):
        """Exactly one expert is split across input shards: MTP (45,197),
        up_proj pair in shard 2, the rest in shard 1. The converter's
        deferral machinery exists for this case; if a future checkpoint
        revision splits MORE experts the mechanism still applies, but
        this test must be revisited consciously."""
        shards = defaultdict(set)
        for name, shard in self.wm.items():
            m = EXPERT_SRC_RE.match(name)
            if m:
                shards[(int(m.group(1)), int(m.group(2)))].add(shard)
        split = {k: sorted(v) for k, v in shards.items() if len(v) > 1}
        self.assertEqual(list(split), [(45, 197)])
        self.assertEqual(split[(45, 197)],
                         ["model-00001-of-00062.safetensors",
                          "model-00002-of-00062.safetensors"])
        up = [n for n, s in self.wm.items()
              if n.startswith(
                  "model.language_model.layers.45.mlp.experts.197.up_proj")
              ]
        self.assertEqual(len(up), 2)
        for n in up:
            self.assertEqual(self.wm[n], "model-00002-of-00062.safetensors")

    def test_fp8_pairs_never_split(self):
        """Every weight_scale_inv sits in the SAME shard as its weight —
        per-shard pair validation is sound."""
        for name, shard in self.wm.items():
            if name.endswith(".weight_scale_inv"):
                base = name[:-len("_scale_inv")]
                self.assertIn(base, self.wm, name)
                self.assertEqual(self.wm[base], shard, name)
        # 37,338 pairs: 12,384x3 experts + 43x3 shared + 3x3 dense
        # + 12x4 DSA (q_a/q_b/kv_a/o on 11 layers + MTP)
        n_scales = sum(1 for n in self.wm
                       if n.endswith(".weight_scale_inv"))
        self.assertEqual(n_scales, 12384 * 3 + 43 * 3 + 3 * 3 + 12 * 4)
        self.assertEqual(n_scales, 37338)

    def test_layer_inventories(self):
        layer_of = lambda n: int(re.match(
            r"model\.language_model\.layers\.(\d+)\.", n).group(1))
        idx_layers = sorted({layer_of(n) for n in self.wm
                             if ".self_attn.indexer.wq_b.weight" in n})
        self.assertEqual(idx_layers,
                         [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45])
        kda_layers = sorted({layer_of(n) for n in self.wm
                             if ".self_attn.A_log" in n})
        self.assertEqual(len(kda_layers), 34)
        self.assertEqual(sorted(set(range(45)) - set(idx_layers[:-1])),
                         kda_layers)
        hc_layers = sorted({layer_of(n) for n in self.wm
                            if ".hc_attn_fn" in n})
        self.assertEqual(hc_layers, list(range(45)))   # no hc on MTP
        moe_layers = sorted({layer_of(n) for n in self.wm
                             if ".mlp.gate.weight" in n})
        self.assertEqual(moe_layers, list(range(3, 46)))
        dense_layers = sorted({layer_of(n) for n in self.wm
                               if n.startswith(
                                   "model.language_model.layers.")
                               and ".mlp.gate_proj.weight" in n})
        self.assertEqual(dense_layers, [0, 1, 2])

    def test_mtp_inventory(self):
        l45 = {n.split("layers.45.", 1)[1] for n in self.wm
               if n.startswith("model.language_model.layers.45.")}
        for marker in ("eh_proj.weight", "enorm.weight", "hnorm.weight",
                       "shared_head.norm.weight"):
            self.assertIn(marker, l45)
        self.assertFalse(any(n.startswith("hc_") for n in l45))
        self.assertIn("self_attn.indexer.wq_b.weight", l45)
        self.assertIn("mlp.gate.e_score_correction_bias", l45)
        self.assertEqual(
            sum(1 for n in l45 if n.startswith("mlp.experts.")),
            288 * 6)

    def test_slab_arithmetic_of_record(self):
        """Pinned numbers (tests/m1/README.md). Deviation from Phase A:
        STATUS/ARCHITECTURE quoted slab ~= 25,190,400 B and "~297 GiB";
        the config-derived exact values are 25,171,968 B and ~290.32 GiB
        (Phase A appears to have counted 16 B per F32 scale element).
        The converter validates scale shapes against the 128x128 rule at
        conversion time, so a real-shard disagreement fails loudly."""
        H = self.text["hidden_size"]               # 4096
        M = self.text["moe_intermediate_size"]     # 2048
        q = self.cfg["quantization_config"]
        self.assertEqual(q["weight_block_size"], [128, 128])
        self.assertEqual(q["fmt"], "e4m3")
        weights = 3 * M * H                       # FP8 codes, 1 B each
        scales = 3 * (M // 128) * (H // 128) * 4  # F32
        self.assertEqual(weights, 25165824)
        self.assertEqual(scales, 6144)
        slab = weights + scales
        self.assertEqual(slab, 25171968)          # 24.0059 MiB
        self.assertAlmostEqual(slab / 1024**2, 24.0059, places=4)
        total = slab * 12384
        self.assertEqual(total, 311729651712)
        self.assertAlmostEqual(total / 1024**3, 290.32, places=2)
        # Bookkeeping: the dense+vision remainder of the 305.78 GiB
        # checkpoint is ~15.5 GiB (embed+head 2.36 GiB, KDA smalls,
        # DSA, dense MLPs, router, mHC, vision tower).
        rest = self.index["metadata"]["total_size"] - total
        self.assertTrue(14 * 1024**3 < rest < 17 * 1024**3,
                        f"remainder {rest / 1024**3:.2f} GiB unexpected")
        print(f"\nindex realism: 12,384 experts x {slab:,} B = "
              f"{total / 1024**3:.2f} GiB; dense+vision remainder "
              f"{rest / 1024**3:.2f} GiB of "
              f"{self.index['metadata']['total_size'] / 1024**3:.2f} GiB")

    def test_group_classification(self):
        """Every text tensor routes to main or mtp by the 45-layer rule;
        lm_head/embed land in main."""
        n_main = self.text["num_hidden_layers"]
        self.assertEqual(n_main, 45)
        for name in self.wm:
            if name.startswith(apus_convert.VISION_PREFIX):
                continue
            stripped = apus_convert.output_name(name)
            group = apus_convert.classify_group(stripped, n_main)
            layer = apus_convert.layer_of(stripped)
            if layer is not None and layer >= n_main:
                self.assertEqual(group, "mtp", stripped)
            else:
                self.assertEqual(group, "main", stripped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
