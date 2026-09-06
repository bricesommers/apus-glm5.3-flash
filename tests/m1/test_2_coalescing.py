"""M1 test 2 — coalesced per-expert layout (incl. split experts).

Parses the output shard headers manually and asserts, for every expert:

  * its 6 tensors {gate,up,down}_proj.{weight,weight_scale_inv} appear
    consecutively in the shard header, in that exact order,
  * their data regions are contiguous and adjacent (end == next start),
  * the whole slab lives in one shard — including the two experts the
    fixture splits across INPUT shards (the converter defers and
    reassembles them),
  * the manifest's slab record (shard, offset, nbytes) matches the
    actual header offsets exactly,
  * slab size equals the expected per-expert byte count.
"""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import stutil
import convert as apus_convert

TARGET_BYTES = 64 * 1024
EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")


class TestCoalescing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "src")
        cls.dst = os.path.join(cls.tmp.name, "out")
        os.makedirs(cls.src)
        fixtures.make_fixture_tree(cls.src)
        conv = apus_convert.Converter(cls.src, cls.dst,
                                      target_bytes=TARGET_BYTES)
        conv.convert()
        conv.finalize()
        with open(os.path.join(cls.dst, "apus.index.json")) as f:
            cls.manifest = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _experts_from_headers(self):
        """Rebuild per-expert layout facts from the raw output headers."""
        experts = {}  # (layer, eid) -> {shard, members, positions}
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, data_start = stutil.read_shard(
                os.path.join(self.dst, shard))
            names = [n for n in header if n != "__metadata__"]
            for pos, name in enumerate(names):
                m = EXPERT_RE.match(name)
                if not m:
                    continue
                key = (int(m.group(1)), int(m.group(2)))
                begin, end = header[name]["data_offsets"]
                rec = experts.setdefault(
                    key, {"shard": shard, "members": [], "positions": []})
                self.assertEqual(rec["shard"], shard,
                                 f"expert {key} straddles output shards")
                rec["members"].append(
                    (f"{m.group(3)}.{m.group(4)}", data_start + begin,
                     end - begin))
                rec["positions"].append(pos)
        return experts

    def test_slab_count(self):
        experts = self._experts_from_headers()
        n_moe = len(fixtures.MOE_LAYERS) + 1  # + MTP layer
        self.assertEqual(len(experts), n_moe * fixtures.N_EXPERTS)

    def test_slabs_contiguous_and_adjacent(self):
        experts = self._experts_from_headers()
        for key, rec in experts.items():
            positions = rec["positions"]
            self.assertEqual(
                positions,
                list(range(positions[0], positions[0] + 6)),
                f"{key}: slab members not adjacent in header")
            by_pos = [m for _, m in sorted(
                zip(positions, rec["members"]), key=lambda pm: pm[0])]
            self.assertEqual([s for s, _, _ in by_pos],
                             fixtures.SLAB_ORDER,
                             f"{key}: wrong intra-slab order")
            for (s0, off0, nb0), (s1, off1, nb1) in zip(by_pos,
                                                        by_pos[1:]):
                self.assertEqual(off0 + nb0, off1,
                                 f"{key}: gap between {s0} and {s1}")
            total = sum(nb for _, _, nb in by_pos)
            self.assertEqual(total, fixtures.PER_EXPERT_BYTES,
                             f"{key}: unexpected per-expert byte count")

    def test_split_experts_reassembled(self):
        """The two experts split across INPUT shards must come out as
        ordinary contiguous slabs."""
        experts = self._experts_from_headers()
        for key in (fixtures.SPLIT_MAIN, fixtures.SPLIT_MTP):
            self.assertIn(key, experts)
            rec = experts[key]
            self.assertEqual(len(rec["members"]), 6, key)
            positions = rec["positions"]
            self.assertEqual(
                positions,
                list(range(positions[0], positions[0] + 6)), key)
        # split main expert in a main shard, split MTP expert in an
        # apus-mtp shard
        self.assertTrue(
            experts[fixtures.SPLIT_MAIN]["shard"].startswith("apus-0"))
        self.assertTrue(
            experts[fixtures.SPLIT_MTP]["shard"].startswith("apus-mtp-"))

    def test_manifest_slab_records_match_headers(self):
        experts = self._experts_from_headers()
        slabs = {}
        for s in self.manifest["expert_slabs"]:
            layer = int(s["block"].split(".")[1])
            slabs[(layer, s["expert"])] = s
        self.assertEqual(set(slabs), set(experts))
        for key, rec in experts.items():
            by_pos = [m for _, m in sorted(
                zip(rec["positions"], rec["members"]),
                key=lambda pm: pm[0])]
            slab_start = by_pos[0][1]
            slab_bytes = sum(nb for _, _, nb in by_pos)
            rec_m = slabs[key]
            self.assertEqual(rec_m["shard"], rec["shard"], key)
            self.assertEqual(rec_m["offset"], slab_start, key)
            self.assertEqual(rec_m["nbytes"], slab_bytes, key)

    def test_manifest_slab_member_order_declared(self):
        self.assertEqual(self.manifest["expert_slab_members"],
                         fixtures.SLAB_ORDER)

    def test_manifest_offsets_match_headers_for_all_tensors(self):
        """Every tensor_map entry must point at the same bytes the output
        shard header describes."""
        tmap = self.manifest["tensor_map"]
        n = 0
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            header, data_start = stutil.read_shard(
                os.path.join(self.dst, shard))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                rec = tmap[name]
                self.assertEqual(rec["shard"], shard, name)
                self.assertEqual(rec["offset"],
                                 data_start + meta["data_offsets"][0],
                                 name)
                self.assertEqual(rec["nbytes"],
                                 meta["data_offsets"][1]
                                 - meta["data_offsets"][0], name)
                n += 1
        self.assertEqual(n, len(tmap))


if __name__ == "__main__":
    unittest.main()
