"""M1 test 1 — byte identity + naming + vision strip.

Convert the synthetic glm5_next fixture checkpoint, then compare EVERY
kept output tensor's bytes against the source bytes. Any difference
fails. Also checks:

  * dtype and shape are preserved verbatim,
  * output names follow the container convention (language_model prefix
    dropped, lm_head.weight verbatim — the M0 oracle's naming),
  * vision tensors are absent from the output and counted in the
    manifest,
  * the manifest agrees with the output shard headers,
  * numpy-readable tensors cross-check via the safetensors library
    (allowed for reads where dtypes permit).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import fixtures
import stutil
import convert as apus_convert

TARGET_BYTES = 64 * 1024  # small shards -> exercises multi-shard packing


class TestByteIdentity(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _manifest(self):
        with open(os.path.join(self.dst, "apus.index.json")) as f:
            return json.load(f)

    def test_tensor_count_preserved_minus_vision(self):
        n_src = n_vision = 0
        for shard in os.listdir(self.src):
            if not shard.endswith(".safetensors"):
                continue
            header, _ = stutil.read_shard(os.path.join(self.src, shard))
            n_src += len(header)
            n_vision += sum(1 for n in header
                            if n.startswith(fixtures.VIS))
        self.assertEqual(n_vision, fixtures.N_VISION)
        manifest = self._manifest()
        self.assertEqual(manifest["ntensors"], n_src - n_vision)
        self.assertEqual(len(manifest["tensor_map"]), n_src - n_vision)
        self.assertEqual(manifest["stripped"]["vision_tensors"], n_vision)
        self.assertGreater(manifest["stripped"]["vision_bytes"], 0)

    def test_every_tensor_byte_identical(self):
        manifest = self._manifest()
        tmap = manifest["tensor_map"]
        nchecked = 0
        for shard in sorted(os.listdir(self.src)):
            if not shard.endswith(".safetensors"):
                continue
            spath = os.path.join(self.src, shard)
            src_tensors = stutil.read_tensor_bytes(spath)
            sheader, _ = stutil.read_shard(spath)
            for src_name, payload in src_tensors.items():
                if src_name.startswith(fixtures.VIS):
                    continue
                name = apus_convert.output_name(src_name)
                rec = tmap[name]
                self.assertEqual(rec["dtype"], sheader[src_name]["dtype"],
                                 name)
                self.assertEqual(rec["shape"], sheader[src_name]["shape"],
                                 name)
                self.assertEqual(rec["nbytes"], len(payload), name)
                opath = os.path.join(self.dst, rec["shard"])
                with open(opath, "rb") as f:
                    f.seek(rec["offset"])
                    out = f.read(rec["nbytes"])
                self.assertEqual(out, payload,
                                 f"{name}: bytes differ after conversion")
                nchecked += 1
        self.assertEqual(nchecked, len(tmap))

    def test_naming_matches_oracle_convention(self):
        """No output name keeps the language_model prefix; lm_head stays
        top-level; every name is one the M0 oracle can ask for."""
        manifest = self._manifest()
        for name in manifest["tensor_map"]:
            self.assertFalse(name.startswith("model."), name)
            self.assertNotIn("language_model", name)
            self.assertNotIn("visual", name)
        self.assertIn("lm_head.weight", manifest["tensor_map"])
        self.assertIn("embed_tokens.weight", manifest["tensor_map"])
        # oracle fixture names, spot-checked
        self.assertIn("layers.3.self_attn.q_a_proj.weight",
                      manifest["tensor_map"])
        self.assertIn("layers.3.self_attn.q_a_proj.weight_scale_inv",
                      manifest["tensor_map"])
        self.assertIn("layers.0.self_attn.q_conv1d.weight",
                      manifest["tensor_map"])
        self.assertIn(f"layers.{fixtures.MTP_LAYER}.eh_proj.weight",
                      manifest["tensor_map"])

    def test_mtp_in_own_shard_group(self):
        manifest = self._manifest()
        groups = manifest["shard_groups"]
        self.assertEqual(sorted(groups), ["main", "mtp"])
        for shard in groups["mtp"]:
            self.assertTrue(shard.startswith("apus-mtp-"), shard)
        for name, rec in manifest["tensor_map"].items():
            layer = apus_convert.layer_of(name)
            want = "apus-mtp-" if layer is not None \
                and layer >= fixtures.N_LAYERS_MAIN else "apus-"
            self.assertTrue(rec["shard"].startswith(want),
                            f"{name} in {rec['shard']}")

    def test_verify_helper(self):
        n = apus_convert.verify_source(self.src, self.dst,
                                       log=lambda m: None)
        self.assertGreater(n, 0)

    def test_safetensors_lib_crosscheck(self):
        """The safetensors library must parse our manually-written output
        shards."""
        from safetensors import safe_open
        n = 0
        for shard in sorted(os.listdir(self.dst)):
            if not shard.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(self.dst, shard),
                           framework="numpy") as f:
                for name in f.keys():
                    meta = f.get_slice(name)
                    self.assertIn(meta.get_dtype(),
                                  ("BF16", "F32", "F8_E4M3"))
                    n += 1
        self.assertGreater(n, 0)

    def test_std_index_for_oracle_loader(self):
        """finalize also writes model.safetensors.index.json — the file
        the M0 oracle's ShardSet and the C loader read."""
        with open(os.path.join(self.dst,
                               "model.safetensors.index.json")) as f:
            idx = json.load(f)
        manifest = self._manifest()
        self.assertEqual(set(idx["weight_map"]),
                         set(manifest["tensor_map"]))
        self.assertEqual(idx["metadata"]["total_size"],
                         sum(t["nbytes"]
                             for t in manifest["tensor_map"].values()))


if __name__ == "__main__":
    unittest.main()
