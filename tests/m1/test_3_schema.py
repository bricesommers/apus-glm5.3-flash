"""M1 test 3 — schema validation + E4M3 dequant reference.

Two parts:

1. The converter's fail-loudly validation: FP8/scale_inv pairing (dtype,
   128x128 block shape, no orphans, no splits), E4M3 NaN code refusal
   (0x7F/0xFF), name whitelist, MTP structural rules, vision strip.

2. A numpy E4M3 + F32 scale_inv dequant reference for the M3 C kernel:
   S.EEEE.MMM (bias 7), subnormals at e==0, NaN only at 0x7F/0xFF, one
   F32 scale per 128x128 block. Hand-computed and boundary cases. (The
   full dequant -> BF16-RNE -> BF16-matmul numerics contract is pinned
   in tests/m0/README.md; here we pin the storage-level decode.)
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import stutil
import convert as apus_convert

LANG = apus_convert.LANG_PREFIX


def e4m3_decode(b):
    """E4M3 (S.EEEE.MMM, exp bias 7) byte -> float, NaN at 0x7F/0xFF."""
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0xF and m == 0x7:
        return float("nan")
    if e == 0:
        return s * (m / 8.0) * 2.0 ** -6     # subnormals
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def dequant_block(codes, scales):
    """codes u8 [O,K], scales f32 [ceil(O/128), ceil(K/128)] -> f32 [O,K]
    (q x scale_inv per 128x128 block — the normative dequant)."""
    O, K = codes.shape
    out = np.empty((O, K), np.float32)
    lut = np.array([e4m3_decode(i) for i in range(256)], np.float32)
    vals = lut[codes]
    for ob in range(0, O, 128):
        for kb in range(0, K, 128):
            out[ob:ob + 128, kb:kb + 128] = (
                vals[ob:ob + 128, kb:kb + 128]
                * scales[ob // 128, kb // 128])
    return out


class TestE4M3Reference(unittest.TestCase):
    def test_landmarks(self):
        self.assertEqual(e4m3_decode(0x00), 0.0)
        self.assertEqual(e4m3_decode(0x80), 0.0)        # -0
        self.assertTrue(math.copysign(1.0, e4m3_decode(0x80)) < 0)
        self.assertEqual(e4m3_decode(0x38), 1.0)        # e=7: 1.0
        self.assertEqual(e4m3_decode(0x30), 0.5)
        self.assertEqual(e4m3_decode(0x40), 2.0)
        self.assertEqual(e4m3_decode(0xC0), -2.0)
        self.assertEqual(e4m3_decode(0x7E), 448.0)      # max normal
        self.assertEqual(e4m3_decode(0xFE), -448.0)
        self.assertEqual(e4m3_decode(0x01), 2.0 ** -9)  # min subnormal
        self.assertEqual(e4m3_decode(0x07), 7 * 2.0 ** -9)
        self.assertTrue(math.isnan(e4m3_decode(0x7F)))
        self.assertTrue(math.isnan(e4m3_decode(0xFF)))
        # exactly two NaN codes in the whole encoding
        nans = [b for b in range(256) if math.isnan(e4m3_decode(b))]
        self.assertEqual(nans, [0x7F, 0xFF])

    def test_monotonic_over_non_nan_codes(self):
        # sign-magnitude: positives ascend with the byte, negatives descend
        vals = [e4m3_decode(b) for b in range(0x00, 0x7F)]
        self.assertEqual(vals, sorted(vals))
        vals = [e4m3_decode(b) for b in range(0x80, 0xFF)]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_block_dequant_hand_case(self):
        codes = np.full((128, 256), 0x38, np.uint8)     # all 1.0
        scales = np.array([[2.0, 0.5]], np.float32)     # [1, 2] blocks
        out = dequant_block(codes, scales)
        self.assertTrue(np.all(out[:, :128] == 2.0))
        self.assertTrue(np.all(out[:, 128:] == 0.5))
        self.assertEqual(out.dtype, np.float32)

    def test_block_dequant_subnormal_and_sign(self):
        codes = np.zeros((1, 128), np.uint8)
        codes[0, 0] = 0x01          # 2^-9
        codes[0, 1] = 0xFE          # -448
        scales = np.array([[4.0]], np.float32)
        out = dequant_block(codes, scales)
        self.assertEqual(out[0, 0], 2.0 ** -7)
        self.assertEqual(out[0, 1], -1792.0)
        self.assertEqual(out[0, 2], 0.0)


class TestSchemaValidation(unittest.TestCase):
    """Each case builds a tiny bespoke shard tree and expects a clear
    failure (or, for the vision case, a clean strip)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "out")
        os.makedirs(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_tree(self, tensors, n_main=5):
        """tensors: [(name, dtype, shape, payload)]. Returns shard name."""
        stutil.write_shard(os.path.join(self.src, "model-00001.safetensors"),
                           tensors)
        import json
        with open(os.path.join(self.src, "config.json"), "w") as f:
            json.dump({"text_config": {"num_hidden_layers": n_main}}, f)
        return "model-00001.safetensors"

    def _convert(self):
        conv = apus_convert.Converter(self.src, self.dst,
                                      target_bytes=64 * 1024)
        conv.convert()

    def _rand(self, dtype, shape, rng_seed=7):
        rng = np.random.RandomState(rng_seed)
        n = stutil.tensor_nbytes(dtype, shape)
        if dtype == "F8_E4M3":
            vals = rng.randint(0, 254, size=n).astype(np.uint8)
            vals[vals >= 0x7F] += 1
            return vals.tobytes()
        return rng.randint(0, 256, size=n, dtype=np.uint8).tobytes()

    def _fp8(self, name, rows, cols, payload=None, scale_dtype="F32",
             scale_shape=None):
        w = payload if payload is not None else self._rand("F8_E4M3",
                                                           [rows, cols])
        if scale_shape is None:
            scale_shape = [(rows + 127) // 128, (cols + 127) // 128]
        return [
            (f"{name}.weight", "F8_E4M3", [rows, cols], w),
            (f"{name}.weight_scale_inv", scale_dtype, scale_shape,
             self._rand(scale_dtype, scale_shape)),
        ]

    # -- FP8 pairing --------------------------------------------------------

    def test_orphan_scale_refused(self):
        self._write_tree([
            (f"{LANG}layers.0.mlp.gate_proj.weight_scale_inv", "F32",
             [1, 1], self._rand("F32", [1, 1])),
        ])
        with self.assertRaisesRegex(ValueError, "orphan"):
            self._convert()

    def test_scale_on_bf16_weight_refused(self):
        # dense-MLP name (whitelisted with a scale) but BF16 payload:
        # the pair check must reject the dtype combination.
        self._write_tree([
            (f"{LANG}layers.0.mlp.gate_proj.weight", "BF16", [160, 64],
             self._rand("BF16", [160, 64])),
            (f"{LANG}layers.0.mlp.gate_proj.weight_scale_inv", "F32",
             [2, 1], self._rand("F32", [2, 1])),
        ])
        with self.assertRaisesRegex(ValueError, "paired with BF16"):
            self._convert()

    def test_wrong_scale_shape_refused(self):
        t = self._fp8(f"{LANG}layers.0.mlp.gate_proj", 64, 256,
                      scale_shape=[1, 1])   # needs [1, 2]
        self._write_tree(t)
        with self.assertRaisesRegex(ValueError, "scale shape"):
            self._convert()

    def test_wrong_scale_dtype_refused(self):
        t = self._fp8(f"{LANG}layers.0.mlp.gate_proj", 64, 64,
                      scale_dtype="BF16")
        self._write_tree(t)
        with self.assertRaisesRegex(ValueError, "expected F32"):
            self._convert()

    def test_fp8_without_scale_refused(self):
        self._write_tree([
            (f"{LANG}layers.0.mlp.gate_proj.weight", "F8_E4M3", [64, 64],
             self._rand("F8_E4M3", [64, 64])),
        ])
        with self.assertRaisesRegex(ValueError, "without its"):
            self._convert()

    def test_fp8_scale_pairing_accepts_ceil_shapes(self):
        """160 rows -> 2 scale blocks; the pairing must accept ceil
        division exactly like the fixture's dense MLP."""
        t = self._fp8(f"{LANG}layers.0.mlp.gate_proj", 160, 200)
        self._write_tree(t)
        self._convert()   # must not raise

    # -- E4M3 NaN refusal -----------------------------------------------------

    def test_nan_code_7f_refused(self):
        payload = bytearray(self._rand("F8_E4M3", [64, 64]))
        payload[123] = 0x7F
        t = self._fp8(f"{LANG}layers.0.mlp.gate_proj", 64, 64,
                      payload=bytes(payload))
        self._write_tree(t)
        with self.assertRaisesRegex(ValueError, "NaN"):
            self._convert()

    def test_nan_code_ff_refused(self):
        payload = bytearray(self._rand("F8_E4M3", [64, 64]))
        payload[4095] = 0xFF
        t = self._fp8(f"{LANG}layers.0.mlp.gate_proj", 64, 64,
                      payload=bytes(payload))
        self._write_tree(t)
        with self.assertRaisesRegex(ValueError, "NaN"):
            self._convert()

    # -- naming whitelist ------------------------------------------------------

    def test_unknown_toplevel_name_refused(self):
        self._write_tree([
            ("foo.bar", "BF16", [4], self._rand("BF16", [4])),
        ])
        with self.assertRaisesRegex(ValueError, "unrecognized tensor"):
            self._convert()

    def test_unknown_layer_tensor_refused(self):
        self._write_tree([
            (f"{LANG}layers.0.self_attn.made_up.weight", "BF16", [4],
             self._rand("BF16", [4])),
        ])
        with self.assertRaisesRegex(ValueError, "known glm5_next"):
            self._convert()

    def test_vision_stripped_not_refused(self):
        self._write_tree([
            ("model.visual.blocks.0.attn.qkv.weight", "BF16", [24, 8],
             self._rand("BF16", [24, 8])),
            ("model.visual.whatever.new_layer.weight", "F32", [4],
             self._rand("F32", [4])),
            (f"{LANG}norm.weight", "BF16", [64], self._rand("BF16", [64])),
        ])
        conv = apus_convert.Converter(self.src, self.dst,
                                      target_bytes=64 * 1024)
        conv.convert()
        conv.finalize()
        self.assertEqual(conv.state["stripped"]["vision_tensors"], 2)

    # -- MTP structural rules ---------------------------------------------------

    def test_hc_in_mtp_layer_refused(self):
        self._write_tree([
            (f"{LANG}layers.5.hc_attn_fn", "BF16", [24, 256],
             self._rand("BF16", [24, 256])),
        ], n_main=5)
        with self.assertRaisesRegex(ValueError, "MTP layer"):
            self._convert()

    def test_eh_proj_in_main_layer_refused(self):
        self._write_tree([
            (f"{LANG}layers.2.eh_proj.weight", "BF16", [64, 128],
             self._rand("BF16", [64, 128])),
        ], n_main=5)
        with self.assertRaisesRegex(ValueError, "MTP-only"):
            self._convert()

    def test_incomplete_expert_without_index_refused(self):
        """Flat-dir mode (no weight_map): a partial expert cannot be
        deferred -> loud failure."""
        pre = f"{LANG}layers.3.mlp.experts.0"
        t = []
        for suffix in ("gate_proj", "up_proj"):   # down_proj missing
            t += self._fp8(f"{pre}.{suffix}", 8, 64)
        self._write_tree(t)
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self._convert()


if __name__ == "__main__":
    unittest.main()
