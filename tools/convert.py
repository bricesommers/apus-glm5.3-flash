#!/usr/bin/env python3
"""apus M1 — HF safetensors -> apus container converter (glm5_next schema).

Converts a directory of HuggingFace safetensors shards (zai-org/
GLM-5.3-Flash, model_type glm5_next) into the apus weight container
described in docs/ARCHITECTURE.md §6 and pinned by tests/m1/README.md:

  * Byte-identical copy of every kept tensor. Payloads are treated as raw
    bytes; this tool never requantizes, transcodes, or even interprets
    tensor data (the only exception: FP8-E4M3 payloads are scanned for the
    NaN codes 0x7F/0xFF and refused — see below).
  * Naming: the checkpoint's "model.language_model." prefix is DROPPED
    ("model.language_model.layers.3.self_attn.q_a_proj.weight" ->
    "layers.3.self_attn.q_a_proj.weight"), matching the M0 oracle's
    convention so M4/M5 cross-load the container directly. The top-level
    "lm_head.weight" keeps its name. "model.visual.*" tensors are
    STRIPPED (text-only scope) and counted in the manifest.
  * Coalesced per-expert layout: within each output shard, the 6 tensors
    of every routed expert {gate,up,down}_proj.{weight,weight_scale_inv}
    are contiguous and adjacent in that fixed order, so the engine
    fetches a whole expert with one pread (24.0059 MiB slab on the real
    checkpoint: 3 x 8,388,608 B FP8 + 3 x 2,048 B F32 scales).
  * MTP kept: layers.45.* (the deferred NextN head) converts into a
    separate "mtp" shard group for M8+ lazy load.
  * Output shards are ordinary safetensors files, written MANUALLY
    (8-byte little-endian header length + JSON header + raw data).
    safetensors.numpy is not used for writing: numpy has no
    F8_E4M3/BF16 dtypes and we need exact control over tensor order and
    byte identity.
  * Manifest `apus.index.json`: format version, config hash, full tensor
    map (name -> shard, absolute file offset, nbytes, dtype, shape),
    per-expert slab records, and strip statistics.

Validation (fail loudly, never guess):
  * Every tensor name must match the glm5_next whitelist (verified against
    the real index by tests/m1/test_5_index_realism.py); anything else is
    naming drift and aborts the conversion.
  * Every F8_E4M3 `X.weight` must pair with an F32 `X.weight_scale_inv`
    of shape [ceil(O/128), ceil(K/128)] in the SAME input shard; orphan or
    mis-shaped scales abort. (Real index: 37,338 pairs, none split.)
  * F8_E4M3 payloads containing a NaN code (0x7F or 0xFF) abort the
    conversion. These never occur in a trained checkpoint; if they ever
    do, the source is corrupt and we refuse to repack it silently.
  * MTP structure: layers >= num_hidden_layers carry no hc_* tensors;
    eh_proj/enorm/hnorm/shared_head.norm appear only there.

Split experts (real-checkpoint fact)
------------------------------------
The real index splits exactly ONE expert across input shards: MTP expert
(45, 197) — its up_proj pair lives in model-00002, the rest in
model-00001. Slab assembly therefore tolerates partial slabs: when an
input shard holds only part of an expert, the whole slab is DEFERRED
(recorded in the state's pending_slabs) and appended once every member's
input shard has been converted. Resolution order is a deterministic
function of the input set, so one-shot and shard-by-shard (download
driver) conversions produce byte-identical output. The download driver
must not delete a source shard while a pending slab references it (see
tools/download.py).

Resumability / crash safety
---------------------------
Conversion is driven input-shard by input-shard. A state file
(`apus.convert.state.json` in the output dir) is rewritten atomically
after every single tensor append, after every shard seal, after every
pending-slab update, and after every finished input shard. Re-running
after an interruption:

  1. Sealed output shards are verify-before-trust checked (size + header
     hash recorded in the state).
  2. The open output shard is validated against the state: bytes
     committed in the state must be present on disk; any tail beyond the
     last commit (a torn tensor write) is truncated and rewritten.
  3. Completed input shards are skipped entirely; pending slabs
     re-resolve idempotently (already-written members are skipped, the
     rest land contiguously at the open shard's tail).

The append sequence is a deterministic function of the input set, so an
interrupted+resumed run produces byte-identical output to an
uninterrupted run (covered by tests/m1/test_4_resume.py).

Output shard format detail: shards are created with a fixed-capacity
header region (HEADER_RESERVE bytes) holding a placeholder JSON document.
This lets us append tensor data streaming-style and "seal" the shard
later by simply rewriting the header region in place — no data copies,
and the file is a structurally valid safetensors file at every point in
time. Sealing pads the JSON with trailing spaces, which the safetensors
format permits.

Usage:
    python tools/convert.py convert  SRC_DIR DST_DIR [--shard NAME ...]
                                     [--target-bytes N]
    python tools/convert.py verify   SRC_DIR DST_DIR [--shard NAME ...]
    python tools/convert.py finalize SRC_DIR DST_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile

FORMAT_VERSION = 2  # glm5_next container schema (v1 was DeepSeek-V4)

# Fixed header capacity of every output shard. ~150 bytes of JSON per
# tensor entry => room for well over 100k tensors per shard (a 5 GiB
# shard holds ~220 expert slabs ~= 1.3k tensors); validated on seal.
HEADER_RESERVE = 16 * 1024 * 1024

# Target output shard size (~5 GB per docs/ARCHITECTURE.md §6). A shard
# is sealed before a tensor/expert-slab append that would exceed this, so
# the actual size stays within target + one expert slab (~24 MiB).
DEFAULT_TARGET_BYTES = 5 * 1024**3

COPY_CHUNK = 8 * 1024 * 1024

STATE_FILE = "apus.convert.state.json"
MANIFEST_FILE = "apus.index.json"
INDEX_NAME = "model.safetensors.index.json"

MAIN_PREFIX = "apus"
MTP_PREFIX = "apus-mtp"

# FP8 block size from the checkpoint quantization_config
# (weight_block_size: [128, 128], fmt: e4m3, F32 weight_scale_inv).
FP8_BLOCK = 128

# E4M3 NaN encodings (S.1111.111). Refused, never copied.
FP8_NAN_CODES = (0x7F, 0xFF)

# Checkpoint naming (real index, 2026-08): text tensors carry this
# prefix; the container drops it to match the M0 oracle convention.
LANG_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
# Top-level names kept verbatim (lm_head sits outside language_model in
# the real checkpoint — untied, BF16).
TOPLEVEL_KEEP = ("lm_head.weight",)

# layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{weight,weight_scale_inv}
EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
# Fixed intra-slab order: gate, up, down; weight before its scale.
SLAB_MEMBERS = tuple(
    f"{p}.{kind}" for p in ("gate_proj", "up_proj", "down_proj")
    for kind in ("weight", "weight_scale_inv")
)

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")

# glm5_next structural whitelist on STRIPPED names (drift guard; the full
# pattern set is pinned against the real index in
# tests/m1/test_5_index_realism.py).
_ATTN_TAIL = (
    r"self_attn\.(?:"
    # KDA (34 layers; all BF16 except A_log/dt_bias F32)
    r"A_log|dt_bias|o_norm\.weight|"
    r"[qkv]_proj\.weight|[qkv]_conv1d\.weight|"
    r"b_proj\.weight|[fg]_[ab]_proj\.weight|"
    # DSA (11 layers + MTP; q_a/q_b/kv_a/o FP8+scale, kv_b BF16)
    r"o_proj\.(?:weight|weight_scale_inv)|"
    r"q_a_proj\.(?:weight|weight_scale_inv)|q_a_layernorm\.weight|"
    r"q_b_proj\.(?:weight|weight_scale_inv)|"
    r"kv_a_proj_with_mqa\.(?:weight|weight_scale_inv)|"
    r"kv_a_layernorm\.weight|kv_b_proj\.weight|"
    # Lightning indexer (all BF16)
    r"indexer\.(?:wq_b\.weight|wk\.weight|k_norm\.weight|k_norm\.bias|"
    r"weights_proj\.weight|index_kpool_compress_ape|"
    r"index_kpool_compress_gate))"
)
_MLP_TAIL = (
    r"mlp\.(?:"
    # dense MLP, layers 0-2 (FP8+scale)
    r"(?:gate|up|down)_proj\.(?:weight|weight_scale_inv)|"
    # MoE router (BF16 weight, F32 bias)
    r"gate\.weight|gate\.e_score_correction_bias|"
    # shared expert (FP8+scale)
    r"shared_experts\.(?:gate|up|down)_proj\.(?:weight|weight_scale_inv))"
)
_HC_TAIL = r"hc_(?:attn|ffn)_(?:fn|base|scale)"
_MTP_TAIL = (
    r"eh_proj\.weight|enorm\.weight|hnorm\.weight|shared_head\.norm\.weight"
)
_NORM_TAIL = r"(?:input_layernorm|post_attention_layernorm)\.weight"

KNOWN_RE = re.compile(
    r"^(?:embed_tokens\.weight|norm\.weight|lm_head\.weight|"
    r"layers\.(\d+)\.(?:" + "|".join(
        (_NORM_TAIL, _ATTN_TAIL, _MLP_TAIL, _HC_TAIL, _MTP_TAIL)) + r"))$"
)

_MTP_ONLY_RE = re.compile(
    r"^layers\.\d+\.(?:eh_proj\.weight|enorm\.weight|hnorm\.weight|"
    r"shared_head\.norm\.weight)$"
)


def ceil_div(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------
# safetensors header reading (source shards)
# --------------------------------------------------------------------------

class SrcTensor:
    """One tensor in a source shard: location + self-description, no data."""
    __slots__ = ("name", "src_name", "dtype", "shape", "src_path",
                 "file_offset", "nbytes")

    def __init__(self, name, dtype, shape, src_path, file_offset, nbytes,
                 src_name=None):
        self.name = name          # output (stripped) name
        self.src_name = src_name or name   # original checkpoint name
        self.dtype = dtype
        self.shape = shape
        self.src_path = src_path
        self.file_offset = file_offset  # absolute offset in src_path
        self.nbytes = nbytes


def read_st_header(path):
    """Parse a safetensors header. Returns dict name -> SrcTensor.

    Only the header is read; tensor payloads are never touched here.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (too small)")
        (hlen,) = struct.unpack("<Q", raw)
        hjson = f.read(hlen)
        if len(hjson) != hlen:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(hjson)
    data_start = 8 + hlen
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        tensors[name] = SrcTensor(
            name, meta["dtype"], list(meta["shape"]), path,
            data_start + begin, end - begin,
        )
    return tensors


def list_input_shards(src_dir):
    """Input shard file names in deterministic processing order.

    Uses model.safetensors.index.json when present (the real checkpoint
    layout), else every *.safetensors file in the directory.
    """
    index_path = os.path.join(src_dir, INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))
    return sorted(
        n for n in os.listdir(src_dir)
        if n.endswith(".safetensors") and not n.startswith("apus")
    )


def config_hash(src_dir):
    """Stable hash identifying the model configuration being converted."""
    for name in ("config.json", INDEX_NAME):
        path = os.path.join(src_dir, name)
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(COPY_CHUNK), b""):
                    h.update(chunk)
            return f"sha256:{h.hexdigest()}"
    return "sha256:none"


def load_n_main_layers(src_dir):
    """Number of main-model layers (45 on the real checkpoint); layers at
    or above this index form the MTP group. Reads the glm5_next
    multimodal wrapper config (text_config.num_hidden_layers) or a flat
    text config; falls back to 45 when no config is present."""
    path = os.path.join(src_dir, "config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)
        n = text.get("num_hidden_layers")
        if isinstance(n, int) and n > 0:
            return n
    return 45


# --------------------------------------------------------------------------
# Naming, classification, validation
# --------------------------------------------------------------------------

def is_vision(src_name):
    return src_name.startswith(VISION_PREFIX)


def output_name(src_name):
    """Checkpoint name -> container name (drops the language_model prefix)."""
    if src_name.startswith(LANG_PREFIX):
        return src_name[len(LANG_PREFIX):]
    if src_name in TOPLEVEL_KEEP:
        return src_name
    raise ValueError(
        f"{src_name}: unrecognized tensor name — expected "
        f"'{LANG_PREFIX}*', '{VISION_PREFIX}*', or one of "
        f"{list(TOPLEVEL_KEEP)}; refusing to guess (checkpoint schema "
        f"drift?)"
    )


def layer_of(name):
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def validate_structure(name):
    """Enforce the glm5_next whitelist on a stripped tensor name."""
    if EXPERT_RE.match(name):
        return
    if not KNOWN_RE.match(name):
        raise ValueError(
            f"{name}: does not match any known glm5_next tensor pattern; "
            f"refusing to convert (checkpoint schema drift?)"
        )


def expected_scale_shape(weight_shape):
    """F32 weight_scale_inv shape for an FP8 weight: one scale per
    FP8_BLOCK x FP8_BLOCK block."""
    if len(weight_shape) != 2:
        raise ValueError(
            f"FP8 weight shape {weight_shape} is not 2-D; the glm5_next "
            f"schema only has 2-D FP8 weights"
        )
    return [ceil_div(weight_shape[0], FP8_BLOCK),
            ceil_div(weight_shape[1], FP8_BLOCK)]


def validate_fp8_pairs(tensors):
    """Every F8_E4M3 X.weight needs an F32 X.weight_scale_inv of the
    block shape in the same input shard; orphan/mis-typed/mis-shaped
    scales abort. `tensors` maps stripped name -> SrcTensor."""
    for name, t in sorted(tensors.items()):
        if name.endswith(".weight_scale_inv"):
            base = name[:-len(".weight_scale_inv")]
            if t.dtype != "F32":
                raise ValueError(
                    f"{name}: dtype {t.dtype}, expected F32 "
                    f"weight_scale_inv")
            w = tensors.get(base + ".weight")
            if w is None:
                raise ValueError(
                    f"{name}: orphan scale (no {base}.weight alongside); "
                    f"weight/scale pairs must not be split across input "
                    f"shards")
            if w.dtype != "F8_E4M3":
                raise ValueError(
                    f"{name}: paired with {w.dtype} weight, expected "
                    f"F8_E4M3")
            want = expected_scale_shape(w.shape)
            if t.shape != want:
                raise ValueError(
                    f"{name}: scale shape {t.shape}, expected {want} "
                    f"(128x128 blocks over {w.shape})")
    for name, t in sorted(tensors.items()):
        if t.dtype == "F8_E4M3":
            if not name.endswith(".weight"):
                raise ValueError(
                    f"{name}: F8_E4M3 tensor not named *.weight; "
                    f"unexpected in the glm5_next schema")
            if name[:-len(".weight")] + ".weight_scale_inv" not in tensors:
                raise ValueError(
                    f"{name}: F8_E4M3 weight without its "
                    f"weight_scale_inv pair in the same input shard")


def classify_group(name, n_main):
    """Route a stripped tensor name to its output shard group."""
    layer = layer_of(name)
    if layer is not None and layer >= n_main:
        return "mtp"
    return "main"


def validate_mtp_rules(tensors, n_main):
    """MTP layers (>= n_main) use plain residuals (no hc_*); the
    MTP-only tensors (eh_proj/enorm/hnorm/shared_head.norm) must not
    appear in main layers."""
    for name in tensors:
        layer = layer_of(name)
        if layer is None:
            continue
        is_mtp = layer >= n_main
        if is_mtp and re.match(r"layers\.\d+\.hc_(attn|ffn)_", name):
            raise ValueError(
                f"{name}: hc_* tensor in MTP layer {layer}; MTP uses "
                f"plain residuals — checkpoint schema drift?")
        if not is_mtp and _MTP_ONLY_RE.match(name):
            raise ValueError(
                f"{name}: MTP-only tensor in main layer {layer}; "
                f"checkpoint schema drift?")


# --------------------------------------------------------------------------
# Slab planning
# --------------------------------------------------------------------------

def slab_key(name):
    """'layers.3.17' style key for an expert tensor name, else None."""
    m = EXPERT_RE.match(name)
    if not m:
        return None
    return f"layers.{int(m.group(1))}.{int(m.group(2))}"


def slab_sort_key(key):
    """Numeric (layer, expert) ordering for a slab key."""
    _, layer, expert = key.split(".")
    return (int(layer), int(expert))


def slab_key_layer(key):
    return slab_sort_key(key)[0]


def plan_contribution(tensors):
    """Split one input shard's kept tensors into (ordered, deferred).

    `ordered` is the deterministic write order for this shard: complete
    expert slabs first (sorted by slab key, fixed SLAB_MEMBERS order
    inside, so every slab is contiguous in the output), then all other
    tensors sorted by name. `deferred` lists slab keys whose 6 members
    are NOT all in this shard (the real checkpoint splits exactly one
    expert across input shards); the caller verifies the missing members
    exist globally and records the slab as pending.
    """
    slabs = {}   # key -> {member_suffix: name}
    others = []
    for name in tensors:
        m = EXPERT_RE.match(name)
        if m:
            key = slab_key(name)
            slabs.setdefault(key, {})[f"{m.group(3)}.{m.group(4)}"] = name
        else:
            others.append(name)

    ordered = []
    deferred = []
    for key in sorted(slabs, key=slab_sort_key):
        members = slabs[key]
        missing = [s for s in SLAB_MEMBERS if s not in members]
        if missing:
            deferred.append((key, members, missing))
        else:
            ordered.extend(members[s] for s in SLAB_MEMBERS)
    ordered.extend(sorted(others))
    return ordered, deferred


# --------------------------------------------------------------------------
# Output shard writer
# --------------------------------------------------------------------------

def _placeholder_header():
    doc = json.dumps({"__metadata__": {"apus_state": "open"}}).encode()
    return doc + b" " * (HEADER_RESERVE - len(doc))


def _sealed_header_bytes(entries):
    """Final JSON header for a shard, padded with spaces to HEADER_RESERVE."""
    header = {}
    off = 0
    for e in entries:
        header[e["name"]] = {
            "dtype": e["dtype"],
            "shape": e["shape"],
            "data_offsets": [off, off + e["nbytes"]],
        }
        off += e["nbytes"]
    doc = json.dumps(header, separators=(",", ":")).encode()
    if len(doc) > HEADER_RESERVE:
        raise ValueError(
            f"shard header needs {len(doc)} bytes > HEADER_RESERVE "
            f"({HEADER_RESERVE}); raise HEADER_RESERVE in tools/convert.py"
        )
    return doc + b" " * (HEADER_RESERVE - len(doc))


def _header_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(8 + HEADER_RESERVE))
    return h.hexdigest()


class GroupStream:
    """Manages the output shard sequence of one group (main/mtp).

    The "open" shard is created lazily on the first append, appended to
    tensor by tensor, and sealed (header rewritten in place) once the
    next append would exceed the target size. All mutations are reflected
    in the converter state and flushed to disk before the next mutation,
    so a kill at any point leaves a resumable on-disk state.
    """

    PREFIXES = {"main": MAIN_PREFIX, "mtp": MTP_PREFIX}

    def __init__(self, group, out_dir, state, save_state):
        self.group = group
        self.prefix = self.PREFIXES[group]
        self.out_dir = out_dir
        self.state = state          # state["groups"][group], mutated in place
        self.save_state = save_state

    # -- state helpers ----------------------------------------------------

    @property
    def gstate(self):
        return self.state["groups"][self.group]

    def _shard_name(self, idx):
        return f"{self.prefix}-{idx:05d}.safetensors"

    # -- write path --------------------------------------------------------

    def _create_open(self):
        idx = self.gstate["next_shard_idx"]
        name = self._shard_name(idx)
        path = os.path.join(self.out_dir, name)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", HEADER_RESERVE))
            f.write(_placeholder_header())
        self.gstate["open"] = {"file": name, "data_bytes": 0, "entries": []}
        self.save_state()

    def append_tensor(self, tensor, progress_cb=None):
        """Append one tensor's raw bytes to the open shard.

        F8_E4M3 payloads are scanned for the NaN codes 0x7F/0xFF as they
        stream past; a NaN aborts the conversion (the partial append is a
        torn write, cleaned up on the next run's validation)."""
        open_ = self.gstate["open"]
        if open_ is None:
            self._create_open()
            open_ = self.gstate["open"]
        path = os.path.join(self.out_dir, open_["file"])
        data_off = open_["data_bytes"]
        scan_nan = tensor.dtype == "F8_E4M3"
        with open(path, "r+b") as out, open(tensor.src_path, "rb") as src:
            out.seek(8 + HEADER_RESERVE + data_off)
            src.seek(tensor.file_offset)
            remaining = tensor.nbytes
            while remaining:
                chunk = src.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(
                        f"{tensor.src_path}: short read on {tensor.name}")
                if scan_nan and (FP8_NAN_CODES[0] in chunk
                                 or FP8_NAN_CODES[1] in chunk):
                    raise ValueError(
                        f"{tensor.src_name}: E4M3 NaN code (0x7F/0xFF) in "
                        f"{tensor.src_path}; a trained glm5_next "
                        f"checkpoint never contains NaN weights — the "
                        f"source is corrupt or not the official "
                        f"checkpoint, refusing to convert")
                out.write(chunk)
                remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        open_["entries"].append({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "offset": data_off,   # relative to data region start
            "nbytes": tensor.nbytes,
        })
        open_["data_bytes"] += tensor.nbytes
        self.save_state()
        if progress_cb:
            progress_cb("tensor", group=self.group, name=tensor.name,
                        shard=open_["file"])

    def ensure_capacity(self, nbytes, target_bytes, progress_cb=None):
        """Seal the open shard if appending nbytes would exceed the target.

        Called before a whole expert slab (or a single dense tensor), so
        a slab is never split across output shards. Callers must NOT call
        this for the remainder of a partially-written slab (a seal there
        would straddle the slab; overshooting the target is preferred).
        """
        open_ = self.gstate["open"]
        if open_ and open_["data_bytes"] > 0 and \
                open_["data_bytes"] + nbytes > target_bytes:
            self.seal(progress_cb)

    def seal(self, progress_cb=None):
        """Finalize the open shard: rewrite its header region in place."""
        open_ = self.gstate["open"]
        if open_ is None:
            return
        path = os.path.join(self.out_dir, open_["file"])
        header = _sealed_header_bytes(open_["entries"])
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("<Q", HEADER_RESERVE))
            f.write(header)
            f.flush()
            os.fsync(f.fileno())
        size = os.path.getsize(path)
        self.gstate["sealed"][open_["file"]] = {
            "size": size,
            "header_sha256": _header_sha256(path),
            "ntensors": len(open_["entries"]),
        }
        sealed_name = open_["file"]
        self.gstate["open"] = None
        self.gstate["next_shard_idx"] += 1
        self.save_state()
        if progress_cb:
            progress_cb("seal", group=self.group, shard=sealed_name,
                        ntensors=self.gstate["sealed"][sealed_name]["ntensors"])

    # -- resume validation --------------------------------------------------

    def validate(self):
        """Verify-before-trust check of this group's on-disk output.

        Sealed shards must match recorded size + header hash. The open
        shard must contain at least the committed bytes; a torn tail
        (crash mid tensor write, after the previous state flush) is
        truncated. A shard whose seal was written but never recorded in
        the state is adopted as sealed. Anything else is corruption: fail
        loudly.
        """
        for name, rec in self.gstate["sealed"].items():
            path = os.path.join(self.out_dir, name)
            if not os.path.exists(path):
                raise ValueError(f"missing sealed output shard {name}")
            if os.path.getsize(path) != rec["size"]:
                raise ValueError(f"sealed shard {name}: size mismatch "
                                 f"(state {rec['size']}, disk "
                                 f"{os.path.getsize(path)})")
            if _header_sha256(path) != rec["header_sha256"]:
                raise ValueError(
                    f"sealed shard {name}: header hash mismatch")

        open_ = self.gstate["open"]
        if open_ is None:
            return
        path = os.path.join(self.out_dir, open_["file"])
        if not os.path.exists(path):
            raise ValueError(f"missing open output shard {open_['file']}")
        data_start = 8 + HEADER_RESERVE
        expected = data_start + open_["data_bytes"]
        size = os.path.getsize(path)
        if size < expected:
            raise ValueError(
                f"open shard {open_['file']}: {size} bytes on disk but "
                f"state records {expected}; output is corrupt, delete "
                f"the apus-* shards and state file and reconvert"
            )
        if size > expected:
            # Torn write of the tensor that was being appended when we
            # were killed: the state flush for it never happened. Drop
            # the tail.
            with open(path, "r+b") as f:
                f.truncate(expected)

        # Distinguish "placeholder header" from "sealed but unrecorded".
        with open(path, "rb") as f:
            f.read(8)
            raw = f.read(HEADER_RESERVE)
        header = json.loads(raw)
        if "__metadata__" not in header:
            expected_header = _sealed_header_bytes(open_["entries"])
            if raw != expected_header:
                raise ValueError(
                    f"open shard {open_['file']}: header neither "
                    f"placeholder nor the expected sealed header; "
                    f"refusing to trust it"
                )
            # Seal completed but the state flush did not: adopt it.
            self.gstate["sealed"][open_["file"]] = {
                "size": expected,
                "header_sha256": hashlib.sha256(
                    struct.pack("<Q", HEADER_RESERVE) + raw).hexdigest(),
                "ntensors": len(open_["entries"]),
            }
            self.gstate["open"] = None
            self.gstate["next_shard_idx"] += 1
            self.save_state()


# --------------------------------------------------------------------------
# Converter state
# --------------------------------------------------------------------------

def _empty_state(cfg_hash, target_bytes, n_main):
    return {
        "format_version": FORMAT_VERSION,
        "model_type": "glm5_next",
        "config_hash": cfg_hash,
        "target_shard_bytes": target_bytes,
        "header_reserve": HEADER_RESERVE,
        "n_main_layers": n_main,
        "inputs_done": [],
        "groups": {
            g: {"next_shard_idx": 1, "open": None, "sealed": {}}
            for g in ("main", "mtp")
        },
        # key -> {"members": {suffix: src_name}, "shards": [src shards]}
        "pending_slabs": {},
        # src names of resolved slab members not yet byte-verified by the
        # download driver (persisted so a crash between resolution and
        # verification cannot lose them).
        "unverified_resolved": [],
        "stripped": {"vision_tensors": 0, "vision_bytes": 0},
        "complete": False,
    }


class Converter:
    def __init__(self, src_dir, dst_dir, target_bytes=DEFAULT_TARGET_BYTES):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.target_bytes = target_bytes
        self.state_path = os.path.join(dst_dir, STATE_FILE)
        self.cfg_hash = config_hash(src_dir)
        self.n_main = load_n_main_layers(src_dir)
        os.makedirs(dst_dir, exist_ok=True)
        self.state = self._load_or_init()
        # Global weight_map (original names -> shard) for split-slab
        # assembly; None in flat-dir mode (splits then fail loudly).
        self.weight_map = None
        index_path = os.path.join(src_dir, INDEX_NAME)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        self.streams = {
            g: GroupStream(g, dst_dir, self.state, self.save_state)
            for g in ("main", "mtp")
        }
        for stream in self.streams.values():
            stream.validate()
        # Names already written to the output (sealed shards + committed
        # open-shard entries). Conversion of an interrupted input shard
        # resumes exactly after these, never redoing them.
        self.written = set()
        for stream in self.streams.values():
            for shard in stream.gstate["sealed"]:
                self.written.update(read_st_header(
                    os.path.join(dst_dir, shard)))
            open_ = stream.gstate["open"]
            if open_ is not None:
                self.written.update(e["name"] for e in open_["entries"])

    def _load_or_init(self):
        if not os.path.exists(self.state_path):
            existing = [
                n for n in os.listdir(self.dst_dir)
                if n.endswith(".safetensors") and n.startswith("apus")
            ]
            if existing:
                raise ValueError(
                    f"{self.dst_dir} contains apus shards but no state "
                    f"file; refusing to guess — remove them or restore "
                    f"{STATE_FILE}"
                )
            return _empty_state(self.cfg_hash, self.target_bytes,
                                self.n_main)
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state["format_version"] != FORMAT_VERSION:
            raise ValueError("state format version mismatch")
        if state["config_hash"] != self.cfg_hash:
            raise ValueError(
                "config hash mismatch: the source directory changed "
                "since conversion started; refusing to mix outputs"
            )
        if state["target_shard_bytes"] != self.target_bytes:
            raise ValueError(
                f"target shard size changed "
                f"({state['target_shard_bytes']} -> {self.target_bytes}); "
                f"keep it constant across a run"
            )
        return state

    def save_state(self):
        """Atomic state flush: every crash window collapses to the last
        fully recorded step."""
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".state-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def referenced_shards(self):
        """Source shards still needed to resolve pending slabs; the
        download driver must not delete these."""
        out = set()
        for rec in self.state["pending_slabs"].values():
            out.update(rec["shards"])
        return out

    # -- main flow ---------------------------------------------------------

    def convert(self, shard_names=None, progress_cb=None):
        """Convert input shards (all pending, or the named ones)."""
        pending = [
            n for n in list_input_shards(self.src_dir)
            if n not in self.state["inputs_done"]
        ]
        if shard_names is not None:
            wanted = set(shard_names)
            missing = wanted - set(list_input_shards(self.src_dir))
            if missing:
                raise ValueError(f"unknown input shard(s): {sorted(missing)}")
            pending = [n for n in pending if n in wanted]
        for name in pending:
            resolved = self._convert_input_shard(name, progress_cb)
            self.state["inputs_done"].append(name)
            self.state["unverified_resolved"].extend(resolved)
            self.save_state()
            if progress_cb:
                progress_cb("input_done", shard=name)

    def mark_resolved_verified(self, src_names):
        """The download driver byte-verified these resolved slab members;
        drop them from the persisted unverified set."""
        drop = set(src_names)
        self.state["unverified_resolved"] = [
            n for n in self.state["unverified_resolved"] if n not in drop
        ]
        self.save_state()

    def _rename_and_validate(self, raw_tensors, shard_name):
        """Split vision from kept tensors; rename, whitelist, and
        pair-validate the kept set. Returns (kept, n_vision, vision_bytes)
        where kept maps stripped name -> SrcTensor."""
        kept = {}
        n_vision = 0
        vision_bytes = 0
        for src_name, t in raw_tensors.items():
            if is_vision(src_name):
                n_vision += 1
                vision_bytes += t.nbytes
                continue
            out = output_name(src_name)
            validate_structure(out)
            kept[out] = SrcTensor(out, t.dtype, t.shape, t.src_path,
                                  t.file_offset, t.nbytes,
                                  src_name=src_name)
        validate_fp8_pairs(kept)
        validate_mtp_rules(kept, self.n_main)
        return kept, n_vision, vision_bytes

    def _convert_input_shard(self, shard_name, progress_cb):
        raw = read_st_header(os.path.join(self.src_dir, shard_name))
        tensors, n_vis, vis_bytes = self._rename_and_validate(
            raw, shard_name)

        ordered, deferred = plan_contribution(tensors)
        for key, members, missing in deferred:
            self._defer_slab(key, members, missing, tensors, shard_name)
        if deferred:
            self.save_state()

        by_group = {"main": [], "mtp": []}
        for name in ordered:
            by_group[classify_group(name, self.n_main)].append(name)
        for group, names in by_group.items():
            self._write_ordered(self.streams[group], names, tensors,
                                progress_cb)

        # A later shard may complete slabs deferred earlier (the real
        # checkpoint's split expert resolves one shard after it starts).
        resolved = self._resolve_pending(shard_name, progress_cb)

        self.state["stripped"]["vision_tensors"] += n_vis
        self.state["stripped"]["vision_bytes"] += vis_bytes
        return resolved

    def _write_ordered(self, stream, names, tensors, progress_cb):
        todo = [n for n in names if n not in self.written]
        i = 0
        while i < len(todo):
            key = slab_key(todo[i])
            if key is not None:
                # Whole slab: the capacity check covers all remaining
                # members so the slab can never straddle a shard
                # boundary. After an interruption only the unwritten
                # members remain here; they still land contiguously at
                # the open shard's tail (and must NOT trigger a seal —
                # overshooting the target beats straddling the slab).
                run = []
                while i + len(run) < len(todo) and \
                        slab_key(todo[i + len(run)]) == key:
                    run.append(todo[i + len(run)])
                partial = any(
                    slab_key(n) == key for n in self.written) and \
                    len(run) < len(SLAB_MEMBERS)
                if not partial:
                    total = sum(tensors[s].nbytes for s in run)
                    stream.ensure_capacity(total, self.target_bytes,
                                           progress_cb)
                for s in run:
                    stream.append_tensor(tensors[s], progress_cb)
                i += len(run)
            else:
                t = tensors[todo[i]]
                stream.ensure_capacity(t.nbytes, self.target_bytes,
                                       progress_cb)
                stream.append_tensor(t, progress_cb)
                i += 1
        self.written.update(todo)

    # -- split-expert slabs --------------------------------------------------

    def _defer_slab(self, key, members, missing, tensors, shard_name):
        """Record a slab whose members span input shards. All 6 members
        must exist in the global weight_map, else the input is corrupt."""
        layer, expert = key.split(".")[1:]
        src_names = {}
        shards = set()
        for suffix in SLAB_MEMBERS:
            if suffix in members:
                src_names[suffix] = tensors[members[suffix]].src_name
                shards.add(shard_name)
            else:
                src = (f"{LANG_PREFIX}layers.{layer}.mlp.experts.{expert}."
                       f"{suffix}")
                if self.weight_map is None or src not in self.weight_map:
                    raise ValueError(
                        f"expert {key}: member {suffix} missing from "
                        f"{shard_name} and from the checkpoint index; "
                        f"corrupt or unexpected input")
                src_names[suffix] = src
                shards.add(self.weight_map[src])
        if key in self.state["pending_slabs"]:
            # Re-deferral after a crash before inputs_done was recorded:
            # the record is a pure function of the input, so it must
            # agree with what is already there.
            existing = self.state["pending_slabs"][key]
            if existing["members"] != src_names:
                raise ValueError(f"pending slab {key}: inconsistent "
                                 f"re-deferral")
            return
        self.state["pending_slabs"][key] = {
            "members": src_names,
            "shards": sorted(shards),
        }

    def _resolve_pending(self, current_shard, progress_cb):
        """Append pending slabs whose member shards have all been
        converted. Returns the src names of resolved members (for the
        download driver's byte-verification)."""
        done = set(self.state["inputs_done"]) | {current_shard}
        resolved = []
        for key in sorted(list(self.state["pending_slabs"]),
                          key=slab_sort_key):
            rec = self.state["pending_slabs"][key]
            if not set(rec["shards"]) <= done:
                continue
            # Rebuild SrcTensor records from the source shards (kept on
            # disk until the slab resolves — the download driver honors
            # referenced_shards()).
            members = {}
            for suffix, src_name in rec["members"].items():
                shard = self.weight_map[src_name]
                hdr = read_st_header(os.path.join(self.src_dir, shard))
                t = hdr[src_name]
                out = output_name(src_name)
                members[suffix] = SrcTensor(
                    out, t.dtype, t.shape, t.src_path, t.file_offset,
                    t.nbytes, src_name=src_name)
            group = "mtp" if slab_key_layer(key) >= self.n_main else "main"
            stream = self.streams[group]
            remaining = [s for s in SLAB_MEMBERS
                         if members[s].name not in self.written]
            if remaining:
                if len(remaining) == len(SLAB_MEMBERS):
                    total = sum(members[s].nbytes for s in remaining)
                    stream.ensure_capacity(total, self.target_bytes,
                                           progress_cb)
                # else: partially written before a crash — the rest must
                # land contiguously; never seal mid-slab.
                for s in remaining:
                    stream.append_tensor(members[s], progress_cb)
                self.written.update(members[s].name for s in remaining)
            # Verify all 6 members (also those written before a crash).
            resolved.extend(members[s].src_name for s in SLAB_MEMBERS)
            del self.state["pending_slabs"][key]
            self.save_state()
        return resolved

    def finalize(self, progress_cb=None):
        """Seal any open shards and (re)write the manifest. Idempotent."""
        if self.state["pending_slabs"]:
            raise ValueError(
                f"cannot finalize: {len(self.state['pending_slabs'])} "
                f"expert slab(s) still pending "
                f"({sorted(self.state['pending_slabs'])[:4]}...); convert "
                f"the input shards holding their remaining members first"
            )
        for stream in self.streams.values():
            stream.seal(progress_cb)
        manifest = self.build_manifest()
        path = os.path.join(self.dst_dir, MANIFEST_FILE)
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".manifest-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Standard safetensors index for the C loader and the M0 oracle
        # (both read model.safetensors.index.json, not the apus manifest).
        std_index = {
            "metadata": {"total_size": sum(t["nbytes"] for t in
                                           manifest["tensor_map"].values())},
            "weight_map": {name: t["shard"]
                           for name, t in manifest["tensor_map"].items()},
        }
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".stindex-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(std_index, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.dst_dir, INDEX_NAME))
        self.state["complete"] = True
        self.save_state()
        return manifest

    # -- introspection ------------------------------------------------------

    def output_tensor_map(self):
        """name -> (shard_file, absolute_offset, nbytes, dtype, shape) for
        everything written so far, from sealed headers + open entries."""
        out = {}
        for group, stream in self.streams.items():
            gstate = stream.gstate
            for shard in gstate["sealed"]:
                path = os.path.join(self.dst_dir, shard)
                for name, t in read_st_header(path).items():
                    out[name] = (shard, t.file_offset, t.nbytes, t.dtype,
                                 t.shape)
            open_ = gstate["open"]
            if open_ is not None:
                base = 8 + HEADER_RESERVE
                for e in open_["entries"]:
                    out[e["name"]] = (open_["file"], base + e["offset"],
                                      e["nbytes"], e["dtype"], e["shape"])
        return out

    def build_manifest(self):
        tensor_map = {}
        slabs = []
        groups = {}
        for group, stream in self.streams.items():
            gstate = stream.gstate
            files = sorted(gstate["sealed"])
            groups[group] = files
            expert_parts = {}
            for shard in files:
                for name, t in read_st_header(
                        os.path.join(self.dst_dir, shard)).items():
                    tensor_map[name] = {
                        "shard": shard,
                        "offset": t.file_offset,   # absolute file offset
                        "nbytes": t.nbytes,
                        "dtype": t.dtype,
                        "shape": t.shape,
                    }
                    key = slab_key(name)
                    if key is not None:
                        expert_parts.setdefault(key, []).append(
                            (t.file_offset, t.nbytes, shard))
            for key, parts in sorted(expert_parts.items(),
                                     key=lambda kv: slab_sort_key(kv[0])):
                if len(parts) != len(SLAB_MEMBERS):
                    raise ValueError(
                        f"expert {key}: {len(parts)} tensors in output, "
                        f"expected {len(SLAB_MEMBERS)}"
                    )
                parts.sort()
                shards = {p[2] for p in parts}
                if len(shards) != 1:
                    raise ValueError(
                        f"expert {key} straddles output shards {shards}")
                start = parts[0][0]
                contiguous = all(
                    parts[k][0] + parts[k][1] == parts[k + 1][0]
                    for k in range(len(parts) - 1)
                )
                if not contiguous:
                    raise ValueError(
                        f"expert {key} tensors not contiguous")
                _, layer, expert = key.split(".")
                slabs.append({
                    "block": f"layers.{layer}",
                    "expert": int(expert),
                    "shard": parts[0][2],
                    "offset": start,
                    "nbytes": sum(p[1] for p in parts),
                })
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "glm5_next",
            "config_hash": self.cfg_hash,
            "offset_base": "file",
            "header_reserve": HEADER_RESERVE,
            "n_main_layers": self.n_main,
            "expert_slab_members": list(SLAB_MEMBERS),
            "shard_groups": groups,
            "ntensors": len(tensor_map),
            "tensor_map": tensor_map,
            "expert_slabs": slabs,
            "stripped": dict(self.state["stripped"]),
        }


# --------------------------------------------------------------------------
# Verification: byte-compare source tensors against the output
# --------------------------------------------------------------------------

def verify_source(src_dir, dst_dir, shard_names=None, include_names=None,
                  log=print):
    """Byte-compare kept tensors of the given (or all converted) source
    shards against their copies in the output. Vision tensors must be
    ABSENT from the output. Members of still-pending slabs are skipped
    (verified when the slab resolves). `include_names` restricts the
    check to specific source names (used by the download driver to verify
    freshly resolved slabs). Returns the number of tensors verified;
    raises on the first mismatch."""
    conv = Converter.__new__(Converter)   # lightweight: no validation writes
    conv.src_dir, conv.dst_dir = src_dir, dst_dir
    conv.state_path = os.path.join(dst_dir, STATE_FILE)
    with open(conv.state_path, "r", encoding="utf-8") as f:
        conv.state = json.load(f)
    conv.target_bytes = conv.state["target_shard_bytes"]
    conv.streams = {
        g: GroupStream(g, dst_dir, conv.state, lambda: None)
        for g in ("main", "mtp")
    }
    out_map = conv.output_tensor_map()
    pending_srcs = set()
    for rec in conv.state["pending_slabs"].values():
        pending_srcs.update(rec["members"].values())

    shards = list_input_shards(src_dir)
    if shard_names is not None:
        shards = [s for s in shards if s in set(shard_names)]
    done = set(conv.state["inputs_done"])
    nverified = 0
    for shard in shards:
        if shard not in done:
            continue
        src_tensors = read_st_header(os.path.join(src_dir, shard))
        for name, t in src_tensors.items():
            if include_names is not None and name not in include_names:
                continue
            if is_vision(name):
                continue
            if name in pending_srcs:
                continue
            oname = output_name(name)
            if oname not in out_map:
                raise ValueError(f"{oname}: missing from output")
            oshard, ooff, onbytes, odtype, oshape = out_map[oname]
            if onbytes != t.nbytes or odtype != t.dtype \
                    or oshape != t.shape:
                raise ValueError(f"{oname}: metadata mismatch vs output")
            with open(t.src_path, "rb") as fs, \
                    open(os.path.join(dst_dir, oshard), "rb") as fo:
                fs.seek(t.file_offset)
                fo.seek(ooff)
                remaining = t.nbytes
                while remaining:
                    a = fs.read(min(COPY_CHUNK, remaining))
                    b = fo.read(min(COPY_CHUNK, remaining))
                    if a != b:
                        raise ValueError(
                            f"{oname}: BYTE MISMATCH between source and "
                            f"output")
                    remaining -= len(a)
            nverified += 1
        log(f"verify: {shard}: tensors byte-identical")
    # Vision tensors must NOT appear in the output.
    for oname in out_map:
        if is_vision(oname):
            raise ValueError(f"{oname}: vision tensor leaked into output")
    return nverified


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("convert", "verify", "finalize"):
        sp = sub.add_parser(cmd)
        sp.add_argument("src_dir")
        sp.add_argument("dst_dir")
        sp.add_argument("--shard", action="append", default=None,
                        help="limit to these input shard(s); repeatable")
        if cmd == "convert":
            sp.add_argument("--target-bytes", type=int,
                            default=DEFAULT_TARGET_BYTES)
    args = p.parse_args(argv)

    if args.cmd == "convert":
        conv = Converter(args.src_dir, args.dst_dir, args.target_bytes)
        conv.convert(shard_names=args.shard)
        conv.finalize()
        m = conv.build_manifest()
        s = m["stripped"]
        print(f"convert: OK — {m['ntensors']} tensors, "
              f"{len(m['expert_slabs'])} expert slabs; stripped "
              f"{s['vision_tensors']} vision tensors "
              f"({s['vision_bytes']} B)")
    elif args.cmd == "finalize":
        conv = Converter(args.src_dir, args.dst_dir)
        conv.finalize()
    elif args.cmd == "verify":
        n = verify_source(args.src_dir, args.dst_dir, args.shard)
        print(f"verify: OK, {n} tensors byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
