# apus — GLM-5.3-Flash

[![CI](https://github.com/bricesommers/apus-glm5.3-flash/actions/workflows/ci.yml/badge.svg)](https://github.com/bricesommers/apus-glm5.3-flash/actions/workflows/ci.yml)

Local inference engine for **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**
(320B-total / 18B-active hybrid-attention MoE, `glm5_next`, text-only,
official FP8 checkpoint) on consumer hardware. apus runs the full model —
~306 GiB of weights, 1M-token context — on a Mac with 32 GB of unified
memory by streaming routed experts from NVMe through a bounded RAM cache,
instead of holding the model in memory.

Measured on a MacBook Pro M1 Pro 32 GB (real container, 2 GiB expert
cache, `--tiered`):

- **Decode:** ~0.085 tok/s (~11–12 s/token) — the engine reads ~7.9 GiB
  of experts per generated token cold
- **Prefill:** ~76 s for a 26-token prompt
- **Model load:** ~7 s; peak footprint ~22.3 GiB
- **Weights:** ~306 GiB in the apus container (FP8-E4M3 blocks, F32 scales)
- **Context:** up to 1,048,576 tokens (per the model config)

This is a single-user, single-machine engine: correctness and exactness
first, throughput second. The wall at 32 GB is memory capacity, not disk
— a bigger-RAM machine helps steeply.

## Features

- **Exact quantized kernels** — FP8-E4M3 (128×128 blocks, F32
  `weight_scale_inv`) dequant→BF16 GEMV/GEMM kernels in hand-written ARM
  NEON and x86 AVX2 (scalar fallbacks everywhere), plus an optional Metal
  GPU backend for the dense compute (zero-copy unified memory, bitwise ==
  the CPU kernels). Numerics follow the HF reference dequant→BF16→matmul
  path exactly; insufficient RAM costs speed, never quality.
- **Hybrid attention** — 34 KDA linear-attention layers (chunked prefill
  + recurrent decode, fused causal conv, FP32 delta-rule state) and 11
  DSA (MLA pure-NoPE) layers with the Lightning Indexer (top-2048 sparse
  attention over a 1M context), implemented per the HF `glm5_next`
  reference.
- **mHC residual stream** — Manifold-Constrained Hyper-Connections
  (4-stream, Sinkhorn-20) on all 45 layers.
- **Tiered expert store** — the 288 routed experts per MoE layer live on
  NVMe in coalesced slabs and are demand-paged through a bounded RAM
  cache with an RSS guard; a router-lookahead prefetch pilot warms the
  cache one layer ahead. Prefetching changes only *when* an expert is in
  RAM, never the numerics (verified bitwise).
- **MTP speculative decoding** — the checkpoint's classic NextN/MTP layer
  drafts tokens, verified against the main model's own picks; the emitted
  stream is bitwise identical to non-speculative decoding. **On by
  default** (draft depth 3) in run, serve, and chat; `--no-spec` opts out
  per workload (at low draft acceptance spec can be slower).
- **OpenAI-compatible server** (`tools/server.py`) and a terminal chat
  client (`tools/chat.py`), with GLM specifics handled (`clear_thinking`,
  `reasoning_effort`, `<tool_call>` parsing, reasoning as
  `reasoning_content`).
- **Byte-exact tokenizer and chat encoding** — validated against the
  checkpoint's `tokenizer.json` (exhaustive 1.49M-probe gate) and the
  jinja2-rendered chat template (byte-exact goldens).
- **Portable** — builds and passes the full test battery on macOS/ARM
  (clang, NEON, optional Metal), Linux/x86_64 (gcc, AVX2 kernels with
  scalar fallbacks; libc + pthreads only, no BLAS dependency), and
  Windows/x86_64 (MinGW-w64 gcc via MSYS2 UCRT64; the POSIX surface is
  shimmed in `c/compat.h`).

## Requirements

- **macOS/Apple Silicon** (M1 or later) with Xcode command line tools, or
  **Linux/x86_64** with gcc and make (AVX2 auto-detected; scalar fallback
  otherwise), or **Windows 10/11 x86_64** with
  [MSYS2](https://www.msys2.org/) — from the **UCRT64** shell:
  `pacman -S mingw-w64-ucrt-x86_64-gcc make`, then the same build/run
  commands as Linux. (WSL2 also works: follow the Linux path inside it.)
- **≥ 32 GB unified memory recommended** (the tiered store trades speed
  for memory headroom; more RAM helps steeply)
- **~315 GB free disk** (the ~306 GiB weights container plus headroom for
  one ~5 GiB source shard during download/convert; the driver fetches
  shards one at a time so peak usage stays near the container size). The
  container may live on an external drive via a symlink — see
  `docs/USAGE.md`.
- Python 3.11+ for the tools (download/convert, chat, server)

## Quickstart (step by step, no experience needed)

You need: a Mac with Apple Silicon (M1 or later), a Linux PC, **or** a
Windows PC — and about **315 GB of free disk space** for the model.
Windows users: install [MSYS2](https://www.msys2.org/) first, then use
the **Windows** command blocks below (WSL2 works too — that follows the
Linux path).

Every grey box below is a command to paste into your terminal (on Mac:
open **Terminal** from Applications → Utilities; on Windows: open
**MSYS2 UCRT64** from the Start menu). Paste one box at a time, press
Enter, wait for it to finish.

### Step 1 — get the code onto your computer

Either clone it (if you have git):

```sh
git clone https://github.com/bricesommers/apus-glm5.3-flash.git
cd apus-glm5.3-flash
```

Or without git: click the green **Code** button on the GitHub page →
**Download ZIP** → double-click the downloaded zip → open a terminal
inside the unzipped folder (on Mac: right-click the folder → Services →
New Terminal at Folder). Then:

```sh
cd apus-glm5.3-flash-main    # only if you used the ZIP (folder name may vary)
```

From now on, run everything from inside this folder.

### Step 2 — install the tools

On **Mac** (installs the compiler if asked — say yes):

```sh
xcode-select --install 2>/dev/null; python3 -m venv .venv
.venv/bin/pip install numpy safetensors tokenizers huggingface_hub jinja2
```

On **Linux / WSL2**:

```sh
sudo apt update && sudo apt install -y build-essential python3-venv python3-pip
python3 -m venv .venv
.venv/bin/pip install numpy safetensors tokenizers huggingface_hub jinja2
```

On **Windows** (native): first get Python — open **PowerShell** and run
`winget install -e --id Python.Python.3.12` (or install it from
python.org and tick **"Add python.exe to PATH"**). Then in the **MSYS2
UCRT64** shell (Start menu → "MSYS2 UCRT64"):

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-gcc make
python -m venv .venv
.venv/Scripts/pip install numpy safetensors tokenizers huggingface_hub jinja2
```

This creates a small private Python environment called `.venv` inside the
folder — nothing is installed system-wide, and deleting the folder
removes everything.

> **Windows note:** a Windows venv puts its programs in `.venv/Scripts/`
> instead of `.venv/bin/`. In every command below that starts with
> `.venv/bin/`, use `.venv/Scripts/` instead (e.g.
> `.venv/Scripts/python tools/download.py ...`). Everything else —
> `make apus`, `./bin/apus run ...` — works as written in the UCRT64
> shell (it produces `bin/apus.exe` and runs it either way).

### Step 3 — download the model (the long part: ~306 GiB)

```sh
.venv/bin/python tools/download.py \
    --work weights/work --out weights/glm-5.3-flash
```

This downloads the model **from Z.AI's official Hugging Face repo
(`zai-org/GLM-5.3-Flash`)** — lab-original weights only, no community
quants — and repacks it byte-identically for streaming (the converter
repacks, it never requantizes). Expect **many hours depending on your
internet**. It is safe to interrupt (Ctrl-C, closing the lid, losing
wifi): run the same command again and it resumes exactly where it
stopped. When it prints `download+convert complete`, the model is in
`weights/glm-5.3-flash/`, with the config and tokenizer files alongside
the shards — you do not need to copy or move anything.

### Step 4 — build the engine

```sh
make apus
```

Takes under a minute. You now have the engine at `bin/apus`. (On macOS,
`make metal=1 apus` additionally builds `bin/apus_metal` with the Metal
GPU backend.)

### Step 5 — talk to it

```sh
.venv/bin/python tools/chat.py --model weights/glm-5.3-flash --tiered
```

Wait ~7 seconds for it to load, then type a question at the `you>`
prompt. Answers appear slowly (about one token every 12 seconds on a
32 GB Mac — this is normal for a 320B model on a laptop). Type `/help`
for commands, `/quit` to exit.

Prefer a one-off question instead of a chat?

```sh
./bin/apus run --model weights/glm-5.3-flash --tiered \
    --prompt "The capital of France is" --max-tokens 100 --temp 0
```

Want an app-like UI (Open WebUI, your own scripts)? Run the server and
point any OpenAI-compatible client at `http://localhost:8080/v1`:

```sh
.venv/bin/python tools/server.py --model weights/glm-5.3-flash --tiered --port 8080
```

The server follows OpenAI semantics: a request that omits `max_tokens`
generates until EOS (capped at 65,536 tokens by default,
`APUS_GSERVE_MAX_TOKENS`). **Open WebUI users**: see the dedicated
section in `docs/USAGE.md` — Open WebUI injects ~7K tokens of built-in
tool definitions into every request unless you create the model entry
with Built-in Tools all off; at this model's prefill speed that turns
every message into hours of silent prefill. The one-time fix (and the
recommended background-task settings) is documented there.

More options (speed knobs, environment variables, troubleshooting):
`docs/USAGE.md`.

## How it works

GLM-5.3-Flash is a 45-layer hybrid: 34 KDA gated-delta linear-attention
layers handle most positions in O(1) state, while 11 DSA layers (MLA with
pure NoPE attention plus a Lightning Indexer that selects the top-2048
keys per query) carry precise long-range retrieval; an mHC
hyper-connection residual stream (4 streams, Sinkhorn-routed) ties the
layers together. Each MoE layer routes every token to 8 of 288 routed
experts plus one shared expert, but naively you still need all 306 GiB
resident. apus instead keeps the dense path (attention, shared expert,
norms, router — BF16) in memory and stores the routed experts as
coalesced FP8 slabs on NVMe. A bounded RAM cache pages in the experts
each token actually routes to, while the pilot — a small predictor
reading the router's own math one layer ahead — prefetches likely
experts so most demand loads hit warm RAM. All dequantization happens at
exactly the points the HF reference specifies (FP8 block × F32 scale →
BF16 matmul), so the streamed model is numerically the same model, just
slower when the cache is cold. On top, the checkpoint's MTP layer drafts
speculative tokens that are verified against the main model's own picks,
so speculative decoding never changes a single emitted token. Full
design: `docs/ARCHITECTURE.md`.

## Quality discipline

apus was built gate-first: every subsystem was verified against the
reference implementation before integration — the FP8/BF16 kernels
against a numpy oracle ported from the HF `glm5_next` modeling file
(bitwise on expf-matching hosts), the tokenizer against an exhaustive
1.49M-codepoint probe of the checkpoint's `tokenizer.json` (0
mismatches), the chat encoding against jinja2-rendered goldens
(byte-exact), and the full forward pass against the oracle run back from
the converted container. The whole engine was then verified **bitwise
against the HF reference on the real 306 GiB container**: prefill plus
184/184 teacher-forced decode digests identical, and the chat-template
greedy path byte-identical (evidence: `tests/smoke/README.md`). Every
memory/caching/prefetch/speculative configuration is required to produce
bitwise identical tokens — insufficient memory may cost speed, never
output quality.

## Testing

Every milestone shipped with a hard-gate suite under `tests/`; the full
battery is what CI runs. Fixture/golden directories are gitignored and
regenerated deterministically by the make targets. On macOS the suites
use the local venv python; elsewhere pass `PY=python3`.

| Target | Suite |
|---|---|
| `test-m2` | Tokenizer + chat encoding, byte-exact vs the checkpoint's reference files |
| `test-m3g` | FP8-block dequant + BF16 GEMV/GEMM kernel hard gate |
| `test-m4g` | mHC residual stream + MoE router/experts vs the oracle |
| `test-m4h` | KDA (both orderings) + DSA + Lightning Indexer vs the oracle |
| `test-m5g` | Full-model forward pass, end-to-end from the converted container |
| `test-m6g` | Expert-store tiering (NVMe slabs, bounded RAM cache) + prefetch pilot |
| `test-m7a` | OpenAI-compatible server end-to-end (scripted fixtures) |
| `test-m7b` | Metal GPU backend (macOS only) |
| `test-m8g` | MTP speculative decoding (spec == non-spec, bitwise) |
| `tests/m1` | Converter/downloader python suite: `python3 -m unittest discover -s tests/m1` |

On macOS/ARM, or in Docker for Linux/x86_64 (`tools/docker/test-linux.sh`
runs the whole portable battery in an ubuntu:24.04 container):

```sh
tools/docker/test-linux.sh                    # full portable battery
tools/docker/test-linux.sh test-m3g test-m5g  # selected targets
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs the full battery on every
push to `main`: a `linux` job (ubuntu-latest, gcc/x86_64), a `macos` job
(macos-latest, clang/NEON + the Metal suite), and a `windows` job
(windows-latest, MinGW-w64 gcc via MSYS2 UCRT64 — same battery minus the
sanitizer twins and Metal, which is macOS-only).

## Repository layout

- `c/` — the engine: C11 header-only core + `apus.c` CLI/server driver +
  optional Metal backend (`backend_gmetal.mm`) + x86 kernels (`x86.h`)
- `tools/` — download/convert driver, chat client, OpenAI server, numpy
  oracle, dockerized Linux test harness (`tools/docker/`)
- `tests/` — the milestone test battery (see Testing above)
- `reference/` — the checkpoint's config/tokenizer/chat-template files
  (MIT, `LICENSE.zai`) plus the verbatim HuggingFace `glm5_next` modeling
  files (Apache-2.0) that the oracle and gates verify against
- `docs/` — architecture and usage

## License and notices

apus is **source-available, not open-source**, under the
**PolyForm Noncommercial License 1.0.0** (see
`LICENCE.PolyForm-Noncommercial`): you may use, copy, modify, and
distribute the apus source code for **noncommercial purposes only**; any
commercial use requires a separate license from the authors.

Third-party components remain under their own licenses — see `NOTICE`:

- Design and adapted code from **[colibri](https://github.com/JustVugg/colibri)**
  (Apache License 2.0, text in `LICENSE.apache-2.0`)
- **Z.AI** model weights (downloaded separately by `tools/download.py`
  from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
  not included in this repository) under the MIT License
- **HuggingFace** `glm5_next` reference modeling files under
  `reference/inference/` (Apache License 2.0)

apus is an independent project and is **not affiliated with, endorsed by,
or sponsored by Z.AI, the colibri project, or HuggingFace**.
