# apus — terminal usage

GLM-5.3-Flash (text-only), fully local on your machine. Nothing leaves it.

> **Status (2026-09-04):** the GLM engine is verified **bitwise
> against the HF reference on the real 306 GiB container** (prefill,
> decode, chat-template path — the tests/smoke/README.md evidence)
> and servable end to end — CLI, OpenAI-compatible HTTP server, terminal
> chat, Metal backend. The recommended dev-machine config is
> `--tiered --metal` with `APUS_GEXPERT_CACHE_MB=2048` (the wall is memory
> capacity, not disk or GPU). Real numbers in "What to expect" below.

Commands below assume a checkout built per the README (macOS, Linux, or
Windows/MSYS2-UCRT64). On Windows the Python venv lives at
`.venv/Scripts/` — use `.venv/Scripts/python` wherever the commands say
`.venv/bin/python`.

## The model dir

`--model DIR` points at a converted M1 v2 container (what
`tools/convert.py` produces) plus three small files the converter does
not copy — place them next to the shards:

```
weights/glm-5.3-flash/
  apus.index.json            # the v2 manifest (engine auto-detects glm5_next)
  model.safetensors.index.json
  apus-*.safetensors
  config.json                # copy from the HF checkpoint
  tokenizer.json             # copy from the HF checkpoint
  generation_config.json     # optional; eos stop set (default {154820,
                             # 154827, 154829}, the reference set)
```

`bin/apus` requires a glm5_next container (`apus.index.json` with
`model_type: "glm5_next"`); anything else is a hard error. (apus was
originally a DeepSeek-V4 engine; this repository is GLM-only.)

## Interactive chat (like colibri's `coli chat`)

```bash
cd apus-glm-5.3-flash
.venv/bin/python tools/chat.py --model weights/glm-5.3-flash --tiered
```

MTP speculative decoding is ON by default (`--no-spec` opts out,
`--spec-k K` sets the draft depth, default 3) — the reply text is
bitwise identical either way, spec just gets there faster.

In-chat commands:

| command | effect |
|---|---|
| `/quit` | exit (Ctrl-D also works) |
| `/reset` | clear conversation history |
| `/system <text>` | set a system message |
| `/clear on\|off` | clear_thinking: drop past-turn reasoning from the prompt (default on — the GLM multi-turn convention) |
| `/effort low\|high\|max` | reasoning effort (everything but low/high renders Max) |
| `/temp 0` | greedy/deterministic (default 1.0) |
| `/max 2000` | max tokens per reply (default 2048) |
| `/raw <text>` | raw completion for this turn (no chat template) |
| `/help` | show all commands |

Note: the GLM template has no thinking on/off switch — generation always
starts `<think>`; reasoning is printed dimmed up to `</think>`.

## One-shot generation

```bash
cd apus-glm-5.3-flash
./bin/apus run --model weights/glm-5.3-flash --tiered \
    --prompt "Your prompt here" --max-tokens 100 --temp 0
```

`--temp 0` = greedy. Add `--seed 42` for reproducibility. `--ids 1,2,3`
feeds raw token ids (no tokenizer needed). `--quiet` prints just the
token ids. **Sampling defaults come from the container's
`generation_config.json`** (`temperature`/`top_p` — the sealed GLM
container declares 1.0/0.95): they apply in `run` and `serve` whenever
the user/request does not set the parameter explicitly; `--temp` /
`--top-p` / `--greedy` (run) and the request fields or chat's `/temp` /
`/top_p` (serve) override them. A container without
`generation_config.json` keeps the old 1.0/1.0 defaults.
`--spec [--spec-k K]` (M8b; also `APUS_SPEC`/`APUS_SPEC_K`,
default K=3) runs MTP (NextN) speculative decoding — the emitted tokens
are bitwise identical to non-spec decoding at the same seed (the
tests/m8g gate); it requires the container's MTP block (`layers.45.*`)
and explicit `--spec` fails loudly without it. **DEFAULT ON in `run`
(M8c) and `serve`/`tools/chat.py` (M8d)** — `--no-spec` opts out; the
default-on form falls back to non-spec with a note on MTP-less
containers.

**Recommended config on a 32 GB dev machine** (measured):
`--tiered --metal` (use `bin/apus_metal`) with
`APUS_GEXPERT_CACHE_MB=2048`. Larger expert caches buy almost nothing at
32 GB (the pressure guard trims them to the working set; the win is
avoiding compressor churn). On hosts with 48+ GB RAM, raise
`APUS_GEXPERT_CACHE_MB` (16 GiB ≈ 8 payload slots/layer) and consider
`APUS_GMETAL_MIN_KB=32768` (the fixture-tuned bf16 offload split).

## OpenAI-compatible server (for other apps)

```bash
cd apus-glm-5.3-flash
.venv/bin/python tools/server.py --model weights/glm-5.3-flash --tiered --port 8080
```

Then point any OpenAI client at `http://localhost:8080/v1` (any non-empty
API key; set `APUS_API_KEY` to enforce one). Quick check:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "apus", "stream": true,
  "messages": [{"role": "user", "content": "Hello"}]}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
(SSE with `"stream": true`), `POST /v1/completions`, plus the
non-standard `POST /debug/encode` (rendered prompt + ids — a test
endpoint). Request schema GLM specifics (see tests/m7a/README.md):

- `tools` is a **sibling of `messages`** (a top-level template
  parameter), never attached to a message.
- No `"thinking"` field — GLM uses `"clear_thinking": bool` and
  `"reasoning_effort": "low"|"high"` (anything else renders Max), either
  top-level or inside `chat_template_kwargs`.
- Responses expose reasoning as `reasoning_content`; GLM `<tool_call>`
  output is parsed into OpenAI `tool_calls`
  (`finish_reason: "tool_calls"`).
- Message/encoding problems (`glm_encode_*` failures) come back as HTTP
  400 `invalid_request_error`.

### Open WebUI (browser chat)

```bash
# one-time
docker pull ghcr.io/open-webui/open-webui:main
docker run -d --name open-webui -p 3000:8080 \
  -e ENABLE_OLLAMA_API=false \
  -e OPENAI_API_BASE_URLS='http://host.docker.internal:8000/v1' \
  -e OPENAI_API_KEYS='apus' \
  -v open-webui:/app/backend/data --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
# later: docker start open-webui   (stop: docker stop open-webui)
```

(Adjust `OPENAI_API_BASE_URLS` to the gateway's host/port — 8000 above;
use `host.docker.internal` so the container reaches the host's
localhost.)

Then open http://localhost:3000 (first signup = local admin) and select
`glm-5.3-flash`. **Required one-time fix**: Open WebUI injects ~7K
tokens of built-in tool definitions into every request for models
without a model entry (`meta.builtinTools` defaults all-on). At this
model's prefill speed (~0.3 tok/s) that turns every message into HOURS
of silent prefill — the UI looks dead while the engine churns. Create
the model entry once (Admin Panel → Models → glm-5.3-flash → Edit →
Built-in Tools → all off), or in the DB:

```bash
docker exec open-webui python3 -c "
import sqlite3, json, time
db = sqlite3.connect('/app/backend/data/webui.db')
cats = ['automations','calendar','channels','chats','code_interpreter',
        'files','image_generation','knowledge','memory','notes',
        'notifications','subagents','tasks','time','user_input','web_search']
meta = {'builtinTools': {c: False for c in cats}}
now = int(time.time())
db.execute('insert or replace into model (id, user_id, name, params, meta, updated_at, created_at, is_active) values (?,?,?,?,?,?,?,1)',
           ('glm-5.3-flash', db.execute(\"select id from user where role='admin'\").fetchone()[0],
            'glm-5.3-flash', '{}', json.dumps(meta), now, now))
db.commit()"
docker restart open-webui
```

Also recommended: Admin Panel → Settings → Interface → off: Title
Auto-Generation, Tags, Follow-Up Generation, Autocomplete (each fires an
extra engine request that queues on the single engine — at GLM decode
speed these background calls dominate).

## What to expect

- Base-engine reference (160 GB model, 32 GB M1 Pro): ~10 s model load,
  ~0.8 tok/s prefill, ~0.3 tok/s decode. GLM-5.3-Flash streams ~7.9 GiB
  of experts per generated token cold (vs ~3.45 GB for the base model).
  Measured on the real container (26-token prompt, 64 greedy tokens,
  `--tiered`, 2 GiB expert cache, 32 GB M1 Pro):
  ~7 s model load, prefill ~76 s, decode ~11–12 s/token (~0.085 tok/s),
  peak footprint ~22.3 GiB. The wall is memory capacity, not disk — a
  bigger-RAM machine helps steeply.
- Peak RAM will be close to the 32 GB budget — close heavy apps for best
  speed.
- Multi-turn chat re-reads the conversation each turn (no KV reuse yet);
  use `/reset` for new topics.
- Stop the server/chat with Ctrl-C.

## Useful knobs (environment variables)

| var | default | effect |
|---|---|---|
| `APUS_GEXPERT_CACHE_MB` | 6144 | GLM expert RAM cache (48 MiB payloads); raise with free RAM |
| `APUS_GPILOT` | 1 | GLM router-lookahead prefetch pilot (tiered only) |
| `APUS_GPILOT_K` / `APUS_GPILOT_PREFILL_K` | 12 | pilot decode/prefill depth |
| `APUS_RSS_GUARD_MB` | 26624 | memory-pressure ceiling (phys_footprint on macOS); coldest payloads drop at layer boundaries |
| `APUS_GSPEC_QUEUE` | 2×I/O threads | GLM speculative-prefetch backlog cap; hints beyond it (or while demand loads are queued/in flight, or near the pressure ceiling) are dropped |
| `APUS_GPREFILL_MOE_CHUNK` | 8 | GLM tiered prefill MoE token-chunk size (bounds the live expert union = prefill peak RAM; 0 = unchunked). Bitwise-neutral at any setting |
| `APUS_THREADS` | P-cores | compute threads |
| `APUS_API_KEY` | unset | require `Authorization: Bearer` on the gateway |
| `APUS_METAL=1` | off | GPU dense offload (macOS, `bin/apus_metal`; GLM: BF16 GEMV/GEMM + fused FP8-block GEMM, bitwise == CPU). P4 real-scale: with persistent weight wraps + the default floors, metal ≈ CPU or a small win (fused FP8 DSA linears); the bf16 GEMV offload defaults OFF (measured net loss under memory pressure — set `APUS_GMETAL_MIN_KB=32768` on bigger-RAM hosts) |
| `APUS_GMETAL_MIN_KB` | off (∞) | GLM Metal: BF16 weights below this stay on the CPU (0 = offload all) |
| `APUS_GMETAL_FP8_MIN_KB` | 0 | GLM Metal: floor for the fused FP8-block path (the real-scale win) |
| `APUS_GMETAL_MAX_M` | 1 | GLM Metal: activation rows above this stay on the CPU (prefill GEMMs; the shaders have no m-blocking) |
| `APUS_DECODE_TIMING=1` | off | per-token decode timing + cache counter deltas on stderr (diagnostics) |

## Note: weights on an external drive

The ~306 GiB container (`weights/glm-5.3-flash/`) may live on an external
drive to save internal disk space (it does not fit on the internal disk
next to the 305.8 GiB source shards — see docs/ARCHITECTURE.md §8). If so,
keep a symlink at `weights/glm-5.3-flash` pointing to it
(`ln -s /Volumes/<DRIVE>/glm-5.3-flash weights/glm-5.3-flash`) — every
command then works unchanged. Use a fast drive: the engine reads up to
~8 GiB of experts per generated token from it.
