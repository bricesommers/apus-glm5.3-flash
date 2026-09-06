# tests/m7a — OpenAI-compatible server, GLM-5.3-Flash (M7a)

Milestone M7a makes the GLM engine servable end to end, following the
colibri split — a **Python stdlib-only HTTP gateway** (`tools/server.py`)
driving the **C engine** (`bin/apus serve`) as a persistent subprocess.
No new dependencies (gateway: `http.server`, `json`, `subprocess`,
`threading`; engine: libc/pthreads only). No numerics/router changes: the
serve path composes the M5/M6 gated engine (`c/gmodel.h` +
`c/gcache.h`/`c/gpilot.h`), `c/encoding.h` (rendering), `c/tok.h`
(tokenizer) and `c/sample.h` (sampling) exactly as the `run` CLI does.

Run:

```
make golden-m7a     # regenerate tests/m7a/fixtures (scripted GLM parrots)
make test-m7a       # 42 tests, exit 0 iff all pass
make ubsan-m7a      # same suite with bin/apus built -fsanitize=undefined
```

Manual use:

```
bin/apus run   --model DIR --prompt "hi" [--tiered] [--greedy]     # CLI
bin/apus serve --model DIR [--tiered] [--spec | --no-spec]         # engine, stdio
.venv/bin/python tools/server.py --model DIR --port 8000 [--tiered] # gateway
.venv/bin/python tools/chat.py   --model DIR [--tiered] [--no-spec] # terminal chat
```

A GLM model dir is an M1 v2 container (apus.index.json with
`model_type: "glm5_next"` — that manifest is how `bin/apus` selects the
GLM engine over the retained DeepSeek-V4 one) plus:

- `config.json` — the model config (the converter does NOT copy it;
  place the HF checkpoint's `config.json` next to the shards; the flat
  oracle-fixture config also works — the fixtures use that),
- `tokenizer.json` — required for text/chat requests (ids-only without
  it), same story,
- `generation_config.json` — optional; `eos_token_id` (int or array) is
  the generation stop set. Default: the reference set
  `GLM_EOS_IDS {154820, 154827, 154829}` (c/encoding.h).

## Architecture: why stdio NDJSON (not a socket)

The C engine speaks **one-JSON-object-per-line (NDJSON) on stdin/stdout**;
the gateway owns all networking. Rationale: the engine stays libc-only (no
socket code in C to audit/fuzz), the process model matches colibri's
proven gateway-drives-engine pattern, pipes make the protocol trivially
testable without a TCP stack, and a single engine process serves exactly
one gateway so a second process is the only concurrency story. The
`--port P` option mentioned in the milestone sketch lives on the gateway
(`tools/server.py --port`), not the engine.

### Engine protocol (request → `bin/apus serve` stdin)

```json
{"id": <any>, "cmd": "encode",
 "messages": [...], "tools": [...]|null,
 "clear_thinking": true|false, "reasoning_effort": "low"|"high"|null}

{"id": <any>, "cmd": "generate",
 "messages": [...] | "text": "raw prompt" | "ids": [1,2,3],
 "tools": [...]|null, "clear_thinking": bool, "reasoning_effort": str|null,
 "max_tokens": int, "temperature": float, "top_p": float,
 "seed": uint, "stop": [str, ...]}
```

- `tools` is a **top-level template parameter** (a sibling of `messages`,
  the GLM template shape — never attached to a message).
- The GLM template has **no thinking on/off switch**: generation always
  starts `<think>`. `reasoning_effort` is "Low"/"High" only for exactly
  "low"/"high"; everything else (including unset) renders "Max".
  `clear_thinking` drops reasoning of assistant messages up to the last
  user message (default false).
- `"text"` is tokenized verbatim (no chat template) — the
  `/v1/completions` path. `"ids"` feeds raw ids (synthetic models without
  a tokenizer).
- Defaults: `max_tokens` unset = **until EOS** (OpenAI semantics, the
  base engine's serve default; bounded by `APUS_GSERVE_MAX_TOKENS`,
  default 65536 — the cap only sizes the per-request KV, ~0.8 GiB at the
  default), `temperature` 1.0, `top_p` 1.0, `seed` 0;
  `temperature <= 0` = greedy. The KV state is sized exactly to
  prompt + `max_tokens` (the M6 policy: no eviction, no quantization,
  loud failure past capacity). (Pre-M8e the default was a flat 32, which
  truncated GLM thinking mid-block for clients that omit max_tokens —
  e.g. Open WebUI — showing reasoning only with no answer.)

### Engine protocol (events → stdout)

```json
{"id","type":"encoded","text","ids"}                 encode reply
{"id","type":"prompt","prompt_tokens"}               generate, first
{"id","type":"token","token_id","text"}              per generated token
{"id","type":"done","finish_reason","prompt_tokens",
 "completion_tokens","text"}                         terminal
{"id","type":"error","code","message"}               request failed
```

- The terminating EOS token is never emitted as a `token` event;
  `finish_reason` is `"stop"` (EOS), `"length"` (max_tokens), or
  `"stop_string"`.
- `code` is `"bad_request"` for request-schema/encoding failures
  (the gateway maps these to HTTP 400 — the M2 handoff:
  `glm_encode_*` failures surface as 400s) or `"engine_error"` for
  internal failures (→ 500/503).
- Stop strings are matched against the assembled decoded text; on a match
  the text is truncated at the match start (a partial piece of the last
  token is emitted if it precedes the match) and generation ends
  `"stop_string"`. `text` fields require a tokenizer in the model dir.
- The process stays alive across requests (model loads once). Every
  request gets a **fresh KV state**; conversation state is the gateway's
  job and multi-turn context is re-prefilled. KV reuse across turns is a
  later optimization.

## Gateway endpoints (`tools/server.py`)

| Endpoint | Notes |
|---|---|
| `GET /health` | `{"status","model","engine"}`; no auth |
| `GET /v1/models` | single-model OpenAI list |
| `POST /v1/chat/completions` | full OpenAI shape; `stream:true` → SSE |
| `POST /v1/completions` | `prompt` (string only); stream supported |
| `POST /debug/encode` | NON-STANDARD test endpoint: rendered prompt text + token ids for a message list (conformance/usage verification) |

Honored chat fields: `messages`, `tools`, `temperature`, `top_p`,
`max_tokens`/`max_completion_tokens`, `seed`, `stream`,
`stream_options.include_usage`, `stop` (string or list), `model`,
`reasoning_effort` (top-level or in `chat_template_kwargs`),
`clear_thinking` (top-level or in `chat_template_kwargs`; default from
`--clear-thinking` / `APUS_CLEAR_THINKING`, off). Unknown fields are
ignored.

**Responses.** Reasoning (everything up to `</think>`) is exposed as
`reasoning_content` (DeepSeek API style); the rest is `content`, plus
OpenAI `tool_calls` when GLM `<tool_call>` blocks are present
(`finish_reason: "tool_calls"`).

**GLM ↔ OpenAI tool calls.** Request side: `tools` render into the prompt
as the GLM tools system block via `c/encoding.h` (byte-exact vs
`reference/chat_template.jinja`, gated here through `/debug/encode`).
Assistant `tool_calls` in REQUEST history normally carry
`function.arguments` as a JSON string on the OpenAI wire, but the GLM
template iterates the arguments object — the gateway therefore
**normalizes string arguments back to objects** when they parse
(`normalize_tool_call_arguments`), so clients can feed our own responses
back as history; unparseable strings pass through and fail at the engine
with a 400. Response side: the gateway parses
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`
blocks into OpenAI `tool_calls` (`id` = `call_<24 hex>`,
`function.arguments` = JSON string; a value that `json.loads` parses is
recovered to its typed form, otherwise kept a string — the wire format is
ambiguous for string arguments that look like JSON, documented heuristic).
Tolerant: EOS not required, unterminated thinking → all
`reasoning_content`, malformed tool-call markup → plain content.
`finish_reason`: `"length"` for length, `"tool_calls"` when tool calls
were parsed, else `"stop"`. Streaming limitation: deltas forward the raw
text (`<tool_call>` markup included) — only the final `finish_reason`
reflects the parse; the parsed structure is available non-streaming.

**SSE streaming.** Chunks are `chat.completion.chunk` with
`choices[0].delta`: first `{"role":"assistant"}`, then
`reasoning_content` deltas (buffered until `</think>` is unambiguous),
then `content` deltas, then a final chunk with empty delta and
`finish_reason`, an optional usage chunk
(`stream_options.include_usage`, `choices: []`), and `data: [DONE]`.
Served with `Connection: close` (no chunked encoding).

**Errors.** `400` malformed JSON / bad fields / encoding failures, `404`
unknown model or path, `401` auth failure — all in the OpenAI shape
`{"error": {"message","type","param","code"}}` (`invalid_request_error` /
`not_found_error` / `authentication_error` / `engine_error`).

**Concurrency.** One engine ⇒ requests are serialized through a single
lock in the gateway: concurrent HTTP requests queue in lock-acquisition
order and run one at a time. **Auth:** env `APUS_API_KEY` →
`Authorization: Bearer` required on `/v1/*` and `/debug/*` (off by
default; `/health` open).

## Tiered serving (the M6 cache + pilot)

`--tiered` (engine flag, or the gateway's `--tiered`) opens the container
through `apus_gmodel_open2` with the slab-streaming expert cache behind
`apus_gmodel_expert()`: budget `APUS_GEXPERT_CACHE_MB` (counted in 48 MiB
BF16 payloads at real scale), and the M6 router-lookahead pilot attaches
automatically (`APUS_GPILOT=0` disables; `APUS_GPILOT_K` /
`APUS_GPILOT_PREFILL_K` tune it). tests/m6g gates eager == tiered ==
tiered+pilot digests at the model level; this battery gates the same
equality END TO END (the token stream through `--tiered` with a
1 MB cache — evictions forced — is bitwise the oracle stream).

## Speculative serving (M8d: MTP --spec in serve mode)

`--spec` (DEFAULT ON, like run mode since M8c; `--no-spec` /
`APUS_SPEC=0` opts out, `--spec-k K` / `APUS_SPEC_K`, default 3) runs the
MTP draft/verify loop (c/gmtp.h ApusGspec) inside the generate handler.
The NDJSON protocol is **unchanged**: a verify batch's accepted burst is
emitted as ordinary per-token `token` events (several in a row). The
accept rule (a draft counts only if it equals the main model's own
`apus_sample` draw; one RNG uniform per emitted token in position order,
drafts consume none) makes the emitted stream **bitwise identical** to
`--no-spec` at the same seed — the ServeSpec suite below gates exactly
that through the protocol, greedy and seeded-sampled, eager and tiered.
The EOS stop set and stop strings are checked on every emitted token,
spec-accepted ones included. The KV state gets the same `spec_k + 1`
headroom as run mode. Resolution mirrors run_main: an explicit `--spec`
on an MTP-less container is a loud startup error; the default-on form
falls back to non-spec with a stderr note. `tools/chat.py` passes
`--spec` by default (`--no-spec` opts out; `--spec-k` overrides);
`tools/server.py` leaves the flag unset (default ON, silent fallback).

## The scripted GLM "parrot" fixtures (`tests/m7a/fixtures/`)

Random-weight models can't produce a *known* token stream, which the
server tests need (tool-call output, EOS, stop strings, exact usage
counts) — and the end-to-end oracle gate needs a stream the M0 oracle
reproduces from the same container on ANY host. `gen_fixtures.py` builds
two mini-containers on the m5g tiny config (L=5: KDA 0,1,2,4 + DSA 3;
dense MLP 0-2, MoE 3,4 — every layer kind exercised) with **all layer
weights zero** (every sublayer output vanishes; the hc×4 mHC residual is
preserved exactly: Sinkhorn of zeros is the uniform comb, post =
2·sigmoid(0) = 1), norms at 1.0, random embed rows, and a **scripted
head**: for each transition `a → b`, `head[b] = 8·embed[a]`, so decoding
follows a fixed Markov chain in the last token. The oracle loads the
converted container back and greedy-generates from `[<think>]`: the
stream must equal the script AND every step's top1-top2 logit margin
must exceed 100 (measured ≈ 1824) — that's what makes the token gate
host-exp independent. Chain tokens with multi-char content are added
tokens (ids 268+; each used at exactly one chain position), the GLM
specials are added tokens 256–267, and ids 0–255 are a byte-level vocab
(GPT-2 bytes_to_unicode alphabet, no merges — the Qwen2-style Split
pre-tokenizer in c/tok.h cannot change a merge-less byte sequence).

- `model_chat/`: reasoning `"reasoning: thinking it over."` +
  `</think>` + `"The answer is STOP right here."` + EOS.
- `model_tools/`: reasoning + a complete GLM `<tool_call>` get_weather
  block + EOS.
- `golden.json`: chains, the oracle streams, margins, specials — what
  the tests assert against.

## Test coverage (49 tests, `test_server.py`)

- **Pipe protocol** (11): encode text (jinja2 golden) + special-token id
  placement; reasoning_effort/clear_thinking render variants; malformed
  line / unknown cmd / error recovery; bad_request codes (schema +
  `glm_encode` failures, loop survives); generate event sequence, EOS
  non-emission, text reassembly; **the oracle token gate** (engine
  stream == oracle greedy stream from the same container); >64-token
  prompt (chunked-KDA continuation prefill); usage counts == `encode`
  ids; stop-string truncation; max_tokens clamp; **tiered stream ==
  eager stream** (1 MB cache + pilot).
- **Serve --spec (M8d)** (7, on the tests/m8g dsa_top container — the
  parrot has no MTP block; each leg copies the 36 MB container to a temp
  dir, goldens never mutated): **spec == --no-spec token streams bitwise
  through the protocol** (greedy; seeded temp 0.8; tiered 1 MB cache +
  pilot); APUS_THREADS 1 vs 8 under spec; the eos stop set firing on
  spec-accepted tokens (generation_config eos = the 3rd greedy token →
  both modes emit exactly 2 tokens, `finish_reason "stop"`); the
  default-on silent fallback note on the MTP-less parrot; explicit
  `--spec` there failing loudly at startup.
- **Parser units** (8): GLM `<tool_call>` parsing (single/multiple,
  typed args, CJK), OpenAI shape, tolerant tails (unterminated think,
  malformed/trailing markup), `ThinkSplitter` char-by-char exactness.
- **HTTP chat** (14 on model_chat): health/models; reasoning/content
  split + usage; usage == `/debug/encode` ids; stop strings (list &
  bare string); seed determinism; SSE (chunk shape, event order,
  reasoning/content reassembly == non-stream, usage chunk, `[DONE]`);
  6-way concurrency; error shapes; **400 on encoding failures**
  (dict content, string tool_call arguments; gateway survives);
  `/v1/completions` (+SSE); gateway-vs-pipe encode equality + jinja2
  golden for a multi-turn conversation; `clear_thinking` dropping
  past-turn reasoning (both spellings); `reasoning_effort` rendering.
- **HTTP tools** (4 on model_tools): tools-as-sibling golden; full
  tool-call round trip (`tool_calls` JSON, `finish_reason:
  "tool_calls"`, usage); streaming finish reason; role=`tool`
  follow-up rendering (`<|observation|>` + `<tool_response>`),
  byte-for-byte vs the jinja2 golden.
- **Auth** (1): `APUS_API_KEY` 401/200, open `/health`.
- **CLI end-to-end** (3): `bin/apus run --ids/--prompt` greedy ==
  the oracle stream (run prints the terminating EOS id; serve never
  emits it), eager and `--tiered`.

## Known limitations

- No KV reuse across turns — every request re-prefills the full
  conversation (single-user local serving; revisit with prefix caching).
- Single engine: requests serialized (see Concurrency). Multi-user
  throughput is a non-goal (ARCHITECTURE §1).
- Streaming tool calls arrive as raw `<tool_call>` text in `content`
  deltas; parse client-side or use non-streaming.
- `/v1/completions` accepts a single string prompt (no batching, no
  `echo`/`logprobs`/`suffix`).
- Engine stop strings are matched on decoded text only (needs a
  tokenizer); at most 16 per request.
- The prefill materializes `[prompt, vocab]` logits (the M5 API): fine
  for serving-scale prompts, but a chunked-continuation prefill that
  keeps only the last chunk's logits is the designed memory
  optimization for very long prompts (M7b+ follow-up; bitwise-safe at
  64-token KDA chunk boundaries per the §4 contract).

## M7b (Metal) / M8 (MTP) touchpoints

- **M7b**: the GLM serve path calls only `apus_gmodel_prefill` /
  `apus_gmodel_decode_step` (and the M8d spec loop goes through c/gmtp.h
  on the same gmodel wiring) — a Metal backend activated inside the
  gmodel wiring is transparent to the protocol and gateway. Re-run
  `test-m7a` on a Metal build to prove tokens unchanged.
- **M8 (MTP speculative decoding)**: SHIPPED in serve (M8d — see
  "Speculative serving" above): output tokens are bitwise identical to
  non-speculative per seed, so the protocol stayed unchanged (the
  accepted burst arrives as ordinary per-token events). Deferred
  follow-ups if per-request control is ever wanted: a `"speculative":
  bool` request field and a `done` counter field (accepted/drafted
  tokens) for the acceptance-rate metric; the gateway passes unknown
  fields through only if whitelisted in `_sampling_params`. KV reuse
  across turns (deferred)
  also interacts with MTP draft state — design them together.
