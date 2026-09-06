# tests/m2 — GLM-5.3-Flash tokenizer + chat template (M2)

Gate: `make test-m2` (builds + regenerates goldens + runs both C tests),
`make ubsan-m2`, `make golden-exhaustive` for the full codepoint probe.
Apple's ASan runtime hangs in dyld init on the dev Mac, so the sanitizer
gate is UBSan-only (`asan-m2` exists for platforms where ASan works).

## What is covered

### Tokenizer (`c/tok.h` vs HF `tokenizers` on `reference/tokenizer.json`)

GLM-5.3-Flash uses a GPT-2-style byte-level BPE: vocab 154,820 (ids
0–154,819) + 36 added tokens (ids 154,820–154,855, contiguous) = 154,856
tokenizer ids. Model `config.json` vocab_size 154,880 is padding.

- `test_tok` manifest cases: ASCII, multi-script UTF-8, specials-in-text,
  contractions/case-folding, whitespace torture, empty, paragraph, long
  (paragraph ×40). For each: encode twice (determinism), exact id match vs
  golden, exact decode-bytes match, round-trip `decode(encode(x)) == x`,
  plus `nosplit` variants with added-token recognition off.
- `specials.bin`: all 36 added tokens encode to their own id and decode
  back byte-exact.
- `codepoints.bin` (`make golden-exhaustive`, magic `0xC0DE0002`):
  **1,491,186 probe records** — every codepoint 0x00–0x2FFFF plus
  0x30000–0x10FFFF step 17 (surrogates excluded), six probe strings each
  (`c`, `c*2`, `a+c+b`, `c+"\n "`, `c*4`, `"'"+c+"x"`), exact id match vs
  HF. The sixth probe pins the case-insensitive contraction alternative.
- Edge cases: empty input, invalid UTF-8 bytes (no crash), pinned ids for
  the generation_config eos set {154820, 154827, 154829}, [gMASK], <sop>,
  <think>.

### Pre-tokenizer replication strategy

The tokenizer.json pre-tokenizer is a single `Split` (behavior Isolated)
with the Qwen2-style regex, then `ByteLevel(use_regex=false)`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
|  ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
```

`c/tok.h` implements this as one leftmost-first, alternation-priority,
greedy matcher (`split_match`) — the same semantics the tokenizers regex
engine exhibits, verified exhaustively. Notable pinned facts:

- `(?i:...)` is Unicode simple case folding: ASCII upper/lower plus
  **U+017F (ſ) folds to 's'** (probed: `"'ſt"` splits as `'ſ` + `t`).
- The A2 prefix `[^\r\n\p{L}\p{N}]?` accepts whitespace and symbols, so
  e.g. `"\tabc"` and `" abc"` (NBSP) are single pieces.
- ` ?` in the symbol alternative is the ASCII space only.
- Every codepoint matches some alternative — pieces tile the input, no
  gaps (the C side keeps a defensive gap path, never taken).
- Unicode classes `\p{L}`, `\p{N}`, `\s` come from `c/uni_tables.h`, which
  is GENERATED FROM THE TOKENIZER'S OWN BEHAVIOR
  (`tests/m2/gen_uni_tables.py`), not from a Unicode database — no
  Unicode-version drift. Probing wraps each candidate in `7…b` (a digit
  and a letter make probe boundaries uncrossable) and classifies from
  piece offsets: `7ccccb` → digit chunking for `\p{N}`, `7aXbb` → single
  letter run for `\p{L}`, `7c cb` → 3 vs 4 pieces for `\s` vs symbol.
  Result: N = 143 ranges/1,901 cps, L = 675 ranges/140,976 cps, WS = 8
  ranges/19 cps (0x85, 0xA0, 0x1680, 0x2000–0x200A, 0x2028/9, 0x202F,
  0x205F, 0x3000). ASCII is hardcoded.
- Merges serialize as `["l","r"]` pairs (legacy `"l r"` accepted). All
  321,649 GLM merges reference vocab tokens, so `ignore_merges: true` has
  nothing to skip (the loader still skips-and-reports, mirroring the
  flag's semantics). Merge ranks go past 0x1FFFF — the best-rank sentinel
  is UINT32_MAX.
- The post-processor adds nothing: `add_special_tokens=True` ≡ `False`.
- The HF decoder is lossy (invalid UTF-8 → U+FFFD); `tok_decode` passes
  bytes through verbatim — identical for any ids produced from valid
  UTF-8 text (this is what the goldens exercise).

### Added tokens

All 36 added tokens (special AND non-special) are matched literally
during encoding, leftmost-longest (no prefix collisions exist among
them). Inventory: `<|endoftext|>` 154820, `[MASK]`/`[gMASK]`/`[sMASK]`
154821–3, `<sop>`/`<eop>` 154824–5, `<|system|>`/`<|user|>`/
`<|assistant|>`/`<|observation|>` 154826–9, media pairs
154830–7, `<|code_*|>` 154838–40, `<think>`/`</think>` 154841–2,
`<tool_call>` pair 154843–4, `<tool_response>` pair 154845–6,
`<arg_key>`/`<arg_value>` pairs 154847–50, `/nothink` 154851,
box/image/video 154852–5. eos set (generation_config): {154820, 154827,
154829} = `<|endoftext|>`, `<|user|>`, `<|observation|>`.

### Chat template (`c/encoding.h` vs `reference/chat_template.jinja`)

Goldens are rendered with jinja2 in the HF transformers environment
(ImmutableSandboxedEnvironment, trim_blocks + lstrip_blocks, loopcontrols
extension for `{% break %}`, transformers' `tojson` filter = Python
`json.dumps` with kwargs). 21 conformance cases + 3 error cases; each
conformance case byte-compares the prompt and the token ids; error cases
must fail (the template raises on them too). Determinism double-encode.

Template semantics (all verified against the jinja; it wins):

- Always starts `[gMASK]<sop><|system|>Reasoning Effort: Low|High|Max`.
  Only the exact strings `low`/`high` are honored; everything else
  (unset, `max`, bogus) → `Max`. No newlines between blocks.
- `tools` is a top-level parameter (NOT a message field):
  `<|system|>\n# Tools\n…` block; OpenAI-wrapped tools unwrapped (key
  presence), `defer_loading` tools skipped, `defer_loading`/`strict`
  keys dropped from the per-tool JSON, values via Python json.dumps
  (", "/": " separators, ensure_ascii=False).
- `<|user|>` + visible content (NOT stripped). `<|system|>` + content.
- `<|assistant|>` immediately followed by `<think>…</think>` (no
  newline): reasoning from string `reasoning_content`, else extracted
  from content between `<think>`/`</think>` (split on FIRST `</think>`
  for reasoning — after its LAST `<think>` — and on LAST `</think>` for
  content). `clear_thinking` drops reasoning up to the last user message
  only. Content is `str.strip()`ed (CPython whitespace set); empty after
  strip → omitted. Missing content renders "" (jinja Undefined); explicit
  null renders `None`.
- Tool calls: `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>
  …</tool_call>`, arguments MUST be an object (string arguments raise in
  jinja and fail in C); string values verbatim, others json.dumps.
  OpenAI-wrapped calls unwrapped (truthiness of `function`).
- Consecutive tool messages = one block, prefixed once with
  `<|observation|>`; each response wrapped `<tool_response>…</tool_response>`.
  Responses are re-sorted into the preceding assistant's tool_call order
  iff every block id is truthy, unique, and present among the tool_call
  ids AND every tool_call id is truthy and unique; otherwise message
  order. Content lists concatenate visible items (text / image / video /
  audio tokens — audio has NO middle token); `tool_reference` items
  re-emit the referenced schemas from `tools`; list-of-outputs messages
  (`content: [{tool_call_id, output}, …]`) sort per entry.
- Unknown roles are silently skipped. Generation prompt:
  `<|assistant|><think>`.
- There is NO thinking on/off switch and NO BOS/EOS emission: assistant
  turns end with their content/tool calls; stopping relies on the eos id
  set above.

Documented divergences (template renders Python internals; out of
scope): dict-typed message content (Python `repr`) is a C error;
non-string tool_call/result ids compare as missing.

## Regenerating

```
.venv/bin/python tests/m2/gen_uni_tables.py      # c/uni_tables.h (rarely)
make golden                                       # tests/m2/golden/*
make golden-exhaustive                            # + codepoints.bin (63 MB)
```

Python deps: `tokenizers`, `jinja2` (in the project .venv).
