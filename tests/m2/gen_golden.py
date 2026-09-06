#!/usr/bin/env python3
"""Generate golden test data for M2 (GLM-5.3-Flash tokenizer + chat encoding).

Tokenizer goldens use the HF `tokenizers` library on reference/tokenizer.json.
Encoding goldens render reference/chat_template.jinja with jinja2 in the same
environment HF transformers uses (ImmutableSandboxedEnvironment,
trim_blocks=True, lstrip_blocks=True, loopcontrols for {% break %}, and the
transformers tojson filter = json.dumps with ensure_ascii/separators kwargs).
Outputs into tests/m2/golden/:

  tok_manifest.txt            one case name per line
  <name>.txt                  input text (raw UTF-8 bytes)
  <name>.ids                  [u32 n][n x u32 LE] token ids (encode, specials on)
  <name>.dec                  decode(ids, skip_special_tokens=False) bytes
  <name>.nosplit.ids          ids with special strings treated as plain text
  specials.bin                per-added-token records (covers every added id)
  enc_extra_<N>.json          input spec for encoding case (messages/tools/opts)
  enc_extra_<N>.txt           expected prompt (jinja-rendered template)
  enc_extra_<N>.ids           its token ids
  enc_err_<N>.json            spec that must FAIL to encode (jinja raises too)
  codepoints.bin              (--exhaustive only) per-codepoint probe records
"""

import json
import os
import struct
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tests", "m2", "golden")
TOK_JSON = os.path.join(ROOT, "reference", "tokenizer.json")
TEMPLATE = os.path.join(ROOT, "reference", "chat_template.jinja")

EXHAUSTIVE = "--exhaustive" in sys.argv


def write_ids(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack(f"<{len(ids)}I", *ids) if ids else b"")


def make_renderer():
    # mirror transformers.utils.chat_template_utils._compile_jinja_template
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    return env.from_string(open(TEMPLATE, encoding="utf-8").read())


TPL = make_renderer()


def render(spec):
    kw = {"messages": spec["messages"]}
    if "tools" in spec:
        kw["tools"] = spec["tools"]
    if spec.get("reasoning_effort") is not None:
        kw["reasoning_effort"] = spec["reasoning_effort"]
    if "clear_thinking" in spec:
        kw["clear_thinking"] = spec["clear_thinking"]
    if "add_generation_prompt" in spec:
        kw["add_generation_prompt"] = spec["add_generation_prompt"]
    return TPL.render(**kw)


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = Tokenizer.from_file(TOK_JSON)

    def enc(s):
        return tok.encode(s, add_special_tokens=False).ids

    def dec(ids):
        return tok.decode(ids, skip_special_tokens=False).encode("utf-8")

    paragraph = (
        "The quick brown fox jumps over the lazy dog. In 2026, large language "
        "models grew to 284B parameters — a 37.5% increase over expectations.\n\n"
        "智谱发布了新一代 GLM 模型，它在推理、编程和数学方面表现出色。"
        "日本語のテキストも正しくトークン化されるべきです。한국어도 마찬가지입니다.\n"
        "مرحبا بالعالم! هذا اختبار للنص العربي مع الأرقام ١٢٣٤٥.\n"
        "Code: `def f(x): return x**2 + 2*x - 1  # O(n) solution`\n"
        "URLs: https://example.com/path?q=1&r=2 — emails: user@example.com\n"
        "Emoji: 👋🌍🎉🚀 and math: ∑∫√π ≈ 3.14159, ±0.001, ①②③\n"
        "Combining: café (e + ́) vs café, Devanagari: नमस्ते, Thai: สวัสดี\n"
        "Quotes: “smart” ‘quotes’ — dashes… ellipsis! Punctuation: (a)[b]{c}<d>\n"
        "Contractions: it's DON'T I'm YOU'RE they'll 'Ll 'ſ 'RE ’s ’tis\n"
        "    indented line\n\t tabbed line\n\n\nmultiple blank lines above\n"
    )

    cases = {
        "ascii": "Hello, world! This is a test of the tokenizer: "
                 "numbers 123, 4567, 89012 and symbols!!! Don't stop—it's "
                 "fine. 3.14159 and 1e10; snake_case and CamelCase.",
        "utf8_multi": "中文测试：深度学习模型。日本語：こんにちは世界。"
                      "한국어: 안녕하세요. العربية: مرحبا بالعالم ١٢٣. "
                      "Emoji 👋🌍🎉🚀👨‍👩‍👧‍👦. Combining: áéó "
                      "नमस्ते दुनिया. สวัสดีชาวโลก. €£¥₿ ∑∫√π±≠≤≥.",
        "specials_text": "Use <|user|> and <think> like this: <|assistant|>"
                         "<think>reasoning</think>answer<|observation|>. "
                         "Also <tool_call> markup <arg_key>k</arg_key> and "
                         "<arg_value>v</arg_value> mid-text plus </think> and "
                         "[gMASK]<sop> and [MASK] [sMASK] <sop> <eop> /nothink "
                         "<|begin_of_image|><|image|><|end_of_image|> here "
                         "<|begin_of_box|><|end_of_box|><|video|><|code_prefix|>.",
        "contractions": "it's I'M you're YOU'RE they'll HE'LL don't DON'T "
                        "we've THEY'VE 's 'S 'ſ 't 'T 're 'RE 'rE 've 'VE "
                        "'m 'M 'll 'LL 'Ll 'd 'D 'x 'ſt ’tis ’s a'b x'y",
        "whitespace": "a\n\n  b\t c \n d   \n\n\ne\r\nf \t\ng \n \n h"
                      "\n \n\n \t\n\n\n   trailing   \n\n\n!\n\n! \n  x  ",
        "empty": "",
        "paragraph": paragraph,
        "long": paragraph * 40,
    }

    manifest = []
    for name, text in cases.items():
        ids = enc(text)
        with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8",
                  newline="") as f:
            f.write(text)
        write_ids(os.path.join(OUT, name + ".ids"), ids)
        with open(os.path.join(OUT, name + ".dec"), "wb") as f:
            f.write(dec(ids))
        # round-trip must be stable for valid UTF-8 input
        assert dec(ids).decode("utf-8") == text, f"round-trip unstable: {name}"
        manifest.append(name)

    # nosplit variants: special strings treated as plain text
    tj = json.load(open(TOK_JSON, encoding="utf-8"))
    tj_plain = dict(tj)
    tj_plain["added_tokens"] = []
    tmp_path = os.path.join(OUT, "_tokenizer_plain.json")
    with open(tmp_path, "w") as f:
        json.dump(tj_plain, f)
    tok_plain = Tokenizer.from_file(tmp_path)
    os.remove(tmp_path)
    for name in ("specials_text", "ascii"):
        ids = tok_plain.encode(cases[name], add_special_tokens=False).ids
        write_ids(os.path.join(OUT, name + ".nosplit.ids"), ids)

    with open(os.path.join(OUT, "tok_manifest.txt"), "w", encoding="utf-8",
              newline="") as f:
        f.write("\n".join(manifest) + "\n")

    # ---- specials coverage: every added token id at least once ----
    covered = set()
    with open(os.path.join(OUT, "specials.bin"), "wb") as f:
        for a in tj["added_tokens"]:
            content = a["content"]
            cid = a["id"]
            data = content.encode("utf-8")
            ids = enc(content)
            if ids != [cid]:
                raise RuntimeError(
                    f"added token {cid} {content!r} encodes to {ids}, expected [{cid}]")
            f.write(struct.pack("<III", cid, len(data), len(ids)))
            f.write(data)
            f.write(struct.pack(f"<{len(ids)}I", *ids))
            covered.add(cid)
    n_added = len(tj["added_tokens"])
    assert len(covered) == n_added, (len(covered), n_added)
    print(f"specials: all {n_added} added ids covered")

    # ---- encoding cases (chat template conformance) ----
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
    calc_tool = {  # flat form (no OpenAI wrapper)
        "name": "calc",
        "description": "Calculate π ≈ 3.14159",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}, "prec": {"type": "integer"}},
        },
        "strict": True,  # excluded from the rendered JSON by the template
    }
    deferred_tool = {
        "name": "deferred",
        "description": "not listed",
        "defer_loading": True,
        "parameters": {},
    }
    extras = [
        # 1: minimal: one user message + generation prompt
        {"messages": [{"role": "user", "content": "hi"}],
         "add_generation_prompt": True},
        # 2: system + user + assistant (no reasoning -> <think></think>),
        #    no generation prompt
        {"messages": [{"role": "system", "content": "You are helpful."},
                      {"role": "user", "content": "2+2?"},
                      {"role": "assistant", "content": "4."}]},
        # 3: reasoning_effort low
        {"messages": [{"role": "user", "content": "hi"}],
         "reasoning_effort": "low", "add_generation_prompt": True},
        # 4: reasoning_effort high
        {"messages": [{"role": "user", "content": "hi"}],
         "reasoning_effort": "high", "add_generation_prompt": True},
        # 5: reasoning_effort bogus -> Max
        {"messages": [{"role": "user", "content": "hi"}],
         "reasoning_effort": "ludicrous", "add_generation_prompt": True},
        # 6: assistant reasoning kept (default clear_thinking=false)
        {"messages": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "reasoning_content": "deep thought",
                       "content": "a1"},
                      {"role": "user", "content": "q2"},
                      {"role": "assistant", "reasoning_content": "fresh",
                       "content": "a2"}],
         "add_generation_prompt": True},
        # 7: clear_thinking: drops reasoning up to the last user message,
        #    keeps it after
        {"messages": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "reasoning_content": "old", "content": "a1"},
                      {"role": "user", "content": "q2"},
                      {"role": "assistant", "reasoning_content": "new", "content": "a2"}],
         "clear_thinking": True, "add_generation_prompt": True},
        # 8: <think>...</think> embedded in content (no reasoning_content);
        #    multiple </think> — split on first for reasoning, last for content
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant",
                       "content": "<think>part1</think>mid</think>visible"}]},
        # 9: assistant content stripped (Python str.strip); whitespace-only
        #    content vanishes
        {"messages": [{"role": "user", "content": "  kept  "},
                      {"role": "assistant", "reasoning_content": "r",
                       "content": "\n  padded answer \t\n"},
                      {"role": "user", "content": "q2"},
                      {"role": "assistant", "content": "   "}]},
        # 10: tools block: OpenAI-wrapped + flat + deferred; non-ASCII values
        {"messages": [{"role": "user", "content": "weather in Paris and 2+2?"}],
         "tools": [weather_tool, calc_tool, deferred_tool],
         "add_generation_prompt": True},
        # 11: assistant tool_calls: string args verbatim, non-string args as
        #     JSON (floats, bools, null, nested, big ints, unicode)
        {"messages": [{"role": "user", "content": "run the numbers"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "n1", "type": "function",
                           "function": {"name": "calc", "arguments": {
                               "f1": 1.5, "f2": -0.0, "f3": 1e3, "f4": 0.0001,
                               "f5": 1e-5, "f6": 3.141592653589793, "f7": 1e16,
                               "f8": 2.5e-8,
                               "big": 123456789012345678901234567890,
                               "t": True, "z": False, "n": None,
                               "arr": [1, 2.25, "x"], "obj": {"k": "v", "e": {}},
                               "s": "plain string", "uni": "π中文"}}}]}],
         "tools": [calc_tool]},
        # 12: tool responses sorted back into call order (input reversed)
        {"messages": [{"role": "user", "content": "weather in Paris and 2+2?"},
                      {"role": "assistant",
                       "reasoning_content": "need two calls",
                       "tool_calls": [
                           {"id": "call_a", "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": {"location": "Paris"}}},
                           {"id": "call_b", "type": "function",
                            "function": {"name": "calc",
                                         "arguments": {"expr": "2+2", "prec": 0}}}],
                       "content": ""},
                      {"role": "tool", "tool_call_id": "call_b", "content": "4"},
                      {"role": "tool", "tool_call_id": "call_a",
                       "content": "sunny, 20°C"},
                      {"role": "assistant", "reasoning_content": "have both",
                       "content": "Paris is sunny, 20°C; 2+2 = 4."}],
         "tools": [weather_tool, calc_tool], "add_generation_prompt": True},
        # 13: unsortable tool block (unknown id) -> message order
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"name": "f", "arguments": {}}}]},
                      {"role": "tool", "tool_call_id": "zz", "content": "rz"},
                      {"role": "tool", "tool_call_id": "c1", "content": "r1"}]},
        # 14: unsortable: duplicate result ids; tool message with no
        #     preceding assistant
        {"messages": [{"role": "tool", "tool_call_id": "x", "content": "lone"},
                      {"role": "tool", "tool_call_id": "x", "content": "dupe"},
                      {"role": "user", "content": "after"}]},
        # 15: tool content as list: text parts concatenated, image tokens
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"name": "f", "arguments": {}}},
                          {"id": "c2", "function": {"name": "g", "arguments": {}}}]},
                      {"role": "tool", "tool_call_id": "c1",
                       "content": [{"type": "text", "text": "multi"},
                                   {"type": "text", "text": "parts"}]},
                      {"role": "tool", "tool_call_id": "c2",
                       "content": [{"type": "image"},
                                   {"type": "text", "text": "pic"}]}]},
        # 16: tool_reference response (schema re-emit from tools)
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"name": "get_weather",
                                                    "arguments": {}}}]},
                      {"role": "tool", "tool_call_id": "c1",
                       "content": [{"type": "tool_reference",
                                    "name": "get_weather"}]}],
         "tools": [weather_tool]},
        # 17: list-of-outputs messages, sorted per entry
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"name": "f", "arguments": {}}},
                          {"id": "c2", "function": {"name": "g", "arguments": {}}}]},
                      {"role": "tool",
                       "content": [{"tool_call_id": "c2", "output": "o2"},
                                   {"tool_call_id": "c1", "output": "o1"}]}]},
        # 18: two tool rounds in one conversation
        {"messages": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "a1", "function": {"name": "f", "arguments": {}}}]},
                      {"role": "tool", "tool_call_id": "a1", "content": "r1"},
                      {"role": "assistant", "reasoning_content": "again",
                       "tool_calls": [
                           {"id": "b1", "function": {"name": "g",
                                                     "arguments": {"x": 1}}}]},
                      {"role": "tool", "tool_call_id": "b1", "content": "r2"},
                      {"role": "assistant", "content": "done"}],
         "add_generation_prompt": True},
        # 19: user content as list (text + image + bare string items)
        {"messages": [{"role": "user",
                       "content": [{"type": "text", "text": "look: "},
                                   {"type": "image"},
                                   "tail"]},
                      {"role": "assistant", "content": "seen"}]},
        # 20: unknown roles silently skipped; numeric and null content
        {"messages": [{"role": "system", "content": "s"},
                      {"role": "weird", "content": "ignored"},
                      {"role": "user", "content": 42},
                      {"role": "assistant", "content": None}],
         "add_generation_prompt": True},
        # 21: unicode + special chars in contents and tool JSON
        {"messages": [{"role": "system", "content": "π中文 emoji 🎉 \"quoted\" <xml>"},
                      {"role": "user", "content": "line1\nline2\ttab"},
                      {"role": "assistant",
                       "reasoning_content": "thinking with <think> literal",
                       "content": "answer with </think> literal"}],
         "tools": [{"name": "uni", "description": "déjà vu 中文",
                    "parameters": {"type": "object"}}]},
    ]
    for i, spec in enumerate(extras, start=1):
        # make the opts unambiguous: the C side would otherwise apply its
        # engine default (add_generation_prompt=1) while jinja defaults to
        # false
        spec.setdefault("add_generation_prompt", False)
        prompt = render(spec)
        with open(os.path.join(OUT, f"enc_extra_{i}.json"), "w") as f:
            json.dump(spec, f, ensure_ascii=False, indent=1)
        with open(os.path.join(OUT, f"enc_extra_{i}.txt"), "w", encoding="utf-8",
                  newline="") as f:
            f.write(prompt)
        write_ids(os.path.join(OUT, f"enc_extra_{i}.ids"), enc(prompt))
    print(f"encoding: {len(extras)} cases")

    # ---- error cases: the template raises; the C encoder must fail too ----
    errors = [
        # 1: arguments as a JSON string (jinja: no .items on str)
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"name": "f",
                                                    "arguments": "{\"a\": 1}"}}]}]},
        # 2: tool_call without arguments
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"name": "f"}}]}]},
        # 3: tool_call without name
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "tool_calls": [
                          {"id": "c1", "function": {"arguments": {}}}]}]},
    ]
    for i, spec in enumerate(errors, start=1):
        try:
            render(spec)
            raise RuntimeError(f"error case {i} rendered successfully")
        except jinja2.exceptions.TemplateError:
            pass
        with open(os.path.join(OUT, f"enc_err_{i}.json"), "w") as f:
            json.dump(spec, f, ensure_ascii=False, indent=1)
    print(f"encoding errors: {len(errors)} cases")

    # ---- exhaustive per-codepoint probes (optional) ----
    if EXHAUSTIVE:
        cps = list(range(0x00, 0x30000))
        cps += range(0x30000, 0x110000, 17)
        cps = [c for c in cps if not (0xD800 <= c <= 0xDFFF)]
        texts = []
        index = []  # (cp,) per group of 6
        for cp in cps:
            c = chr(cp)
            texts += [c, c * 2, "a" + c + "b", c + "\n ", c * 4, "'" + c + "x"]
            index.append(cp)
        print(f"encoding {len(texts)} probe strings...")
        results = tok.encode_batch(texts, add_special_tokens=False)
        path = os.path.join(OUT, "codepoints.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<II", 0xC0DE0002, len(index) * 6))
            k = 0
            for cp in index:
                for _ in range(6):
                    s = texts[k].encode("utf-8")
                    ids = results[k].ids
                    f.write(struct.pack("<II", len(s), len(ids)))
                    f.write(s)
                    if ids:
                        f.write(struct.pack(f"<{len(ids)}I", *ids))
                    k += 1
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")

    print("golden data written to", OUT)


if __name__ == "__main__":
    main()
