#!/usr/bin/env python3
"""tests/m7a/test_server.py — M7a server test suite, GLM-5.3-Flash (stdlib
only, plus jinja2 for the template-conformance goldens — the tests/m2
rendering environment).

Layers:
  1. Pipe protocol tests against `bin/apus serve` directly (NDJSON), on the
     scripted GLM parrot fixtures: encode goldens (jinja2-rendered from
     reference/chat_template.jinja at test time), event sequence, EOS
     handling, stop strings, usage, bad_request error codes, ids path,
     multi-chunk prefill, and the ORACLE TOKEN GATE (engine token stream ==
     the oracle's greedy stream from the same container, bitwise — the
     parrot's >100 logit margins make this host-exp independent). A tiered
     (--tiered, 1 MB cache) pipe must produce the identical stream
     (eager == tiered == pilot end to end).
  1b. serve --spec (M8d): MTP speculative decoding in serve mode on the
     tests/m8g dsa_top container (the parrot has no MTP block) — spec vs
     --no-spec bitwise stream equivalence through the NDJSON protocol
     (greedy + seeded temp 0.8, eager + tiered, APUS_THREADS 1/8), the
     eos stop set firing on spec-accepted tokens, the default-on silent
     fallback note on MTP-less containers, and the loud explicit --spec
     failure there.
  2. Gateway parser unit tests (GLM <tool_call> parsing, ThinkSplitter).
  3. HTTP gateway tests (tools/server.py): chat (reasoning split, stop
     strings, max_tokens, seed determinism, usage), SSE streaming, request
     schema validation (400s, incl. glm_encode failures — the M2 handoff),
     template conformance through /debug/encode (multi-turn,
     clear_thinking, reasoning_effort, tools sibling), tool-call round
     trip + tool-result follow-up rendering, auth, concurrency.
  4. CLI end-to-end: `bin/apus run` (eager and --tiered) on the parrot
     container, greedy --quiet ids == the oracle stream.

Run from the repo root: `.venv/bin/python tests/m7a/test_server.py [-v]`.
Env APUS_BIN selects the engine binary (ubsan-m7a uses bin/apus_ubsan).
Fixtures are (re)generated with `make golden-m7a` if missing.
"""

import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APUS = os.path.abspath(os.environ.get("APUS_BIN", os.path.join(ROOT, "bin", "apus")))
SERVER = os.path.join(ROOT, "tools", "server.py")
FIX = os.path.join(ROOT, "tests", "m7a", "fixtures")
TEMPLATE = os.path.join(ROOT, "reference", "chat_template.jinja")

sys.path.insert(0, os.path.join(ROOT, "tools"))
import server as gw  # noqa: E402

THINKING_REPLY = "reasoning: thinking it over."
CHAT_REPLY = "The answer is STOP right here."
TOOL_REPLY = "I should check the weather."
TOOL_CALL_TEXT = ("<tool_call>get_weather<arg_key>location</arg_key>"
                  "<arg_value>Beijing</arg_value></tool_call>")
# parrot tokenizer special ids (tests/m7a/gen_fixtures.py)
EOS, GMASK, SOP = 256, 258, 260
SYSTEM, USER, ASSISTANT, THINK, THINK_END = 262, 263, 264, 266, 267


def ensure_fixtures():
    if os.path.isfile(os.path.join(FIX, "model_chat", "config.json")):
        return
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "tests", "m7a", "gen_fixtures.py")],
                   check=True, cwd=ROOT)


def golden():
    ensure_fixtures()
    with open(os.path.join(FIX, "golden.json")) as f:
        return json.load(f)


# ---- jinja2 goldens (the tests/m2 rendering environment) --------------------


def make_renderer():
    import jinja2  # noqa: F401  (required, like tests/m2)
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    with open(TEMPLATE, encoding="utf-8") as f:
        return env.from_string(f.read())


TPL = None


def render(messages, tools=None, clear_thinking=None, reasoning_effort=None):
    global TPL
    if TPL is None:
        TPL = make_renderer()
    kw = {"messages": messages, "add_generation_prompt": True}
    if tools is not None:
        kw["tools"] = tools
    if clear_thinking is not None:
        kw["clear_thinking"] = clear_thinking
    if reasoning_effort is not None:
        kw["reasoning_effort"] = reasoning_effort
    return TPL.render(**kw)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------
# 1. engine pipe protocol
# ---------------------------------------------------------------------------


class Pipe:
    def __init__(self, model_dir, tiered=False, env=None, extra=(),
                 capture_stderr=False):
        cmd = [APUS, "serve", "--model", model_dir]
        if tiered:
            cmd.append("--tiered")
        cmd += list(extra)
        e = dict(os.environ)
        if env:
            e.update(env)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            text=True, cwd=ROOT, env=e)

    def rpc(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        events = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise AssertionError("engine died")
            ev = json.loads(line)
            events.append(ev)
            if ev["type"] in ("done", "error", "encoded"):
                return events

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
        self.proc.stdout.close()
        if self.proc.stderr:
            err = self.proc.stderr.read()
            self.proc.stderr.close()
            return err
        return ""


class PipeProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_fixtures()
        cls.gold = golden()
        cls.pipe = Pipe(os.path.join(FIX, "model_chat"))

    @classmethod
    def tearDownClass(cls):
        cls.pipe.close()

    def req(self, payload, i=1):
        return self.pipe.rpc(dict(payload, id=i))

    def test_encode_basic(self):
        msgs = [{"role": "user", "content": "hi"}]
        ev = self.req({"cmd": "encode", "messages": msgs})[0]
        self.assertEqual(ev["type"], "encoded")
        self.assertEqual(ev["text"], render(msgs))   # jinja2 golden
        ids = ev["ids"]
        self.assertEqual(ids[:3], [GMASK, SOP, SYSTEM])
        self.assertEqual(ids[-5:], [USER, ord("h"), ord("i"),
                                    ASSISTANT, THINK])

    def test_encode_effort_and_clear(self):
        msgs = [{"role": "user", "content": "hi"}]
        for effort, word in (("low", "Low"), ("high", "High"),
                             ("bogus", "Max"), (None, "Max")):
            req = {"cmd": "encode", "messages": msgs}
            if effort:
                req["reasoning_effort"] = effort
            ev = self.req(req)[0]
            self.assertIn(f"Reasoning Effort: {word}", ev["text"])
            self.assertEqual(ev["text"],
                             render(msgs, reasoning_effort=effort))

    def test_error_recovery(self):
        # malformed line, then unknown cmd, then a good request — the loop
        # must survive all of it
        self.pipe.proc.stdin.write("this is not json\n")
        self.pipe.proc.stdin.flush()
        ev = json.loads(self.pipe.proc.stdout.readline())
        self.assertEqual(ev["type"], "error")
        self.assertEqual(ev["code"], "bad_request")
        self.assertIsNone(ev["id"])
        ev = self.req({"cmd": "bogus"})[0]
        self.assertEqual(ev["type"], "error")
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "temperature": 0})
        self.assertEqual(ev[-1]["type"], "done")

    def test_bad_request_schema(self):
        # glm_encode failure (dict-typed content) -> bad_request, loop alive
        ev = self.req({"cmd": "encode",
                       "messages": [{"role": "user", "content": {"x": 1}}]})[0]
        self.assertEqual(ev["type"], "error")
        self.assertEqual(ev["code"], "bad_request")
        ev = self.req({"cmd": "generate", "messages": "hi"})[0]
        self.assertEqual(ev["type"], "error")
        self.assertEqual(ev["code"], "bad_request")
        ev = self.req({"cmd": "generate", "ids": []})[0]
        self.assertEqual(ev["code"], "bad_request")
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1, "temperature": 0})
        self.assertEqual(ev[-1]["type"], "done")

    def test_generate_events_and_eos(self):
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 64, "temperature": 0})
        self.assertEqual(ev[0]["type"], "prompt")
        toks = [e for e in ev if e["type"] == "token"]
        done = ev[-1]
        self.assertEqual(done["finish_reason"], "stop")   # EOS
        self.assertNotIn(EOS, [t["token_id"] for t in toks])  # EOS not emitted
        self.assertEqual(done["completion_tokens"], len(toks))
        self.assertEqual("".join(t["text"] for t in toks), done["text"])
        self.assertEqual(done["text"],
                         THINKING_REPLY + "</think>" + CHAT_REPLY)

    def test_oracle_token_gate(self):
        # THE end-to-end bitwise gate: engine token stream == the oracle's
        # greedy stream from the same container (parrot margins > 100 make
        # this host-exp independent). The oracle stream ends in EOS, which
        # the engine never emits.
        stream = self.gold["variants"]["model_chat"]["oracle_stream"]
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 64, "temperature": 0})
        toks = [e["token_id"] for e in ev if e["type"] == "token"]
        self.assertEqual(toks, stream[:-1])
        self.assertEqual(stream[-1], EOS)
        self.assertEqual(ev[-1]["finish_reason"], "stop")

    def test_long_prompt_chunked_prefill(self):
        # >64 tokens => the chunked-KDA prefill continuation path; the
        # parrot is Markov in the last token, so the stream is unchanged.
        ids = [ord("a")] * 100 + [THINK]
        ev = self.req({"cmd": "generate", "ids": ids,
                       "max_tokens": 64, "temperature": 0}, i=5)
        self.assertEqual(ev[0]["prompt_tokens"], 101)
        toks = [e["token_id"] for e in ev if e["type"] == "token"]
        stream = self.gold["variants"]["model_chat"]["oracle_stream"]
        self.assertEqual(toks, stream[:-1])

    def test_usage_matches_encode_ids(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "count my tokens"}]
        enc = self.req({"cmd": "encode", "messages": msgs}, i=2)[0]
        gen = self.req({"cmd": "generate", "messages": msgs,
                        "max_tokens": 1, "temperature": 0}, i=3)
        self.assertEqual(gen[0]["prompt_tokens"], len(enc["ids"]))
        self.assertEqual(gen[-1]["prompt_tokens"], len(enc["ids"]))
        self.assertEqual(gen[-1]["completion_tokens"], 1)
        self.assertEqual(gen[-1]["finish_reason"], "length")

    def test_stop_string(self):
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 64, "temperature": 0, "stop": ["STOP"]})
        done = ev[-1]
        self.assertEqual(done["finish_reason"], "stop_string")
        self.assertEqual(done["text"],
                         THINKING_REPLY + "</think>" + "The answer is ")
        # the streamed pieces never contain the stop string
        toks = [e for e in ev if e["type"] == "token"]
        self.assertNotIn("STOP", "".join(t["text"] for t in toks))

    def test_max_tokens_clamp(self):
        ev = self.req({"cmd": "generate",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "temperature": 0})
        done = ev[-1]
        self.assertEqual(done["finish_reason"], "length")
        self.assertEqual(done["completion_tokens"], 2)

    def test_tiered_stream_identical(self):
        # --tiered (1 MB cache, evictions forced) + the M6 pilot attached:
        # the token stream must be identical to the eager path end to end.
        p = Pipe(os.path.join(FIX, "model_chat"), tiered=True,
                 env={"APUS_GEXPERT_CACHE_MB": "1"})
        try:
            ev = p.rpc({"id": 1, "cmd": "generate",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 64, "temperature": 0})
            toks = [e["token_id"] for e in ev if e["type"] == "token"]
            stream = self.gold["variants"]["model_chat"]["oracle_stream"]
            self.assertEqual(toks, stream[:-1])
            self.assertEqual(ev[-1]["text"],
                             THINKING_REPLY + "</think>" + CHAT_REPLY)
        finally:
            p.close()


# ---------------------------------------------------------------------------
# 1b. serve --spec (M8d): MTP speculative decoding in serve mode
# ---------------------------------------------------------------------------
#
# The m7a parrot fixtures carry no MTP block (they exercise the silent
# fallback below); the MTP-bearing fixture is the tests/m8g dsa_top
# container (made by golden-m8g, a test-m7a Makefile dependency). The
# accept rule makes the emitted stream bitwise identical to non-spec at
# the same seed — asserted here through the NDJSON protocol, greedy AND
# sampled, eager AND tiered, at APUS_THREADS 1 and 8. The golden dirs are
# never mutated: each leg copies the container to a temp dir (36 MB) and
# the eos leg drops a generation_config.json into its own copy.

M8G = os.path.join(ROOT, "tests", "m8g", "golden", "dsa_top")


def m8g_prompt_ids():
    with open(os.path.join(M8G, "prompt_ids.bin"), "rb") as f:
        d = f.read()
    return list(struct.unpack("<%di" % (len(d) // 4), d))


def m8g_container(tmp):
    """A private copy of the dsa_top container inside tmp."""
    dst = os.path.join(tmp, "container")
    shutil.copytree(os.path.join(M8G, "container"), dst)
    cfg = os.path.join(dst, "config.json")
    if not os.path.isfile(cfg):
        shutil.copy(os.path.join(M8G, "config.json"), cfg)
    return dst


@unittest.skipUnless(
    os.path.isfile(os.path.join(M8G, "container", "apus.index.json")),
    "m8g fixtures missing (make golden-m8g)")
class ServeSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = m8g_prompt_ids()

    def gen(self, pipe, **kw):
        req = {"id": 1, "cmd": "generate", "ids": self.ids,
               "max_tokens": 16, "seed": 7}
        req.update(kw)
        ev = pipe.rpc(req)
        assert ev[-1]["type"] == "done", ev
        toks = [e["token_id"] for e in ev if e["type"] == "token"]
        return toks, ev[-1]

    def both_modes(self, cont, tiered=False, env=None, **kw):
        """Same seeded request through --spec and --no-spec pipes."""
        out = {}
        for mode, args in (("spec", ["--spec", "--spec-k", "3"]),
                           ("nonspec", ["--no-spec"])):
            p = Pipe(cont, tiered=tiered, env=env, extra=args,
                     capture_stderr=(mode == "spec"))
            try:
                out[mode] = self.gen(p, **kw)
            finally:
                err = p.close()
                if mode == "spec":
                    out["spec_stderr"] = err
        return out

    def check_equiv(self, r, expect_tokens=16, finish="length"):
        (st, sd), (nt, nd) = r["spec"], r["nonspec"]
        self.assertIn(", spec", r["spec_stderr"])   # spec really loaded
        self.assertEqual(nd["finish_reason"], finish)
        self.assertEqual(nd["completion_tokens"], expect_tokens)
        self.assertEqual(st, nt)       # the bitwise stream equivalence
        self.assertEqual(sd, nd)       # done event identical too

    def test_spec_nonspec_greedy(self):
        with tempfile.TemporaryDirectory(prefix="apus_m8d_") as tmp:
            r = self.both_modes(m8g_container(tmp), temperature=0.0)
            self.check_equiv(r)

    def test_spec_nonspec_sampled(self):
        # temp > 0: spec consumes exactly one RNG uniform per emitted
        # token in position order (drafts none) — same seed => same stream
        with tempfile.TemporaryDirectory(prefix="apus_m8d_") as tmp:
            r = self.both_modes(m8g_container(tmp),
                                temperature=0.8, top_p=0.95)
            self.check_equiv(r)

    def test_spec_nonspec_tiered(self):
        # the M6 cache (1 MB budget, evictions forced) + pilot under spec
        with tempfile.TemporaryDirectory(prefix="apus_m8d_") as tmp:
            r = self.both_modes(m8g_container(tmp), tiered=True,
                                env={"APUS_GEXPERT_CACHE_MB": "1"},
                                temperature=0.8, top_p=0.95)
            self.check_equiv(r)

    def test_spec_thread_count_independence(self):
        # APUS_THREADS 1 vs 8 under --spec: identical streams (the m7a
        # harness itself runs at the ambient thread count)
        with tempfile.TemporaryDirectory(prefix="apus_m8d_") as tmp:
            cont = m8g_container(tmp)
            streams = []
            for t in ("1", "8"):
                p = Pipe(cont, extra=["--spec"], env={"APUS_THREADS": t})
                try:
                    streams.append(self.gen(p, temperature=0.8,
                                            top_p=0.95))
                finally:
                    p.close()
            self.assertEqual(streams[0], streams[1])

    def test_spec_eos_stop_set(self):
        # The eos stop set must apply to every emitted token, including
        # spec-accepted ones: set eos to the 3rd greedy token (via a
        # generation_config.json in a private container copy) — both modes
        # must emit exactly the 2 preceding tokens and finish "stop".
        with tempfile.TemporaryDirectory(prefix="apus_m8d_") as tmp:
            ref = Pipe(m8g_container(tmp), extra=["--no-spec"])
            try:
                stream, _ = self.gen(ref, temperature=0.0)
            finally:
                ref.close()
            self.assertGreaterEqual(len(stream), 3)
            cont = m8g_container(os.path.join(tmp, "eos"))
            with open(os.path.join(cont, "generation_config.json"),
                      "w") as f:
                json.dump({"eos_token_id": [stream[2]]}, f)
            r = self.both_modes(cont, temperature=0.0)
            self.check_equiv(r, expect_tokens=2, finish="stop")
            self.assertEqual(r["spec"][0], stream[:2])

    def test_implicit_spec_fallback_nomtp(self):
        # The m7a parrot has NO MTP block: the DEFAULT-ON spec (no flags)
        # must fall back to non-spec with a stderr note and serve normally.
        p = Pipe(os.path.join(FIX, "model_chat"), capture_stderr=True)
        try:
            ev = p.rpc({"id": 1, "cmd": "generate",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4, "temperature": 0})
            self.assertEqual(ev[-1]["type"], "done")
            self.assertEqual(ev[-1]["completion_tokens"], 4)
        finally:
            err = p.close()
        self.assertIn("speculative decoding off", err)

    def test_explicit_spec_nomtp_fails(self):
        # An EXPLICIT --spec on an MTP-less container is a loud startup
        # error (the run-mode nomtp negative gate's serve twin).
        proc = subprocess.run(
            [APUS, "serve", "--model", os.path.join(FIX, "model_chat"),
             "--spec"],
            input='{"id":1,"cmd":"generate","ids":[1,2,3],"max_tokens":1}\n',
            capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("requires the MTP", proc.stderr)


# ---------------------------------------------------------------------------
# 2. gateway parser unit tests (GLM <tool_call> + ThinkSplitter)
# ---------------------------------------------------------------------------


def tc(text):
    return (f"<tool_call>get_weather<arg_key>location</arg_key>"
            f"<arg_value>{text}</arg_value></tool_call>")


class ParserUnit(unittest.TestCase):
    def test_plain_thinking(self):
        r, c, tcs = gw.parse_completion("some reasoning</think>the answer")
        self.assertEqual((r, c, tcs), ("some reasoning", "the answer", []))

    def test_unterminated_think_is_reasoning(self):
        r, c, tcs = gw.parse_completion("never closed")
        self.assertEqual((r, c, tcs), ("never closed", "", []))

    def test_empty_reasoning(self):
        r, c, tcs = gw.parse_completion("</think>content only")
        self.assertEqual((r, c, tcs), ("", "content only", []))

    def test_single_tool_call(self):
        r, c, tcs = gw.parse_completion("let me check</think>" + TOOL_CALL_TEXT)
        self.assertEqual(r, "let me check")
        self.assertEqual(c, "")
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]),
                         {"location": "Beijing"})

    def test_tool_call_with_content(self):
        r, c, tcs = gw.parse_completion(
            "r</think>checking first" + TOOL_CALL_TEXT)
        self.assertEqual((r, c), ("r", "checking first"))
        self.assertEqual(len(tcs), 1)

    def test_multiple_tool_calls_and_typed_args(self):
        text = ("</think>"
                "<tool_call>search<arg_key>query</arg_key>"
                "<arg_value>apus m7a</arg_value>"
                "<arg_key>num_results</arg_key><arg_value>5</arg_value>"
                "<arg_key>verbose</arg_key><arg_value>false</arg_value>"
                "</tool_call>"
                "<tool_call>get_weather<arg_key>location</arg_key>"
                "<arg_value>上海</arg_value></tool_call>")
        _, _, tcs = gw.parse_completion(text)
        self.assertEqual(len(tcs), 2)
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]),
                         {"query": "apus m7a", "num_results": 5,
                          "verbose": False})
        self.assertEqual(json.loads(tcs[1]["function"]["arguments"]),
                         {"location": "上海"})

    def test_tool_calls_openai_shape(self):
        _, content, tcs = gw.parse_completion("</think>" + TOOL_CALL_TEXT)
        self.assertEqual(content, "")
        self.assertEqual(len(tcs), 1)
        tc0 = tcs[0]
        self.assertTrue(tc0["id"].startswith("call_"))
        self.assertEqual(tc0["type"], "function")
        self.assertIsInstance(tc0["function"]["arguments"], str)

    def test_tolerant_tails(self):
        # malformed tool-call markup falls back to plain content
        bad = "<tool_call>get_weather<arg_key>oops"
        r, c, tcs = gw.parse_completion("x</think>" + bad)
        self.assertEqual((r, c, tcs), ("x", bad, []))
        # trailing text after a tool call block: also plain content
        r, c, tcs = gw.parse_completion("x</think>" + TOOL_CALL_TEXT + " tail")
        self.assertEqual(tcs, [])
        self.assertIn("tail", c)

    def test_think_splitter(self):
        text = THINKING_REPLY + "</think>" + CHAT_REPLY
        # feed character by character: the split must be exact
        sp = gw.ThinkSplitter(thinking=True)
        parts = []
        for ch in text:
            parts.extend(sp.feed(ch))
        parts.extend(sp.flush())
        reasoning = "".join(t for k, t in parts if k == "reasoning_content")
        content = "".join(t for k, t in parts if k == "content")
        self.assertEqual(reasoning, THINKING_REPLY)
        self.assertEqual(content, CHAT_REPLY)


# ---------------------------------------------------------------------------
# 3. HTTP gateway tests
# ---------------------------------------------------------------------------


class GatewayCase(unittest.TestCase):
    """Base: spawns tools/server.py on a fixture model."""
    MODEL_DIR = None
    MODEL_ID = "test-model"
    EXTRA_ARGS = ()
    EXTRA_ENV = ()

    @classmethod
    def setUpClass(cls):
        ensure_fixtures()
        cls.port = free_port()
        env = dict(os.environ)
        env.pop("APUS_API_KEY", None)
        env.update(dict(cls.EXTRA_ENV))
        # Capture gateway stderr so a CI startup failure is diagnosable
        # (was DEVNULL: the failure mode is invisible otherwise).
        import tempfile
        cls._errlog = tempfile.NamedTemporaryFile(
            mode="rb", prefix="apus-gw-", suffix=".log", delete=False)
        cls.proc = subprocess.Popen(
            [sys.executable, SERVER, "--model", cls.MODEL_DIR,
             "--apus", APUS, "--port", str(cls.port),
             "--model-id", cls.MODEL_ID, *cls.EXTRA_ARGS],
            cwd=ROOT, env=env, stderr=cls._errlog)
        deadline = 120  # shared CI runners can be very slow to boot
        import time
        t0 = time.time()
        while time.time() - t0 < deadline:
            try:
                st, body = cls.get("/health", auth=False)
                if st == 200 and body.get("engine") == "up":
                    return
            except OSError:
                pass
            time.sleep(0.2)
        cls.proc.kill()
        cls._errlog.close()
        with open(cls._errlog.name, "rb") as f:
            tail = f.read()[-2000:].decode("utf-8", "replace")
        raise AssertionError(
            f"gateway did not come up within {deadline}s\n"
            f"--- gateway stderr tail ---\n{tail}")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=30)
        cls.proc.stderr.close() if cls.proc.stderr else None

    @classmethod
    def _conn(cls):
        return http.client.HTTPConnection("127.0.0.1", cls.port, timeout=120)

    @classmethod
    def get(cls, path, auth=True):
        headers = {}
        if auth and os.environ.get("APUS_API_KEY_TEST"):
            headers["Authorization"] = ("Bearer "
                                        + os.environ["APUS_API_KEY_TEST"])
        c = cls._conn()
        c.request("GET", path, headers=headers)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body)

    @classmethod
    def post(cls, path, payload=None, raw=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth and os.environ.get("APUS_API_KEY_TEST"):
            headers["Authorization"] = ("Bearer "
                                        + os.environ["APUS_API_KEY_TEST"])
        c = cls._conn()
        c.request("POST", path,
                  body=raw if raw is not None else json.dumps(payload),
                  headers=headers)
        r = c.getresponse()
        body = r.read()
        status = r.status
        c.close()
        try:
            return status, json.loads(body)
        except ValueError:
            return status, body

    @classmethod
    def post_sse(cls, path, payload):
        """Returns (status, [parsed data: payloads]); "[DONE]" kept as str."""
        c = cls._conn()
        c.request("POST", path, body=json.dumps(payload),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read().decode("utf-8")
        status = r.status
        c.close()
        events = []
        for block in raw.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    data = line[6:]
                    events.append(data if data == "[DONE]"
                                  else json.loads(data))
        return status, events

    def chat(self, messages, **kw):
        payload = {"model": self.MODEL_ID, "messages": messages,
                   "temperature": 0}
        payload.update(kw)
        return self.post("/v1/chat/completions", payload)


class GatewayChat(GatewayCase):
    MODEL_DIR = os.path.join(FIX, "model_chat")
    MODEL_ID = "test-chat"

    def test_health_and_models(self):
        st, body = self.get("/health")
        self.assertEqual(st, 200)
        self.assertEqual(body["status"], "ok")
        st, body = self.get("/v1/models")
        self.assertEqual(st, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], self.MODEL_ID)

    def test_chat_reasoning_split(self):
        st, r = self.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(st, 200)
        self.assertEqual(r["object"], "chat.completion")
        self.assertTrue(r["id"].startswith("chatcmpl-"))
        self.assertEqual(r["model"], self.MODEL_ID)
        ch = r["choices"][0]
        self.assertEqual(ch["index"], 0)
        self.assertEqual(ch["finish_reason"], "stop")
        msg = ch["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["reasoning_content"], THINKING_REPLY)
        self.assertEqual(msg["content"], CHAT_REPLY)
        self.assertNotIn("tool_calls", msg)
        u = r["usage"]
        self.assertEqual(u["completion_tokens"], 5)
        self.assertEqual(u["total_tokens"],
                         u["prompt_tokens"] + u["completion_tokens"])

    def test_usage_prompt_tokens_match_encoding(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "count my tokens"}]
        st, enc = self.post("/debug/encode", {"messages": msgs})
        self.assertEqual(st, 200)
        st, r = self.chat(msgs, max_tokens=1)
        self.assertEqual(r["usage"]["prompt_tokens"], len(enc["ids"]))
        self.assertEqual(r["usage"]["completion_tokens"], 1)
        self.assertEqual(r["choices"][0]["finish_reason"], "length")

    def test_stop_strings(self):
        st, r = self.chat([{"role": "user", "content": "hi"}],
                          stop=["STOP"])
        self.assertEqual(st, 200)
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "stop")
        self.assertEqual(ch["message"]["content"], "The answer is ")
        # stop as a bare string is accepted too
        st, r = self.chat([{"role": "user", "content": "hi"}], stop="STOP")
        self.assertEqual(ch["message"]["content"], "The answer is ")
        self.assertEqual(st, 200)

    def test_seed_determinism(self):
        msgs = [{"role": "user", "content": "hi"}]
        st, a = self.chat(msgs, temperature=0.7, top_p=0.9, seed=1234)
        st, b = self.chat(msgs, temperature=0.7, top_p=0.9, seed=1234)
        self.assertEqual(a["choices"][0]["message"],
                         b["choices"][0]["message"])

    def test_stream_sse(self):
        st, events = self.post_sse("/v1/chat/completions", {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}})
        self.assertEqual(st, 200)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        for c in chunks:
            self.assertEqual(c["object"], "chat.completion.chunk")
            self.assertTrue(c["id"].startswith("chatcmpl-"))
            self.assertEqual(c["model"], self.MODEL_ID)
        # first chunk: role delta; then deltas; final: empty delta + reason
        self.assertEqual(chunks[0]["choices"][0]["delta"],
                         {"role": "assistant"})
        self.assertEqual(chunks[0]["choices"][0]["finish_reason"], None)
        # locate the finish chunk: empty delta with finish_reason
        finish_chunks = [c for c in chunks
                         if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finish_chunks), 1)
        self.assertEqual(finish_chunks[0]["choices"][0]["finish_reason"],
                         "stop")
        self.assertEqual(finish_chunks[0]["choices"][0]["delta"], {})
        # reassembled stream == non-stream message
        reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "")
                            for c in chunks if c["choices"])
        content = "".join(c["choices"][0]["delta"].get("content", "")
                          for c in chunks if c["choices"])
        self.assertEqual(reasoning, THINKING_REPLY)
        self.assertEqual(content, CHAT_REPLY)
        # usage chunk present (include_usage) with empty choices
        usage_chunks = [c for c in chunks if c.get("usage")]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["choices"], [])
        self.assertEqual(usage_chunks[0]["usage"]["completion_tokens"], 5)
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"],
                         usage_chunks[0]["usage"]["prompt_tokens"] + 5)

    def test_stream_order_before_done(self):
        st, events = self.post_sse("/v1/chat/completions", {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "stream": True})
        kinds = []
        for e in events:
            if e == "[DONE]":
                kinds.append("done")
            elif e["choices"] and e["choices"][0]["finish_reason"]:
                kinds.append("finish")
            else:
                kinds.append("delta")
        self.assertEqual(kinds[0], "delta")
        self.assertEqual(kinds[-2:], ["finish", "done"])
        self.assertNotIn("finish", kinds[:-2])

    def test_concurrent_requests_serialize(self):
        # 6 concurrent requests: single engine -> serialized, all correct
        results = [None] * 6

        def work(i):
            results[i] = self.chat([{"role": "user",
                                     "content": f"request {i}"}])
        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for st, r in results:
            self.assertEqual(st, 200)
            self.assertEqual(r["choices"][0]["message"]["content"], CHAT_REPLY)

    def test_errors(self):
        # malformed JSON -> 400 with the OpenAI error shape
        st, r = self.post("/v1/chat/completions", raw="{not json")
        self.assertEqual(st, 400)
        self.assertIn("error", r)
        self.assertEqual(r["error"]["type"], "invalid_request_error")
        self.assertIn("message", r["error"])
        # unknown model -> 404
        st, r = self.chat([{"role": "user", "content": "hi"}],
                          model="no-such-model")
        self.assertEqual(st, 404)
        self.assertEqual(r["error"]["type"], "not_found_error")
        # unknown path -> 404
        st, r = self.post("/v1/engines", {})
        self.assertEqual(st, 404)
        st, r = self.get("/v1/unknown")
        self.assertEqual(st, 404)
        # bad messages -> 400
        st, r = self.post("/v1/chat/completions",
                          {"model": self.MODEL_ID, "messages": "hi"})
        self.assertEqual(st, 400)

    def test_encode_failure_is_400(self):
        # glm_encode_* failures surface as HTTP 400 (the M2 handoff):
        # dict-typed message content is an encoding error in c/encoding.h
        st, r = self.chat([{"role": "user", "content": {"x": 1}}])
        self.assertEqual(st, 400)
        self.assertEqual(r["error"]["type"], "invalid_request_error")
        # same through /debug/encode
        st, r = self.post("/debug/encode",
                          {"messages": [{"role": "user",
                                         "content": {"x": 1}}]})
        self.assertEqual(st, 400)
        self.assertEqual(r["error"]["type"], "invalid_request_error")
        # string tool_call arguments fail too (template UndefinedError
        # semantics, c/encoding.h)
        st, r = self.post("/debug/encode", {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f",
                                          "arguments": "not-an-object"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "x"}]})
        self.assertEqual(st, 400)
        # the gateway survives: a good request still works
        st, r = self.chat([{"role": "user", "content": "hi"}], max_tokens=1)
        self.assertEqual(st, 200)

    def test_completions(self):
        st, r = self.post("/v1/completions", {
            "model": self.MODEL_ID, "prompt": "once upon a <think>",
            "max_tokens": 8, "temperature": 0})
        self.assertEqual(st, 200)
        self.assertEqual(r["object"], "text_completion")
        ch = r["choices"][0]
        self.assertEqual(ch["index"], 0)
        self.assertIsInstance(ch["text"], str)
        self.assertIn(ch["finish_reason"], ("stop", "length"))
        u = r["usage"]
        self.assertGreater(u["prompt_tokens"], 0)
        self.assertLessEqual(u["completion_tokens"], 8)
        # raw prompts ending in <think> fire the parrot chain
        self.assertTrue(ch["text"].startswith(THINKING_REPLY))
        # streaming variant
        st, events = self.post_sse("/v1/completions", {
            "model": self.MODEL_ID, "prompt": "once upon a <think>",
            "max_tokens": 4, "temperature": 0, "stream": True})
        self.assertEqual(st, 200)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        self.assertTrue(all(c["object"] == "text_completion" for c in chunks))
        streamed = "".join(c["choices"][0]["text"] for c in chunks)
        finish = [c for c in chunks if c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finish), 1)
        self.assertGreater(len(streamed), 0)

    def test_gateway_encode_matches_engine_pipe(self):
        # the gateway must not alter the messages: text+ids through
        # /debug/encode == direct pipe encode (same encoding.h path)
        msgs = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer",
                 "reasoning_content": "secret thoughts"},
                {"role": "user", "content": "second question"}]
        st, via_http = self.post("/debug/encode", {"messages": msgs})
        self.assertEqual(st, 200)
        pipe = Pipe(self.MODEL_DIR)
        try:
            direct = pipe.rpc({"id": 1, "cmd": "encode", "messages": msgs})[0]
        finally:
            pipe.close()
        self.assertEqual(via_http["text"], direct["text"])
        self.assertEqual(via_http["ids"], direct["ids"])
        # byte-for-byte vs the jinja2 golden: multi-turn rendering keeps
        # the earlier reasoning (clear_thinking unset)
        self.assertEqual(via_http["text"], render(msgs))
        self.assertIn("secret thoughts", via_http["text"])

    def test_clear_thinking_drops_past_reasoning(self):
        msgs = [{"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer",
                 "reasoning_content": "secret thoughts"},
                {"role": "user", "content": "second question"}]
        st, r = self.post("/debug/encode",
                          {"messages": msgs, "clear_thinking": True})
        self.assertEqual(st, 200)
        self.assertEqual(r["text"], render(msgs, clear_thinking=True))
        self.assertNotIn("secret thoughts", r["text"])
        self.assertIn("<think></think>", r["text"])
        # chat_template_kwargs spelling works too
        st, r2 = self.post("/debug/encode",
                           {"messages": msgs,
                            "chat_template_kwargs": {"clear_thinking": True}})
        self.assertEqual(r2["text"], r["text"])

    def test_reasoning_effort_rendering(self):
        msgs = [{"role": "user", "content": "hi"}]
        for effort, word in (("low", "Low"), ("high", "High"),
                             ("max", "Max"), ("bogus", "Max")):
            st, r = self.post("/debug/encode",
                              {"messages": msgs,
                               "reasoning_effort": effort})
            self.assertEqual(st, 200)
            self.assertEqual(r["text"],
                             render(msgs, reasoning_effort=effort))
            self.assertIn(f"Reasoning Effort: {word}", r["text"])


class GatewayTools(GatewayCase):
    MODEL_DIR = os.path.join(FIX, "model_tools")
    MODEL_ID = "test-tools"

    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }]

    def test_tools_render_as_sibling(self):
        # tools ride as a top-level template parameter (never attached to
        # a message): byte-for-byte vs the jinja2 golden
        msgs = [{"role": "user", "content": "Weather in Beijing?"}]
        st, r = self.post("/debug/encode",
                          {"messages": msgs, "tools": self.TOOLS})
        self.assertEqual(st, 200)
        self.assertEqual(r["text"], render(msgs, tools=self.TOOLS))
        self.assertIn("# Tools", r["text"])
        self.assertIn('"get_weather"', r["text"])

    def test_tool_call_roundtrip(self):
        st, r = self.chat(
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "Weather in Beijing?"}],
            tools=self.TOOLS)
        self.assertEqual(st, 200)
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        msg = ch["message"]
        self.assertEqual(msg["reasoning_content"], TOOL_REPLY)
        self.assertIsNone(msg["content"])
        tcs = msg["tool_calls"]
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["type"], "function")
        self.assertTrue(tcs[0]["id"].startswith("call_"))
        self.assertEqual(tcs[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]),
                         {"location": "Beijing"})
        self.assertEqual(r["usage"]["completion_tokens"], 3)

    def test_tool_call_streaming_finish_reason(self):
        st, events = self.post_sse("/v1/chat/completions", {
            "model": self.MODEL_ID,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": self.TOOLS, "temperature": 0, "stream": True})
        self.assertEqual(st, 200)
        self.assertEqual(events[-1], "[DONE]")
        finish = [e for e in events if isinstance(e, dict) and e["choices"]
                  and e["choices"][0]["finish_reason"]]
        self.assertEqual(finish[0]["choices"][0]["finish_reason"],
                         "tool_calls")

    def test_tool_result_followup_rendering(self):
        # follow-up request with the assistant's tool_calls and a role=tool
        # message must render the GLM observation form, byte-for-byte vs
        # the jinja2 golden
        tool_call_id = "call_abc123"
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Beijing?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": tool_call_id, "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"location": "Beijing"}'}}]},
            {"role": "tool", "tool_call_id": tool_call_id,
             "content": "18°C and sunny"},
            {"role": "user", "content": "thanks, and tomorrow?"},
        ]
        st, r = self.post("/debug/encode",
                          {"messages": msgs, "tools": self.TOOLS})
        self.assertEqual(st, 200)
        text = r["text"]
        self.assertIn("<tool_call>get_weather<arg_key>location</arg_key>"
                      "<arg_value>Beijing</arg_value></tool_call>", text)
        self.assertIn("<|observation|>", text)
        self.assertIn("<tool_response>18°C and sunny</tool_response>", text)
        self.assertTrue(text.endswith("<|assistant|><think>"))
        # the gateway normalizes OpenAI string arguments to objects (the
        # GLM template shape); the golden renders the normalized messages
        self.assertEqual(text, render(gw.normalize_tool_call_arguments(msgs),
                                      tools=self.TOOLS))


class GatewayAuth(GatewayCase):
    MODEL_DIR = os.path.join(FIX, "model_chat")
    MODEL_ID = "test-auth"
    EXTRA_ENV = (("APUS_API_KEY", "m7a-secret"),)

    @classmethod
    def setUpClass(cls):
        os.environ["APUS_API_KEY_TEST"] = "m7a-secret"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("APUS_API_KEY_TEST", None)

    def test_api_key_required(self):
        st, r = self.get("/v1/models", auth=False)
        self.assertEqual(st, 401)
        self.assertEqual(r["error"]["type"], "authentication_error")
        st, r = self.chat([{"role": "user", "content": "hi"}], )
        self.assertEqual(st, 200)
        # health stays open
        st, r = self.get("/health", auth=False)
        self.assertEqual(st, 200)


# ---------------------------------------------------------------------------
# 4. CLI end-to-end (bin/apus run) — the oracle token gate on the run path
# ---------------------------------------------------------------------------


class CliEndToEnd(unittest.TestCase):
    MODEL = os.path.join(FIX, "model_chat")

    @classmethod
    def setUpClass(cls):
        ensure_fixtures()
        cls.stream = golden()["variants"]["model_chat"]["oracle_stream"]

    def run_cli(self, extra, env=None):
        e = dict(os.environ)
        if env:
            e.update(env)
        r = subprocess.run(
            [APUS, "run", "--model", self.MODEL, "--greedy",
             "--max-tokens", "16", "--quiet"] + extra,
            capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        return [int(x) for x in r.stdout.split()]

    def test_run_ids_matches_oracle(self):
        # the run path prints the terminating EOS id (unlike serve, which
        # never emits EOS) — the full oracle stream, bitwise.
        ids = self.run_cli(["--ids", str(THINK)])
        self.assertEqual(ids, self.stream)

    def test_run_prompt_matches_oracle(self):
        # --prompt: GLM chat template + tokenizer + prefill + decode
        ids = self.run_cli(["--prompt", "hi"])
        self.assertEqual(ids, self.stream)

    def test_run_tiered_matches_oracle(self):
        # --tiered with a 1 MB cache (evictions forced) + pilot attached:
        # identical tokens (eager == tiered == pilot, end to end).
        ids = self.run_cli(["--prompt", "hi", "--tiered"],
                           env={"APUS_GEXPERT_CACHE_MB": "1"})
        self.assertEqual(ids, self.stream)


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
