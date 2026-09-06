#!/usr/bin/env python3
"""apus OpenAI-compatible serving gateway (M7a, GLM-5.3-Flash). Python
STDLIB ONLY.

Follows the colibri split: this process owns HTTP/JSON/SSE; the C engine
(`bin/apus serve --model DIR`) is spawned as a subprocess and driven over
stdin/stdout with the NDJSON protocol documented in tests/m7a/README.md.
The model loads once (engine stays up); every request re-prefills (no KV
reuse across turns yet).

Endpoints:
  GET  /health                    -> {"status": "ok", ...}
  GET  /v1/models                 -> OpenAI model list (single model)
  POST /v1/chat/completions       -> chat.completion (or SSE chunks when
                                     "stream": true, ending in data: [DONE])
  POST /v1/completions            -> text_completion (stream supported)
  POST /debug/encode              -> NON-STANDARD test endpoint: the exact
                                     prompt text + token ids the engine
                                     renders for a message list.

Request fields honored (chat): messages, tools (a SIBLING of messages —
the GLM template takes them as a top-level parameter, never attached to a
message), temperature, top_p, max_tokens (or max_completion_tokens), seed,
stream, stream_options.include_usage, stop (str | [str]), model,
reasoning_effort ("low" | "high"; top-level or in chat_template_kwargs —
everything else maps to "Max"), clear_thinking (bool; top-level or in
chat_template_kwargs, default false — when true the template drops
reasoning of assistant messages up to the last user message). max_tokens
omitted = generate until EOS (OpenAI semantics; the engine sizes the
per-request KV from APUS_GSERVE_MAX_TOKENS, default 65536). The GLM
template has NO thinking on/off switch: generation always starts <think>;
responses expose reasoning as reasoning_content (DeepSeek API style).
Unknown fields are ignored.

Concurrency: ONE engine, so requests are serialized through a single lock
— concurrent HTTP requests queue in lock-acquisition order and are
processed one at a time. No interleaving, no batching.

Auth: if env APUS_API_KEY is set, /v1/* and /debug/* require
`Authorization: Bearer $APUS_API_KEY` (401 otherwise); /health stays open.
Off by default.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THINK_END = "</think>"
TC_START = "<tool_call>"
TC_END = "</tool_call>"
EOS_STR = "<|endoftext|>"

# ---------------------------------------------------------------------------
# GLM tool-call parsing (response side). The model emits, after its
# "</think>", optional content followed by consecutive
#   <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
# blocks (reference/chat_template.jinja). Argument values: strings render
# verbatim, non-strings as json.dumps — so a value that json.loads parses
# is recovered to its typed form, otherwise kept as a string (documented
# heuristic: a string argument that itself looks like JSON, e.g. "123",
# is mistyped; the wire format is ambiguous there). Tolerant: EOS is not
# required, unterminated thinking -> all reasoning_content, malformed
# tool-call markup -> plain content.
# ---------------------------------------------------------------------------

_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>",
                     re.DOTALL)


def parse_glm_tool_calls(text):
    """Parse consecutive <tool_call> blocks at the start of `text`.
    Returns OpenAI-shaped tool_calls; raises ValueError on malformed
    markup (caller decides how to degrade)."""
    calls = []
    pos = 0
    while text.startswith(TC_START, pos):
        end = text.find(TC_END, pos)
        if end == -1:
            raise ValueError("unterminated tool_call block")
        body = text[pos + len(TC_START):end]
        i = body.find("<arg_key>")
        name = body if i == -1 else body[:i]
        if not name:
            raise ValueError("tool_call without a name")
        args = {}
        for k, v in _ARG_RE.findall(body[i:] if i != -1 else ""):
            if k in args:
                raise ValueError(f"duplicate argument {k!r}")
            try:
                args[k] = json.loads(v)
            except ValueError:
                args[k] = v
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
        pos = end + len(TC_END)
    if not calls:
        raise ValueError("no tool_call blocks")
    if text[pos:].strip():
        raise ValueError("trailing content after tool_call blocks")
    return calls


def normalize_tool_call_arguments(messages):
    """OpenAI round-trip compatibility: assistant tool_calls on the wire
    carry function.arguments as a JSON STRING, but the GLM chat template
    requires an object (it iterates .items(); a string raises, and
    c/encoding.h mirrors that as an encode error -> HTTP 400). Parse
    string arguments back into objects when possible so clients can feed
    our own responses back as history; unparseable strings pass through
    and fail at the engine with a 400. Returns a new message list."""
    out = []
    for m in messages:
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if not tcs:
            out.append(m)
            continue
        m = dict(m)
        m["tool_calls"] = [
            dict(tc, function=dict(
                tc["function"],
                arguments=_arguments_as_object(tc["function"].get(
                    "arguments"))))
            if isinstance(tc, dict)
            and isinstance(tc.get("function"), dict)
            and isinstance(tc["function"].get("arguments"), str)
            else tc
            for tc in tcs]
        out.append(m)
    return out


def _arguments_as_object(args):
    try:
        v = json.loads(args)
        return v if isinstance(v, dict) else args
    except (ValueError, TypeError):
        return args


def parse_completion(text, thinking=True):
    """Split a raw completion into (reasoning_content, content, tool_calls)."""
    # the engine never emits EOS, but cut at it if present
    i = text.find(EOS_STR)
    if i >= 0:
        text = text[:i]
    reasoning, content, tool_calls = "", "", []
    rest = text
    if thinking:
        i = rest.find(THINK_END)
        if i < 0:
            return text, "", []          # unterminated: all reasoning
        reasoning, rest = rest[:i], rest[i + len(THINK_END):]
    i = rest.find(TC_START)
    if i < 0:
        return reasoning, rest, []
    try:
        tool_calls = parse_glm_tool_calls(rest[i:])
        content = rest[:i]
    except ValueError:
        content, tool_calls = rest, []   # malformed: plain content
    return reasoning, content, tool_calls


class ThinkSplitter:
    """Incremental thinking/content split for SSE streaming. Reasoning is
    buffered until "</think>" is seen unambiguously; after that, text is
    forwarded as content deltas raw (<tool_call> markup included — the
    parsed structure is a non-streaming feature; see tests/m7a/README)."""

    _KEEP = len(THINK_END) - 1

    def __init__(self, thinking):
        self.closed = not thinking
        self.pending = ""

    def feed(self, piece):
        out = []
        s = self.pending + piece
        self.pending = ""
        if not self.closed:
            i = s.find(THINK_END)
            if i >= 0:
                if s[:i]:
                    out.append(("reasoning_content", s[:i]))
                self.closed = True
                s = s[i + len(THINK_END):]
            else:
                if len(s) > self._KEEP:
                    out.append(("reasoning_content", s[:-self._KEEP]))
                    s = s[-self._KEEP:]
                self.pending = s
                return out
        if s:
            out.append(("content", s))
        return out

    def flush(self):
        if not self.pending:
            return []
        kind = "content" if self.closed else "reasoning_content"
        p, self.pending = self.pending, ""
        return [(kind, p)]


# ---------------------------------------------------------------------------
# Engine subprocess (single instance, serialized access)
# ---------------------------------------------------------------------------


class EngineError(Exception):
    pass


class ApusEngine:
    def __init__(self, apus_bin, model_dir, tiered=False):
        cmd = [apus_bin, "serve", "--model", model_dir]
        if tiered:
            cmd.append("--tiered")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.lock = threading.Lock()
        self._next_id = 0

    def alive(self):
        return self.proc.poll() is None

    def request(self, payload):
        """Send one request; yield engine events up to and including the
        terminal one (done/error/encoded). Holds the engine lock for the
        whole exchange: one request in flight, others queue. If the caller
        abandons the generator (client disconnect), the remaining events
        are drained so the protocol stays in sync."""
        with self.lock:
            self._next_id += 1
            payload = dict(payload, id=self._next_id)
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise EngineError(f"engine not writable: {e}")
            terminal = False
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise EngineError("engine exited mid-request")
                    ev = json.loads(line)
                    if ev.get("type") in ("done", "error", "encoded"):
                        terminal = True
                    yield ev
                    if terminal:
                        return
            finally:
                if not terminal:       # abandoned: drain to the sentinel
                    while True:
                        line = self.proc.stdout.readline()
                        if not line:
                            break
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if ev.get("type") in ("done", "error", "encoded"):
                            break

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# HTTP gateway
# ---------------------------------------------------------------------------


def openai_error(message, err_type="invalid_request_error", param=None,
                 code=None):
    return {"error": {"message": message, "type": err_type,
                      "param": param, "code": code}}


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, engine, model_id, default_clear_thinking=False):
        super().__init__(addr, Handler)
        self.engine = engine
        self.model_id = model_id
        self.default_clear_thinking = default_clear_thinking
        self.api_key = os.environ.get("APUS_API_KEY") or None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "apus-m7a"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        sys.stderr.write("gateway: " + fmt % args + "\n")

    def _send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, message, err_type="invalid_request_error"):
        self._send_json(status, openai_error(message, err_type))

    def _auth_ok(self):
        key = self.server.api_key
        if not key:
            return True
        if self.headers.get("Authorization") == f"Bearer {key}":
            return True
        self._send_error(401, "missing or invalid API key", "authentication_error")
        return False

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > 64 * 1024 * 1024:
            return None
        return self.rfile.read(n)

    # -- GET ---------------------------------------------------------------

    def do_GET(self):
        if self.path == "/health":
            eng = "up" if self.server.engine.alive() else "down"
            self._send_json(200 if eng == "up" else 503,
                            {"status": "ok" if eng == "up" else "degraded",
                             "model": self.server.model_id, "engine": eng})
            return
        if self.path == "/v1/models":
            if not self._auth_ok():
                return
            self._send_json(200, {"object": "list", "data": [{
                "id": self.server.model_id, "object": "model",
                "created": 0, "owned_by": "apus"}]})
            return
        self._send_error(404, f"unknown path: {self.path}", "not_found_error")

    # -- POST --------------------------------------------------------------

    def do_POST(self):
        if not self._auth_ok():
            return
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else None
        except ValueError as e:
            self._send_error(400, f"malformed JSON body: {e}")
            return
        if not isinstance(body, dict):
            self._send_error(400, "request body must be a JSON object")
            return
        if self.path == "/v1/chat/completions":
            self._chat_completions(body)
        elif self.path == "/v1/completions":
            self._completions(body)
        elif self.path == "/debug/encode":
            self._debug_encode(body)
        else:
            self._send_error(404, f"unknown path: {self.path}", "not_found_error")

    # -- shared request translation ---------------------------------------

    def _check_model(self, body):
        m = body.get("model")
        if m is not None and m != self.server.model_id:
            self._send_error(404, f"unknown model: {m!r}", "not_found_error")
            return False
        return True

    def _sampling_params(self, body):
        req = {}
        mt = body.get("max_tokens", body.get("max_completion_tokens"))
        if mt is not None:
            req["max_tokens"] = int(mt)
        for k in ("temperature", "top_p", "seed"):
            if body.get(k) is not None:
                req[k] = body[k]
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
            req["stop"] = stop
        return req

    def _enc_opts(self, body):
        """GLM encoding options: clear_thinking (default from
        --clear-thinking / APUS_CLEAR_THINKING, off by default) and
        reasoning_effort ("low"|"high"; anything else renders "Max")."""
        kwargs = body.get("chat_template_kwargs") or {}
        clear = body.get("clear_thinking",
                         kwargs.get("clear_thinking",
                                    self.server.default_clear_thinking))
        effort = body.get("reasoning_effort",
                          kwargs.get("reasoning_effort"))
        return bool(clear), effort

    def _enc_fields(self, body):
        req = {}
        clear, effort = self._enc_opts(body)
        if clear:
            req["clear_thinking"] = True
        if effort:
            req["reasoning_effort"] = effort
        return req

    # -- /v1/chat/completions ----------------------------------------------

    def _chat_completions(self, body):
        if not self._check_model(body):
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not all(
                isinstance(m, dict) for m in messages):
            self._send_error(400, "messages must be an array of objects")
            return
        req = {"cmd": "generate",
               "messages": normalize_tool_call_arguments(messages)}
        req.update(self._enc_fields(body))
        if body.get("tools") is not None:
            req["tools"] = body["tools"]
        req.update(self._sampling_params(body))
        if body.get("stream"):
            self._chat_stream(req, body.get("stream_options") or {})
        else:
            self._chat_once(req)

    def _engine_error_status(self, ev):
        """Engine error event -> (HTTP status, OpenAI error type). The M2
        handoff: glm_encode_* failures surface as 400s."""
        if ev.get("code") == "bad_request":
            return 400, "invalid_request_error"
        return 500, "engine_error"

    def _chat_once(self, req):
        try:
            done = None
            for ev in self.server.engine.request(req):
                if ev.get("type") == "done":
                    done = ev
                elif ev.get("type") == "error":
                    st, ty = self._engine_error_status(ev)
                    self._send_error(st, ev.get("message", "engine error"), ty)
                    return
        except EngineError as e:
            self._send_error(503, str(e), "engine_error")
            return
        reasoning, content, tool_calls = parse_completion(done.get("text", ""))
        message = {"role": "assistant"}
        if reasoning:
            message["reasoning_content"] = reasoning
        message["content"] = content if (content or not tool_calls) else None
        if tool_calls:
            message["tool_calls"] = tool_calls
        p, c = done["prompt_tokens"], done["completion_tokens"]
        resp = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self._finish_reason(done, tool_calls),
            }],
            "usage": {"prompt_tokens": p, "completion_tokens": c,
                      "total_tokens": p + c},
        }
        self._send_json(200, resp)

    @staticmethod
    def _finish_reason(done, tool_calls):
        if done["finish_reason"] == "length":
            return "length"
        if tool_calls:
            return "tool_calls"
        return "stop"          # both EOS and stop_string map to "stop"

    def _sse_begin(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse_send(self, obj):
        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False)
                          + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _chat_stream(self, req, stream_opts):
        include_usage = bool(stream_opts.get("include_usage"))
        cmpl_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        def chunk(delta, finish=None):
            return {"id": cmpl_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.server.model_id,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}

        self._sse_begin()
        splitter = ThinkSplitter(thinking=True)   # GLM always thinks
        full = []
        done = None
        try:
            self._sse_send(chunk({"role": "assistant"}))
            for ev in self.server.engine.request(req):
                t = ev.get("type")
                if t == "token":
                    piece = ev.get("text", "")
                    full.append(piece)
                    for kind, text in splitter.feed(piece):
                        self._sse_send(chunk({kind: text}))
                elif t == "done":
                    done = ev
                elif t == "error":
                    err = openai_error(ev.get("message", "engine error"),
                                       "invalid_request_error"
                                       if ev.get("code") == "bad_request"
                                       else "engine_error")
                    self._sse_send(err)
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
            for kind, text in splitter.flush():
                self._sse_send(chunk({kind: text}))
            if done is None:
                self.wfile.write(b"data: [DONE]\n\n")
                return
            reasoning, content, tool_calls = parse_completion("".join(full))
            self._sse_send(chunk({}, self._finish_reason(done, tool_calls)))
            if include_usage:
                p, c = done["prompt_tokens"], done["completion_tokens"]
                self._sse_send({"id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.server.model_id,
                                "choices": [],
                                "usage": {"prompt_tokens": p,
                                          "completion_tokens": c,
                                          "total_tokens": p + c}})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass               # client left; engine generator drains itself
        except EngineError:
            pass

    # -- /v1/completions ----------------------------------------------------

    def _completions(self, body):
        if not self._check_model(body):
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            self._send_error(400, "prompt must be a string "
                                  "(batched prompts not supported)")
            return
        req = {"cmd": "generate", "text": prompt}
        req.update(self._sampling_params(body))
        if body.get("stream"):
            self._compl_stream(req)
            return
        try:
            done = None
            for ev in self.server.engine.request(req):
                if ev.get("type") == "done":
                    done = ev
                elif ev.get("type") == "error":
                    st, ty = self._engine_error_status(ev)
                    self._send_error(st, ev.get("message", "engine error"), ty)
                    return
        except EngineError as e:
            self._send_error(503, str(e), "engine_error")
            return
        p, c = done["prompt_tokens"], done["completion_tokens"]
        self._send_json(200, {
            "id": "cmpl-" + uuid.uuid4().hex[:24],
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.server.model_id,
            "choices": [{"text": done.get("text", ""), "index": 0,
                         "finish_reason": ("length" if done["finish_reason"]
                                           == "length" else "stop")}],
            "usage": {"prompt_tokens": p, "completion_tokens": c,
                      "total_tokens": p + c},
        })

    def _compl_stream(self, req):
        cmpl_id = "cmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        self._sse_begin()
        done = None
        try:
            for ev in self.server.engine.request(req):
                t = ev.get("type")
                if t == "token":
                    self._sse_send({
                        "id": cmpl_id, "object": "text_completion",
                        "created": created, "model": self.server.model_id,
                        "choices": [{"text": ev.get("text", ""), "index": 0,
                                     "finish_reason": None}]})
                elif t == "done":
                    done = ev
                elif t == "error":
                    self._sse_send(openai_error(ev.get("message", "engine error"),
                                                "engine_error"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
            fr = "length" if done["finish_reason"] == "length" else "stop"
            self._sse_send({"id": cmpl_id, "object": "text_completion",
                            "created": created, "model": self.server.model_id,
                            "choices": [{"text": "", "index": 0,
                                         "finish_reason": fr}]})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError, EngineError):
            pass

    # -- /debug/encode (non-standard; test/conformance endpoint) ------------

    def _debug_encode(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list):
            self._send_error(400, "messages must be an array")
            return
        req = {"cmd": "encode",
               "messages": normalize_tool_call_arguments(messages)}
        if body.get("tools") is not None:
            req["tools"] = body["tools"]
        req.update(self._enc_fields(body))
        try:
            for ev in self.server.engine.request(req):
                if ev.get("type") == "encoded":
                    self._send_json(200, {"text": ev.get("text", ""),
                                          "ids": ev.get("ids", [])})
                    return
                if ev.get("type") == "error":
                    st, ty = self._engine_error_status(ev)
                    self._send_error(st, ev.get("message", "encode error"), ty)
                    return
        except EngineError as e:
            self._send_error(503, str(e), "engine_error")


def main():
    ap = argparse.ArgumentParser(description="apus OpenAI-compatible gateway")
    ap.add_argument("--model", required=True, help="model dir")
    ap.add_argument("--apus", default=os.environ.get("APUS_BIN", "bin/apus"),
                    help="path to the apus binary (env APUS_BIN)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tiered", action="store_true",
                    help="engine expert tiering (M6 cache; GLM budget "
                         "APUS_GEXPERT_CACHE_MB)")
    ap.add_argument("--model-id", default=None,
                    help="served model id (default: model dir basename)")
    ap.add_argument("--clear-thinking", action="store_true",
                    help="default clear_thinking=true (per-request "
                         "clear_thinking / chat_template_kwargs overrides)")
    args = ap.parse_args()

    default_clear = (os.environ.get("APUS_CLEAR_THINKING", "0") != "0"
                     or args.clear_thinking)
    model_id = args.model_id or os.path.basename(
        os.path.normpath(args.model)) or "apus"
    engine = ApusEngine(args.apus, args.model, args.tiered)
    srv = Gateway((args.host, args.port), engine, model_id, default_clear)
    sys.stderr.write(f"apus gateway: http://{args.host}:{args.port} "
                     f"model={model_id} engine={' '.join([args.apus, 'serve'])}"
                     f" clear_thinking_default={default_clear}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()


if __name__ == "__main__":
    main()
