#!/usr/bin/env python3
"""apus interactive terminal chat (GLM-5.3-Flash) — colibri `coli chat`
style.

Drives `bin/apus serve` (NDJSON stdio protocol, see tests/m7a/README.md)
directly: no HTTP gateway, no port, no deps beyond the Python stdlib.
The engine loads once; each turn re-prefills the conversation (no KV
reuse yet), so later turns get slower — keep history short or /reset.

Usage:
    python3 tools/chat.py --model weights/apus-glm --tiered
    python3 tools/chat.py --model weights/apus-glm --effort low

MTP speculative decoding (--spec) is ON by default (M8d, K=3 — the
re-pin sweep winner); --no-spec opts out, --spec-k K overrides the
draft depth. Emitted tokens are bitwise identical either way.

In-chat commands:
    /quit              exit (also Ctrl-D / Ctrl-C)
    /reset             clear conversation history
    /system <text>     set/replace the system message
    /clear on|off      clear_thinking (drop past-turn reasoning from the
                       prompt; default on — GLM multi-turn convention)
    /effort low|high|max   reasoning effort (everything but low/high = Max)
    /temp X            sampling temperature (0 = greedy)
    /top_p X           nucleus-sampling top-p (default: the container's
                       generation_config.json, else 1.0)
    /max N             max tokens per reply
    /raw <text>        raw completion mode (no chat template) for this turn
    /help              show commands
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Engine:
    def __init__(self, model, apus, tiered, spec=True, spec_k=None):
        cmd = [apus, "serve", "--model", model]
        if tiered:
            cmd.append("--tiered")
        # M8d: spec explicit (the serve default is ON too, but pass it so
        # an MTP-less container fails loudly at startup instead of the
        # silent fallback — chat targets the real GLM container)
        cmd.append("--spec" if spec else "--no-spec")
        if spec and spec_k is not None:
            cmd += ["--spec-k", str(spec_k)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, encoding="utf-8", errors="replace",
            bufsize=1)
        self.next_id = 0

    def request(self, req):
        self.next_id += 1
        req["id"] = self.next_id
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return self.next_id

    def events(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                yield json.loads(line)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except BaseException:          # timeout AND a second Ctrl-C
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except BaseException:
                pass


def main():
    ap = argparse.ArgumentParser(description="apus terminal chat (GLM)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--apus", default=os.path.join(ROOT, "bin", "apus"))
    ap.add_argument("--tiered", action="store_true")
    ap.add_argument("--effort", default=None,
                    choices=["low", "high", "max"],
                    help="reasoning effort (default: template Max)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus top_p (default: the container's "
                         "generation_config.json via the engine)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spec", action="store_true",
                    help="disable MTP speculative decoding (default: on, "
                         "--spec-k drafts per step)")
    ap.add_argument("--spec-k", type=int, default=None,
                    help="MTP drafts per verify step (default: 3)")
    args = ap.parse_args()

    clear_thinking = True
    effort = args.effort
    temp, max_tokens = args.temp, args.max_tokens
    top_p = args.top_p
    messages = []
    eng = Engine(args.model, args.apus, args.tiered,
                 spec=not args.no_spec, spec_k=args.spec_k)

    print("apus chat — /help for commands, /quit to exit", flush=True)

    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                parts = line.split(None, 1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""
                if cmd == "/quit":
                    break
                elif cmd == "/reset":
                    messages = []
                    print("(history cleared)")
                elif cmd == "/system":
                    messages = [m for m in messages if m.get("role") != "system"]
                    if arg:
                        messages.insert(0, {"role": "system", "content": arg})
                    print("(system message set)" if arg else "(system message removed)")
                elif cmd == "/clear":
                    clear_thinking = arg.strip().lower() not in ("off", "0", "false")
                    print(f"(clear_thinking {'on' if clear_thinking else 'off'})")
                elif cmd == "/effort":
                    effort = arg.strip().lower() or None
                    print(f"(reasoning effort {effort or 'Max'})")
                elif cmd == "/temp":
                    temp = float(arg)
                    print(f"(temperature {temp})")
                elif cmd == "/top_p":
                    if arg.strip().lower() in ("none", "off", "-"):
                        top_p = None
                    else:
                        top_p = float(arg)
                    print(f"(top_p {top_p if top_p is not None else 'engine default'})")
                elif cmd == "/max":
                    max_tokens = int(arg)
                    print(f"(max_tokens {max_tokens})")
                elif cmd == "/raw":
                    if not arg:
                        continue
                    req = {"cmd": "generate", "text": arg,
                           "max_tokens": max_tokens, "temperature": temp,
                           "seed": args.seed}
                    if top_p is not None:
                        req["top_p"] = top_p
                    _stream_turn(eng, req)
                elif cmd == "/help":
                    print(__doc__)
                else:
                    print(f"(unknown command {cmd}; /help)")
                continue

            messages.append({"role": "user", "content": line})
            req = {"cmd": "generate", "messages": messages,
                   "clear_thinking": clear_thinking,
                   "max_tokens": max_tokens,
                   "temperature": temp, "seed": args.seed}
            if top_p is not None:
                req["top_p"] = top_p
            if effort and effort != "max":
                req["reasoning_effort"] = effort
            reply = _stream_turn(eng, req)
            if reply is not None:
                messages.append({"role": "assistant", "content": reply})
    finally:
        eng.close()


def _stream_turn(eng, req):
    """Send one generate request, stream tokens to the terminal.

    Returns the assembled reply text, or None on error. GLM generation
    always starts in <think>: the reasoning (up to </think>) is printed
    dimmed."""
    rid = eng.request(req)
    DIM, RESET = "\033[2m", "\033[0m"
    parts = []
    in_think = True
    printed_any = False
    t0 = time.time()
    print("apus> ", end="", flush=True)
    print(DIM, end="", flush=True)
    done = None
    for ev in eng.events():
        if ev.get("id") != rid:
            continue
        t = ev.get("type")
        if t == "token":
            text = ev.get("text", "")
            parts.append(text)
            if in_think and "</think>" in text:
                pre, post = text.split("</think>", 1)
                print(pre, end="", flush=True)
                print(RESET + "\n---\napus> " + post, end="", flush=True)
                in_think = False
            else:
                print(text, end="", flush=True)
            printed_any = True
        elif t == "done":
            done = ev
            break
        elif t == "error":
            print(RESET + f"\n(engine error: {ev.get('message')})")
            return None
    if in_think:
        print(RESET, end="", flush=True)
    dt = time.time() - t0
    if done:
        n = done.get("completion_tokens", 0)
        print(f"\n[{n} tok, {dt:.0f}s, {n/dt:.2f} tok/s, "
              f"{done.get('finish_reason')}]", flush=True)
    elif printed_any:
        print(flush=True)
    return "".join(parts)


if __name__ == "__main__":
    main()
