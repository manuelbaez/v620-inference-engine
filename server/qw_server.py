#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the qw engine (Qwen3.8-Flash-Next).

The engine (libqw_engine.so, C API in src/engine/capi.hpp) does all per-token
work on the 4 GPUs. This process does the per-request text work: the model's
own chat template (Jinja2), HF `tokenizers`, and output parsing that matches
production vLLM's `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`:

  * output starts inside <think> (the generation prompt ends with it) unless
    enable_thinking is false; </think> switches to content; text before it is
    `reasoning` (also sent as `reasoning_content` for older clients);
  * <tool_call> ... </tool_call> blocks are tool calls: <function=NAME> with
    <parameter=KEY>VALUE</parameter> pairs, one leading/trailing newline
    trimmed, values coerced to the tool's JSON schema type.

Requests run concurrently, one per engine slot (continuous batching: every
active request advances one token per batched step; others queue). Prompt
state is reused: a request goes to the slot holding the longest prefix of its
prompt and only prefills the rest (usage.prompt_tokens_details.cached_tokens).
"""

import argparse
import asyncio
import ctypes
import json
import os
import re
import sys
import threading
import time
import uuid

import jinja2
from aiohttp import web
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

THINK_START, THINK_END = "<think>", "</think>"
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
EOS_TEXT = ("<|im_end|>", "<|endoftext|>")


# ----------------------------------------------------------------- engine
class Sampling(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("min_p", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


class Engine:
    """ctypes wrapper of the slot C API. Not thread-safe: only the scheduler
    thread calls it."""

    def __init__(self, lib_path, options):
        lib = ctypes.CDLL(lib_path)
        P, I32P, FP = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
        sig = {
            "qw_open": (P, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
            "qw_error": (ctypes.c_char_p, [P]),
            "qw_num_slots": (ctypes.c_int, [P]),
            "qw_slot_capacity": (ctypes.c_int64, [P, ctypes.c_int]),
            "qw_acquire": (ctypes.c_int, [P, I32P, ctypes.c_int64, ctypes.c_int64]),
            "qw_release": (ctypes.c_int, [P, ctypes.c_int]),
            "qw_set_prompt": (ctypes.c_int64, [P, ctypes.c_int, I32P, ctypes.c_int64]),
            "qw_sample_prompt": (ctypes.c_int32, [P, ctypes.c_int, ctypes.POINTER(Sampling), FP]),
            "qw_decode": (ctypes.c_int, [P, ctypes.c_int, I32P, I32P]),
            "qw_sample_row": (ctypes.c_int32, [P, ctypes.c_int, ctypes.POINTER(Sampling), FP]),
            "qw_top_logprobs": (ctypes.c_int, [P, ctypes.c_int, ctypes.c_int, I32P, FP]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        err = ctypes.create_string_buffer(4096)
        self.h = lib.qw_open(json.dumps(options).encode(), err, len(err))
        if not self.h:
            raise RuntimeError("engine failed to start: " + err.value.decode(errors="replace"))
        self.lib = lib
        self.capacity = [lib.qw_slot_capacity(self.h, i) for i in range(lib.qw_num_slots(self.h))]
        self.max_tokens = max(self.capacity)

    def _err(self):
        return RuntimeError(self.lib.qw_error(self.h).decode(errors="replace"))

    def _check(self, r):
        if r < 0:
            raise self._err()
        return r

    def acquire(self, tokens, max_new):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        r = self.lib.qw_acquire(self.h, arr, len(tokens), max_new)
        if r < 0 and self.lib.qw_error(self.h):
            raise self._err()
        return r  # -1: no free slot fits right now

    def release(self, slot):
        self._check(self.lib.qw_release(self.h, slot))

    def set_prompt(self, slot, tokens):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        return self._check(self.lib.qw_set_prompt(self.h, slot, arr, len(tokens)))

    def sample_prompt(self, slot, s):
        lp = ctypes.c_float()
        return self._check(self.lib.qw_sample_prompt(self.h, slot, ctypes.byref(s), ctypes.byref(lp))), lp.value

    def decode(self, slots, tokens):
        n = len(slots)
        self._check(self.lib.qw_decode(self.h, n, (ctypes.c_int32 * n)(*slots), (ctypes.c_int32 * n)(*tokens)))

    def sample_row(self, row, s):
        lp = ctypes.c_float()
        return self._check(self.lib.qw_sample_row(self.h, row, ctypes.byref(s), ctypes.byref(lp))), lp.value

    def top_logprobs(self, row, k):
        """row < 0: the distribution after the last set_prompt."""
        ids = (ctypes.c_int32 * k)()
        lps = (ctypes.c_float * k)()
        n = self._check(self.lib.qw_top_logprobs(self.h, row, k, ids, lps))
        return list(zip(ids[:n], lps[:n]))


class Request:
    def __init__(self, prompt_ids, body, emit, eos_ids):
        self.prompt = prompt_ids
        self.emit = emit
        self.cancelled = threading.Event()
        mt = body.get("max_completion_tokens") or body.get("max_tokens")
        self.max_new = int(mt) if mt else None
        s = Sampling()
        s.temperature = float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0)
        s.top_p = float(body.get("top_p") or 1.0)
        s.top_k = int(body.get("top_k") or 0)
        s.min_p = float(body.get("min_p") or 0.0)
        s.presence_penalty = float(body.get("presence_penalty") or 0.0)
        s.frequency_penalty = float(body.get("frequency_penalty") or 0.0)
        s.repetition_penalty = float(body.get("repetition_penalty") or 1.0)
        s.seed = int(body.get("seed") or 0)
        self.sampling = s
        self.want_top = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        self.end_ids = set(body.get("stop_token_ids") or [])
        if not body.get("ignore_eos"):
            self.end_ids |= eos_ids
        self.slot = -1
        self.limit = 0      # max new tokens in the slot it got
        self.generated = 0
        self.next = None    # sampled token not yet fed to the engine


class Scheduler:
    """Continuous batching on one thread: admits queued requests into free
    slots (prefill + first token), then advances every active request by one
    token per batched engine step."""

    DEFAULT_RESERVE = 2048  # room reserved when the request sets no max_tokens

    def __init__(self, engine):
        self.e = engine
        self.cv = threading.Condition()
        self.queue = []    # submitted, guarded by cv
        self.waiting = []  # admitted in FIFO order by the scheduler thread
        self.active = []
        threading.Thread(target=self._loop, daemon=True, name="qw-scheduler").start()

    def submit(self, req):
        n = len(req.prompt)
        if n >= self.e.max_tokens:
            raise ValueError(f"prompt is {n} tokens; the context holds {self.e.max_tokens}")
        with self.cv:
            self.queue.append(req)
            self.cv.notify()

    def _finish(self, r, reason):
        r.emit("end", reason)
        if r.slot >= 0:
            self.e.release(r.slot)
            r.slot = -1

    def _took(self, r, tid, lp, top):
        """Handles a sampled token; returns False when the request is done."""
        r.generated += 1
        if tid in r.end_ids:
            r.emit("eos", tid)  # counted in completion_tokens, like vLLM
            self._finish(r, "stop")
            return False
        r.emit("token", (tid, lp, top))
        if r.generated >= r.limit:
            self._finish(r, "length")
            return False
        r.next = tid
        return True

    def _admit(self):
        while self.waiting:
            r = self.waiting[0]
            if r.cancelled.is_set():
                self.waiting.pop(0)
                r.emit("end", "abort")
                continue
            n = len(r.prompt)
            need = r.max_new if r.max_new else self.DEFAULT_RESERVE
            need = max(1, min(need, self.e.max_tokens - n))
            slot = self.e.acquire(r.prompt, need)
            if slot < 0:
                return  # FIFO: wait for a slot to free up
            self.waiting.pop(0)
            r.slot = slot
            try:
                room = self.e.capacity[slot] - n
                r.limit = min(r.max_new, room) if r.max_new else room
                cached = self.e.set_prompt(slot, r.prompt)
                r.emit("start", cached)
                top = self.e.top_logprobs(-1, r.want_top) if r.want_top else None
                tid, lp = self.e.sample_prompt(slot, r.sampling)
            except Exception as ex:  # noqa: BLE001
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            if self._took(r, tid, lp, top):
                self.active.append(r)

    def _step(self):
        for r in [r for r in self.active if r.cancelled.is_set()]:
            self.active.remove(r)
            self._finish(r, "abort")
        if not self.active:
            return
        batch = self.active[:16]
        self.active = self.active[16:] + batch  # round-robin past 16 requests (at most num_slots anyway)
        self.e.decode([r.slot for r in batch], [r.next for r in batch])
        for i, r in enumerate(batch):
            top = self.e.top_logprobs(i, r.want_top) if r.want_top else None
            tid, lp = self.e.sample_row(i, r.sampling)
            if not self._took(r, tid, lp, top):
                self.active.remove(r)

    def _loop(self):
        while True:
            with self.cv:
                while not self.queue and not self.waiting and not self.active:
                    self.cv.wait()
                self.waiting += self.queue  # the queue is shared; waiting is this thread's
                self.queue = []
            try:
                self._admit()
                self._step()
            except Exception as ex:  # noqa: BLE001  engine failure: fail everything in flight
                print(f"scheduler error: {ex}", file=sys.stderr, flush=True)
                for r in self.active:
                    r.emit("error", str(ex))
                    try:
                        self._finish(r, "abort")
                    except Exception:  # noqa: BLE001
                        pass
                self.active = []


# ----------------------------------------------------------------- text
class Detok:
    """Incremental detokenizer for one output segment (holds back partial UTF-8)."""

    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.prefix = 0  # ids[prefix:read] decode to text already emitted (context window)
        self.read = 0

    def add(self, tid):
        self.ids.append(tid)
        prefix_text = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new_text = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if new_text.endswith("�") or len(new_text) <= len(prefix_text):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return new_text[len(prefix_text):]


_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL)
_FUNC_RE = re.compile(r"<\s*function\s*=\s*([^>\n]*)>?")


def _trim_nl(v):
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _schema_types(schema):
    if not isinstance(schema, dict):
        return ["string"]
    types = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []) or []:
            types.update(_schema_types(sub))
    for v in schema.get("enum", []) or []:
        types.add("null" if v is None else "boolean" if isinstance(v, bool) else
                  "integer" if isinstance(v, int) else "number" if isinstance(v, float) else "string")
    return sorted(types) or ["string"]


def _coerce(value, schema):
    """Best-effort coercion like vLLM's coerce_to_schema_type (null > integer >
    number > boolean > object > array > string)."""
    types = {("integer" if t == "int" else "number" if t in ("float", "double") else t) for t in _schema_types(schema)}
    for t in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if t not in types:
            continue
        if t == "null" and value.strip().lower() == "null":
            return None
        if t == "string":
            return value
        if t == "integer":
            try:
                return int(value.strip())
            except ValueError:
                continue
        if t == "number":
            try:
                f = float(value.strip())
                return int(f) if f.is_integer() and "." not in value and "e" not in value.lower() else f
            except ValueError:
                continue
        if t == "boolean":
            v = value.strip().lower()
            if v in ("true", "false"):
                return v == "true"
            continue
        if t in ("object", "array"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict if t == "object" else list):
                    return parsed
            except ValueError:
                continue
    return value


def parse_tool_call(text, tools):
    m = _FUNC_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    body = text[m.end():]
    end = body.find("</function>")
    if end >= 0:
        body = body[:end]
    props = {}
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
    args = {}
    for pm in _PARAM_RE.finditer(body):
        key = pm.group(1).strip()
        val = _trim_nl(pm.group(2))
        args[key] = _coerce(val, props[key]) if key in props else val
    return {
        "id": "chatcmpl-tool-" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


class OutputParser:
    """Token-level router: think/tool tags are single tokens in this vocabulary."""

    def __init__(self, tok, thinking, tools, parse_tools):
        self.tok = tok
        self.ids = {s: tok.token_to_id(s) for s in (THINK_START, THINK_END, TOOL_START, TOOL_END)}
        self.state = "reasoning" if thinking else "content"
        self.tools = tools
        self.parse_tools = parse_tools
        self.seg = {"reasoning": Detok(tok), "content": Detok(tok)}
        self.tool_ids = []
        self.tool_calls = []
        self.reasoning = ""
        self.content = ""

    def feed(self, tid):
        """Returns a delta dict (reasoning / content / tool_calls) or None."""
        if tid == self.ids[THINK_START] and self.state == "reasoning":
            return None
        if tid == self.ids[THINK_END] and self.state in ("reasoning", "content"):
            self.state = "content"
            return None
        if self.parse_tools and tid == self.ids[TOOL_START] and self.state in ("reasoning", "content"):
            self.state = "tool"
            self.tool_ids = []
            return None
        if self.state == "tool":
            if tid == self.ids[TOOL_END]:
                self.state = "content"
                call = parse_tool_call(self.tok.decode(self.tool_ids, skip_special_tokens=False), self.tools)
                if call is None:
                    return None
                call["index"] = len(self.tool_calls)
                self.tool_calls.append(call)
                return {"tool_calls": [call]}
            self.tool_ids.append(tid)
            return None
        text = self.seg[self.state].add(tid)
        if not text:
            return None
        if self.state == "reasoning":
            self.reasoning += text
            return {"reasoning": text}
        self.content += text
        return {"content": text}

    def finish(self):
        """Unterminated tool call at the end: try to parse it anyway."""
        if self.state == "tool" and self.tool_ids:
            call = parse_tool_call(self.tok.decode(self.tool_ids, skip_special_tokens=False), self.tools)
            if call:
                call["index"] = len(self.tool_calls)
                self.tool_calls.append(call)
                return {"tool_calls": [call]}
        return None


# ----------------------------------------------------------------- server
class Server:
    def __init__(self, args):
        self.args = args
        self.model_name = args.served_model_name
        self.tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
        with open(os.path.join(args.model_dir, "chat_template.jinja")) as f:
            src = f.read()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self.template = env.from_string(src)
        self.eos_ids = {self.tok.token_to_id(t) for t in EOS_TEXT}
        opts = {"model_dir": args.model_dir, "ple_dir": args.ple_dir, "prefill_chunk": args.prefill_chunk,
                "slots": [int(x) for x in args.slots.split(",")]}
        t0 = time.time()
        self.engine = Engine(args.lib, opts)
        print(f"engine ready in {time.time() - t0:.1f}s, slots {self.engine.capacity}", flush=True)
        self.sched = Scheduler(self.engine)

    # --- prompt building
    def render(self, body):
        messages = []
        for m in body.get("messages", []):
            m = dict(m)
            rc = m.get("reasoning_content", m.get("reasoning"))
            if rc is not None:
                m["reasoning_content"] = rc
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    c = json.loads(json.dumps(c))
                    fn = c.get("function", c)
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                        except ValueError:
                            pass
                    calls.append(c)
                m["tool_calls"] = calls
            messages.append(m)
        kw = dict(body.get("chat_template_kwargs") or {})
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        if "reasoning_effort" in body and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = body["reasoning_effort"]
        text = self.template.render(messages=messages, tools=tools, add_generation_prompt=body.get("add_generation_prompt", True), **kw)
        thinking = kw.get("enable_thinking", True) is not False
        return text, thinking, tools

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    # --- generation
    async def run(self, prompt_ids, body):
        """Async generator of (kind, payload) events of one request: 'start'
        (cached prompt tokens), 'token' (id, logprob, top), 'eos', then 'end'
        (finish reason) or 'error'."""
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()

        def emit(kind, payload):
            loop.call_soon_threadsafe(q.put_nowait, (kind, payload))

        req = Request(prompt_ids, body, emit, self.eos_ids)
        self.sched.submit(req)
        try:
            while True:
                kind, payload = await q.get()
                yield kind, payload
                if kind in ("end", "error"):
                    break
        finally:
            req.cancelled.set()  # no-op when finished; frees the slot on client disconnect

    # --- handlers
    async def health(self, _):
        return web.json_response({"status": "ok"})

    async def models(self, _):
        return web.json_response({"object": "list", "data": [{
            "id": self.model_name, "object": "model", "created": int(time.time()), "owned_by": "qw",
            "max_model_len": int(self.engine.max_tokens)}]})

    async def tokenize(self, request):
        body = await request.json()
        if "messages" in body:
            text, _, _ = self.render(body)
        else:
            text = body.get("prompt", "")
        ids = self.encode(text)
        return web.json_response({"count": len(ids), "max_model_len": int(self.engine.max_tokens), "tokens": ids})

    def _stop_strings(self, body):
        stop = body.get("stop")
        if stop is None:
            return []
        return [stop] if isinstance(stop, str) else list(stop)

    async def chat(self, request):
        body = await request.json()
        if int(body.get("n") or 1) != 1:
            return web.json_response({"error": {"message": "n > 1 is not supported", "type": "invalid_request_error"}}, status=400)
        try:
            text, thinking, tools = self.render(body)
        except jinja2.exceptions.TemplateError as e:
            return web.json_response({"error": {"message": str(e), "type": "invalid_request_error"}}, status=400)
        prompt_ids = self.encode(text)
        parser = OutputParser(self.tok, thinking, tools, parse_tools=bool(tools))
        stops = self._stop_strings(body)
        rid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": 0}
        logprobs = []
        finish = "stop"
        cached = 0

        def chunk(delta, fin=None, lp=None):
            c = {"index": 0, "delta": delta, "logprobs": lp, "finish_reason": fin}
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.model_name, "choices": [c]}

        def lp_entry(tid, lp, top):
            s = self.tok.decode([tid], skip_special_tokens=False)
            e = {"token": s, "logprob": lp, "bytes": list(s.encode())}
            e["top_logprobs"] = [{"token": self.tok.decode([i], skip_special_tokens=False), "logprob": l,
                                  "bytes": list(self.tok.decode([i], skip_special_tokens=False).encode())}
                                 for i, l in (top or [])]
            return e

        def check_stop():
            for sstr in stops:
                k = parser.content.find(sstr)
                if k >= 0:
                    return k
            return -1

        resp = None
        if stream:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await resp.prepare(request)
            await resp.write(b"data: " + json.dumps(chunk({"role": "assistant", "content": ""})).encode() + b"\n\n")

        emitted_content = 0  # for stop-string truncation in streaming
        reasoning_tokens = 0
        try:
            async for kind, payload in self.run(prompt_ids, body):
                if kind == "error":
                    raise RuntimeError(payload)
                if kind == "start":
                    cached = payload
                    continue
                if kind == "eos":
                    usage["completion_tokens"] += 1
                    continue
                if kind == "end":
                    finish = {"abort": "stop"}.get(payload, payload)
                    break
                tid, lp, top = payload
                usage["completion_tokens"] += 1
                if parser.state == "reasoning":
                    reasoning_tokens += 1
                delta = parser.feed(tid)
                entry = lp_entry(tid, lp, top) if body.get("logprobs") else None
                if entry:
                    logprobs.append(entry)
                if stops and "content" in (delta or {}):
                    k = check_stop()
                    if k >= 0:
                        parser.content = parser.content[:k]
                        delta = {"content": parser.content[emitted_content:]} if k > emitted_content else None
                        finish = "stop"
                        if stream and delta:
                            await resp.write(b"data: " + json.dumps(chunk(delta)).encode() + b"\n\n")
                        break
                if stream and delta:
                    if "reasoning" in delta:
                        delta["reasoning_content"] = delta["reasoning"]
                    if "content" in delta:
                        emitted_content += len(delta["content"])
                    await resp.write(b"data: " + json.dumps(chunk(delta, lp={"content": [entry]} if entry else None)).encode() + b"\n\n")
            tail = parser.finish()
            if stream and tail:
                await resp.write(b"data: " + json.dumps(chunk(tail)).encode() + b"\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except Exception as e:  # noqa: BLE001
            if stream:
                await resp.write(b"data: " + json.dumps({"error": {"message": str(e)}}).encode() + b"\n\n")
                await resp.write(b"data: [DONE]\n\n")
                return resp
            return web.json_response({"error": {"message": str(e), "type": "server_error"}}, status=500)

        if parser.tool_calls and finish == "stop":
            finish = "tool_calls"
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        if stream:
            await resp.write(b"data: " + json.dumps(chunk({}, fin=finish)).encode() + b"\n\n")
            if include_usage:
                u = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.model_name,
                     "choices": [], "usage": usage}
                await resp.write(b"data: " + json.dumps(u).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp
        msg = {"role": "assistant", "content": parser.content if (parser.content or not parser.tool_calls) else None,
               "reasoning": parser.reasoning or None, "reasoning_content": parser.reasoning or None}
        if parser.tool_calls:
            msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in parser.tool_calls]
        return web.json_response({
            "id": rid, "object": "chat.completion", "created": created, "model": self.model_name,
            "choices": [{"index": 0, "message": msg, "logprobs": {"content": logprobs} if body.get("logprobs") else None,
                         "finish_reason": finish}],
            "usage": usage})

    async def completions(self, request):
        body = await request.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_ids = prompt
        elif isinstance(prompt, str):
            prompt_ids = self.encode(prompt)
        else:
            return web.json_response({"error": {"message": "prompt must be a string or a token list"}}, status=400)
        rid = "cmpl-" + uuid.uuid4().hex
        created = int(time.time())
        stream = bool(body.get("stream"))
        det = Detok(self.tok)
        text = ""
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": 0}
        finish, cached = "stop", 0
        stops = self._stop_strings(body)
        resp = None
        if stream:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await resp.prepare(request)
        async for kind, payload in self.run(prompt_ids, body):
            if kind == "error":
                return web.json_response({"error": {"message": payload}}, status=500)
            if kind == "start":
                cached = payload
                continue
            if kind == "eos":
                usage["completion_tokens"] += 1
                continue
            if kind == "end":
                finish = payload
                break
            tid = payload[0]
            usage["completion_tokens"] += 1
            d = det.add(tid)
            text += d
            hit = min((text.find(s) for s in stops if s in text), default=-1)
            if hit >= 0:
                d = d[: max(0, len(d) - (len(text) - hit))]
                text = text[:hit]
                finish = "stop"
            if stream and d:
                c = {"id": rid, "object": "text_completion", "created": created, "model": self.model_name,
                     "choices": [{"index": 0, "text": d, "finish_reason": None}]}
                await resp.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            if hit >= 0:
                break
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}
        if stream:
            c = {"id": rid, "object": "text_completion", "created": created, "model": self.model_name,
                 "choices": [{"index": 0, "text": "", "finish_reason": finish}], "usage": usage}
            await resp.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"id": rid, "object": "text_completion", "created": created, "model": self.model_name,
                                  "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    ap.add_argument("--ple-dir", default="/mnt/llms/qwen3.8-flash-next-ple/ples_int4")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libqw_engine.so"))
    ap.add_argument("--served-model-name", default="qw/qwen3.8-flash-next")
    ap.add_argument("--slots", default="131072,65536,32768,32768",
                    help="context size of each sequence slot (concurrent requests); the largest is max_model_len")
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    args = ap.parse_args()
    srv = Server(args)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get("/health", srv.health)
    app.router.add_get("/v1/models", srv.models)
    app.router.add_post("/tokenize", srv.tokenize)
    app.router.add_post("/v1/chat/completions", srv.chat)
    app.router.add_post("/v1/completions", srv.completions)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    sys.exit(main())
