"""OpenAI-compatible HTTP API: /v1/chat/completions, /v1/completions,
/v1/models, /tokenize, /health."""

import asyncio
import json
import sys
import time
import traceback
import uuid

import jinja2
from aiohttp import web

from .detok import Detok
from .engine import Engine
from .output_parser import OutputParser
from .prompt import ChatPrompt
from .scheduler import Request, Scheduler
from . import vision


class Server:
    def __init__(self, args):
        self.args = args
        self.model_name = args.served_model_name
        self.prompt = ChatPrompt(args.model_dir)
        self.tok = self.prompt.tok
        self.eos_ids = self.prompt.eos_ids
        opts = {"model_dir": args.model_dir, "ple_dir": args.ple_dir, "prefill_chunk": args.prefill_chunk,
                "slots": [int(x) for x in args.slots.split(",")]}
        t0 = time.time()
        self.engine = Engine(args.lib, opts)
        print(f"engine ready in {time.time() - t0:.1f}s, slots {self.engine.capacity}, "
              f"MTP drafts {args.mtp if self.engine.has_mtp else 0}", flush=True)
        im_start = self.tok.token_to_id("<|im_start|>")
        if im_start is not None:
            self.engine.set_boundary_token(im_start)
        self.vision = vision.VisionPreprocessor(args.model_dir) if self.engine.has_vision else None
        self.sched = Scheduler(self.engine, args.mtp)
        # thinking: server defaults, and what ends a thinking section cut short by its budget
        # (Qwen's recommended wording, then the closing tag)
        self.prompt.default_effort = args.reasoning_effort
        self.prompt.default_budget = args.thinking_budget
        self.think_close = (self.tok.token_to_id("</think>"), self.prompt.encode(
            "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
            "directly now.\n</think>\n\n"))

    def render(self, body):
        return self.prompt.render(body)

    def encode(self, text):
        return self.prompt.encode(text)

    async def offload(self, fn, *args):
        """Runs fn(*args) on a worker thread. Rendering the chat template and tokenizing a 100k-token
        prompt take 80-250 ms, and done on the event loop every stream stalls meanwhile."""
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    async def prepare(self, body):
        """The chat request's template and tokens, off the event loop: (thinking, tools,
        think_budget, prompt_ids)."""
        def work():
            text, thinking, tools, think_budget = self.render(body)
            return thinking, tools, think_budget, self.encode(text)
        return await self.offload(work)

    # --- generation
    async def media(self, body, prompt_ids):
        """Decodes and preprocesses the request's images and videos (off the event loop) and
        expands their placeholders in prompt_ids. Returns (prompt_ids, media)."""
        parts = self.prompt.media_parts(body)
        if not parts:
            return prompt_ids, []
        if self.vision is None:
            raise ValueError("this engine has no vision tower")
        loop = asyncio.get_running_loop()
        media = await loop.run_in_executor(None, lambda: [vision.media_of_part(self.vision, p) for p in parts])
        return vision.expand(prompt_ids, media, self.tok), media

    async def run(self, prompt_ids, body, media=None, think_budget=None, rid="-"):
        """Async generator of (kind, payload) events of one request: 'start'
        (cached prompt tokens), 'token' (id, logprob, top), 'eos', then 'end'
        (finish reason) or 'error'."""
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()

        def emit(kind, payload):
            loop.call_soon_threadsafe(q.put_nowait, (kind, payload))

        req = Request(prompt_ids, body, emit, self.eos_ids, media)
        req.rid = rid
        if think_budget is not None:  # the prompt ends inside <think>: cap the thinking
            req.think_left = think_budget
            req.think_close = self.think_close
        self.sched.submit(req)
        try:
            while True:
                kind, payload = await q.get()
                yield kind, payload
                if kind in ("end", "error"):
                    break
        finally:
            req.cancelled.set()  # no-op when finished; frees the slot on client disconnect

    async def on_shutdown(self, _app):
        """Stop the scheduler, then save the slots' conversations to the disk tier."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.sched.shutdown)
        t0 = time.time()
        await loop.run_in_executor(None, self.engine.persist)
        print(f"persisted the prefix cache in {time.time() - t0:.1f}s", flush=True)

    # --- handlers
    async def health(self, _):
        st = dict(self.sched.stats)
        if st["rows"]:
            st["tokens_per_request_step"] = round(st["tokens"] / st["rows"], 3)
            st["ms_per_step"] = round(1e3 * st["time"] / st["steps"], 2)
        ok, why = self.sched.health()  # 503 for a failed or wedged engine, so a supervisor can restart it
        body = {"status": "ok" if ok else "unhealthy", "decode": st, "prefix_cache": self.sched.cache_stats}
        if not ok:
            body["reason"] = why
        return web.json_response(body, status=200 if ok else 503)

    async def models(self, _):
        return web.json_response({"object": "list", "data": [{
            "id": self.model_name, "object": "model", "created": int(time.time()), "owned_by": "qw",
            "max_model_len": int(self.engine.max_tokens)}]})

    async def tokenize(self, request):
        body = await request.json()

        def work():
            text = self.render(body)[0] if "messages" in body else body.get("prompt", "")
            return self.encode(text)
        ids = await self.offload(work)
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
            thinking, tools, think_budget, prompt_ids = await self.prepare(body)
        except (jinja2.exceptions.TemplateError, ValueError) as e:
            return web.json_response({"error": {"message": str(e), "type": "invalid_request_error"}}, status=400)
        try:
            prompt_ids, media = await self.media(body, prompt_ids)
        except (ValueError, OSError) as e:
            return web.json_response({"error": {"message": f"media: {e}", "type": "invalid_request_error"}}, status=400)
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
        timings = None  # llama.cpp-style per-request timings (llama-swap's activity log reads them)
        reasoning_tokens = 0
        try:
            async for kind, payload in self.run(prompt_ids, body, media, think_budget, rid[-12:]):
                if kind == "error":
                    raise RuntimeError(payload)
                if kind == "start":
                    cached = payload
                    continue
                if kind == "timings":
                    timings = payload
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
        except ConnectionResetError:  # the client went away (e.g. an agent cancelled): nothing to answer
            print(f"request {rid[-12:]}: client disconnected", flush=True)
            return resp
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"request {rid[-12:]}: error: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
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
            last = chunk({}, fin=finish)
            if timings:
                last["timings"] = timings
            try:
                await resp.write(b"data: " + json.dumps(last).encode() + b"\n\n")
                if include_usage:
                    u = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.model_name,
                         "choices": [], "usage": usage}
                    if timings:
                        u["timings"] = timings
                    await resp.write(b"data: " + json.dumps(u).encode() + b"\n\n")
                await resp.write(b"data: [DONE]\n\n")
            except ConnectionResetError:  # the client left before the last chunks: the answer was complete
                print(f"request {rid[-12:]}: client disconnected before the end of the stream", flush=True)
            return resp
        msg = {"role": "assistant", "content": parser.content if (parser.content or not parser.tool_calls) else None,
               "reasoning": parser.reasoning or None, "reasoning_content": parser.reasoning or None}
        if parser.tool_calls:
            msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in parser.tool_calls]
        return web.json_response({
            "id": rid, "object": "chat.completion", "created": created, "model": self.model_name,
            "choices": [{"index": 0, "message": msg, "logprobs": {"content": logprobs} if body.get("logprobs") else None,
                         "finish_reason": finish}],
            "usage": usage, **({"timings": timings} if timings else {})})

    async def completions(self, request):
        body = await request.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_ids = prompt
        elif isinstance(prompt, str):
            prompt_ids = await self.offload(self.encode, prompt)
        else:
            return web.json_response({"error": {"message": "prompt must be a string or a token list"}}, status=400)
        rid = "cmpl-" + uuid.uuid4().hex
        created = int(time.time())
        stream = bool(body.get("stream"))
        det = Detok(self.tok)
        text = ""
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": 0}
        finish, cached, timings = "stop", 0, None
        stops = self._stop_strings(body)
        resp = None
        if stream:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await resp.prepare(request)
        async for kind, payload in self.run(prompt_ids, body, rid=rid[-12:]):
            if kind == "error":
                return web.json_response({"error": {"message": payload}}, status=500)
            if kind == "start":
                cached = payload
                continue
            if kind == "timings":
                timings = payload
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
                 "choices": [{"index": 0, "text": "", "finish_reason": finish}], "usage": usage,
                 **({"timings": timings} if timings else {})}
            await resp.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"id": rid, "object": "text_completion", "created": created, "model": self.model_name,
                                  "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage,
                                  **({"timings": timings} if timings else {})})
