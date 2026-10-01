#!/usr/bin/env python3
"""The HTTP API end to end (no GPU, no checkpoint): the real aiohttp handlers, scheduler thread, chat
template and output parser over a FakeEngine and a tiny tokenizer: chat (plain and streaming), /tokenize,
/health, concurrent requests, and the 400s for bad input.

  python3 server/tests/test_api.py
"""

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402
from qwserve import vision  # noqa: E402
from qwserve.api import Server  # noqa: E402
from qwserve.prompt import ChatPrompt  # noqa: E402
from qwserve.scheduler import Scheduler  # noqa: E402


def make_server(model_dir):
    srv = Server.__new__(Server)  # no engine library: the text layer, the scheduler and the handlers are real
    srv.args = SimpleNamespace()
    srv.model_name = "qw/test"
    srv.prompt = ChatPrompt(model_dir)
    srv.tok = srv.prompt.tok
    srv.eos_ids = srv.prompt.eos_ids
    srv.engine = fixtures.FakeEngine(capacity=(8192, 8192), step_s=0.002)
    srv.engine.has_vision = False
    srv.vision = None
    srv.sched = Scheduler(srv.engine, 0)
    srv.prompt.default_effort = "medium"
    srv.prompt.default_budget = -1
    srv.think_close = (srv.tok.token_to_id("</think>"), srv.prompt.encode("\n</think>\n\n"))
    return srv


async def run(srv):
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    app = web.Application()
    app.router.add_get("/health", srv.health)
    app.router.add_post("/tokenize", srv.tokenize)
    app.router.add_post("/v1/chat/completions", srv.chat)
    app.router.add_post("/v1/completions", srv.completions)
    async with TestClient(TestServer(app)) as c:
        msg = [{"role": "user", "content": "hello alpha beta"}]
        chat = {"model": "qw/test", "messages": msg, "max_tokens": 5, "chat_template_kwargs": {"enable_thinking": False}}

        r = await c.post("/v1/chat/completions", json=chat)
        body = await r.json()
        check("chat answers 200 with a length-limited completion",
              r.status == 200 and body["choices"][0]["finish_reason"] == "length"
              and body["usage"]["completion_tokens"] == 5, f"({r.status}, {body['usage']})")
        check("and it carries llama-swap's timings", "timings" in body)

        r = await c.post("/v1/chat/completions", json={**chat, "stream": True, "stream_options": {"include_usage": True}})
        text = await r.text()
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        check("streaming ends with [DONE] and has the role, content and finish chunks",
              text.rstrip().endswith("data: [DONE]") and events[0]["choices"][0]["delta"].get("role") == "assistant"
              and any(e["choices"] and e["choices"][0]["finish_reason"] == "length" for e in events)
              and any(e["choices"] and e["choices"][0]["delta"].get("content") for e in events),
              f"({len(events)} events)")
        check("the usage chunk follows", events[-1].get("usage", {}).get("completion_tokens") == 5)

        r = await c.post("/v1/completions", json={"model": "qw/test", "prompt": "alpha beta", "max_tokens": 3})
        body = await r.json()
        check("/v1/completions works", r.status == 200 and body["usage"]["completion_tokens"] == 3, f"({r.status})")

        r = await c.post("/tokenize", json={"messages": msg})
        body = await r.json()
        text_ids = srv.encode(srv.render({"messages": msg})[0])
        check("/tokenize returns the rendered prompt's tokens", body["tokens"] == text_ids and body["count"] == len(text_ids))

        results = await asyncio.gather(*[c.post("/v1/chat/completions", json=chat) for _ in range(6)])
        check("six concurrent requests all complete (two slots)", all(x.status == 200 for x in results),
              f"({[x.status for x in results]})")

        r = await c.get("/health")
        check("/health is 200 and ok", r.status == 200 and (await r.json())["status"] == "ok")
        srv.engine.failed_with = "hip: an illegal memory access was encountered"
        r = await c.get("/health")
        body = await r.json()
        check("/health is 503 with the reason when the engine failed",
              r.status == 503 and "illegal memory" in body["reason"], f"({r.status})")
        srv.engine.failed_with = ""

        r = await c.post("/v1/chat/completions", json={**chat, "reasoning_effort": "extreme"})
        check("a bad reasoning_effort is a 400", r.status == 400, f"({r.status})")
        r = await c.post("/v1/chat/completions", json={**chat, "n": 2})
        check("n > 1 is a 400", r.status == 400)

        # media: no vision tower is a 400; with one, a private URL is refused (400), not fetched
        part = {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/x.png"}}
        media_chat = {**chat, "messages": [{"role": "user", "content": [part, {"type": "text", "text": "what"}]}]}
        r = await c.post("/v1/chat/completions", json=media_chat)
        check("an image without a vision tower is a 400", r.status == 400, f"({(await r.json())['error']['message'][:50]})")
        srv.vision = vision.VisionPreprocessor("/nonexistent")
        r = await c.post("/v1/chat/completions", json=media_chat)
        msg_text = (await r.json())["error"]["message"]
        check("an image at a loopback URL is refused with a 400", r.status == 400 and "private" in msg_text,
              f"({r.status}, {msg_text[:60]!r})")
    return fails


def main():
    with tempfile.TemporaryDirectory() as d:
        fixtures.make_model_dir(d)
        srv = make_server(d)
        fails = asyncio.run(run(srv))
        srv.sched.shutdown(2)
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
