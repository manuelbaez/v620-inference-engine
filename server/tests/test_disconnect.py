#!/usr/bin/env python3
"""A client that goes away (no GPU, no checkpoint; the real handlers and scheduler over a FakeEngine): its request
must stop where it is. Three cases, each cut off by the client after 0.3 s: a long prompt still prefilling, a request
that does not stream and is decoding, and a streaming one. With the server's run options (qwserve.api.RUN_OPTIONS:
handler_cancellation) the slot is free and the engine idle shortly after; the first two are also run without that
option, where they carry on, which is the behaviour the option exists for.

  python3 server/tests/test_disconnect.py
"""

import asyncio
import os
import sys
import tempfile

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402
from qwserve.api import RUN_OPTIONS  # noqa: E402
from test_api import make_server  # noqa: E402


async def case(model_dir, options, body, what):
    """Sends `body`, drops the connection after 0.3 s, and returns (engine busy 0.5 s later, work done after that)."""
    srv = make_server(model_dir)
    srv.engine = fixtures.FakeEngine(capacity=(8192, 8192), step_s=0.01, prefill_s=0.02)
    srv.engine.has_vision = False
    srv.engine.prefill_chunk = 64  # a 5,000-token prompt takes ~80 passes, 1.6 s
    srv.sched.e = srv.engine
    app = web.Application()
    app.router.add_post("/v1/chat/completions", srv.chat)
    # a real listener: aiohttp's TestServer always turns handler cancellation on, whatever the server itself does
    runner = web.AppRunner(app, **options)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    async with aiohttp.ClientSession() as c:
        try:
            await asyncio.wait_for(c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body), 0.3)
        except asyncio.TimeoutError:
            pass
    await asyncio.sleep(0.5)
    busy = any(srv.engine.busy)
    before = (srv.engine.prefills, srv.engine.steps)
    await asyncio.sleep(0.3)
    carried_on = (srv.engine.prefills, srv.engine.steps) != before
    await runner.cleanup()
    print(f"  {what}: slot busy {busy}, still working {carried_on} (prefill passes {srv.engine.prefills}, steps {srv.engine.steps})")
    return busy or carried_on


async def run():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        model_dir = fixtures.make_model_dir(d)
        long_prompt = [{"role": "user", "content": fixtures.words(2500)}]
        short = [{"role": "user", "content": "hello alpha beta"}]
        off = {"enable_thinking": False}
        cases = [("a prompt still prefilling", {"model": "qw/test", "messages": long_prompt, "max_tokens": 5, "chat_template_kwargs": off}),
                 ("decoding, not streaming", {"model": "qw/test", "messages": short, "max_tokens": 400, "chat_template_kwargs": off}),
                 ("decoding, streaming", {"model": "qw/test", "messages": short, "max_tokens": 400, "stream": True, "chat_template_kwargs": off})]
        print("with the server's run options:")
        for what, body in cases:
            alive = await case(model_dir, RUN_OPTIONS, body, what)
            fails += alive
            print(f"{what}: {'FAIL (the request carried on)' if alive else 'ok'}")
        print("without handler cancellation (what the option is for):")
        for what, body in cases[:2]:
            alive = await case(model_dir, {}, body, what)
            fails += not alive
            print(f"{what}: {'carries on, as expected' if alive else 'FAIL (expected it to carry on: the test does not show the problem)'}")
    print("FAILED" if fails else "all ok")
    return fails


def main():
    sys.exit(1 if asyncio.run(run()) else 0)


if __name__ == "__main__":
    main()
