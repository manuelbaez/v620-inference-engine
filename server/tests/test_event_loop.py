#!/usr/bin/env python3
"""The event loop stays free while a large chat request is rendered and tokenized (no GPU, no
checkpoint: a tiny tokenizer from fixtures.py). Done on the loop, a 100k-token prompt stalls every
stream for ~0.1 s; the tokenizer's encode keeps the GIL even on a worker thread, so the server
uses encode_batch, which releases it.

  python3 server/tests/test_event_loop.py
"""

import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402
from qwserve.api import Server  # noqa: E402
from qwserve.prompt import ChatPrompt  # noqa: E402


async def worst_lateness(coro_fn):
    """Worst wake-up lateness (ms) of a 2 ms ticker while coro_fn() runs."""
    late, stop = [], asyncio.Event()

    async def ticker():
        while not stop.is_set():
            t = time.perf_counter()
            await asyncio.sleep(0.002)
            late.append((time.perf_counter() - t - 0.002) * 1e3)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    result = await coro_fn()
    took = time.perf_counter() - t0
    await asyncio.sleep(0.05)
    stop.set()
    await task
    return max(late), took, result


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    with tempfile.TemporaryDirectory() as d:
        fixtures.make_model_dir(d)
        prompt = ChatPrompt(d)
        srv = Server.__new__(Server)  # no engine: only the text layer is under test
        srv.prompt = prompt
        srv.tok = prompt.tok

        # encode_batch gives the same ids as encode
        for text in ("", "hello world", fixtures.words(300, 3), "<|im_start|>user\nhi<|im_end|>\n<think>x</think>"):
            same = prompt.encode(text) == prompt.tok.encode(text, add_special_tokens=False).ids
            check(f"same ids as encode for {text[:24]!r}", same)

        # a request big enough that rendering and tokenizing take a few hundred ms
        n_words = 20_000
        while True:
            body = {"messages": [{"role": "user", "content": fixtures.words(n_words, 7)}]}
            t0 = time.perf_counter()
            text = srv.render(body)[0]
            srv.encode(text)
            if time.perf_counter() - t0 > 0.3 or n_words > 3_000_000:
                break
            n_words *= 2
        print(f"request: {len(text) / 1e6:.2f} MB, render + encode {1e3 * (time.perf_counter() - t0):.0f} ms")

        async def on_the_loop():  # what chat() did before
            text = srv.render(body)[0]
            return srv.encode(text)

        async def off_the_loop():
            return (await srv.prepare(body))[3]

        late_on, took_on, ids_on = asyncio.run(worst_lateness(on_the_loop))
        late_off, took_off, ids_off = asyncio.run(worst_lateness(off_the_loop))
        print(f"on the loop: worst tick lateness {late_on:.1f} ms ({took_on * 1e3:.0f} ms); "
              f"prepare(): {late_off:.1f} ms ({took_off * 1e3:.0f} ms)")
        check("the probe sees the old blocking (on the loop, lateness over 100 ms)", late_on > 100)
        check("prepare() keeps the loop free (worst tick lateness under 60 ms)", late_off < 60)
        check("prepare() returns the same tokens", ids_on == ids_off)

        # the handlers' error mapping is unchanged: a template error is a ValueError/TemplateError
        async def bad():
            return await srv.prepare({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "extreme"})
        try:
            asyncio.run(bad())
            check("a bad reasoning_effort still raises ValueError from prepare()", False)
        except ValueError:
            check("a bad reasoning_effort still raises ValueError from prepare()", True)

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
