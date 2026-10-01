#!/usr/bin/env python3
"""Token marshalling (no GPU, no model): the scheduler hands one request's prompt to the engine on
every pass while the request waits for a slot or loads from disk, and rebuilding its ctypes array
costs 5 ms per 100k tokens with the GIL held, which delays the event loop that streams every
client's tokens. The array is built once per request; polling a long prompt leaves the loop free.

  python3 server/tests/test_marshal.py
"""

import asyncio
import ctypes
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve.engine import Engine, Tokens, c_array  # noqa: E402


class FakeLib:
    """Records the token arrays the C API receives."""

    def __init__(self):
        self.seen = []

    def qw_acquire(self, h, arr, n, max_new):
        self.seen.append(("acquire", arr, n))
        return 0

    def qw_prefetch(self, h, slot, arr, n, media, n_media):
        self.seen.append(("prefetch", arr, n))
        return 0

    def qw_begin_prompt(self, h, slot, arr, n, media, n_media):
        self.seen.append(("begin_prompt", arr, n))
        return 0

    def qw_error(self, h):
        return b""


def engine():
    e = Engine.__new__(Engine)  # no library: only the marshalling is under test
    e.h, e.lib = 1, FakeLib()
    return e


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    n = 100_000
    toks = [random.randrange(150000) for _ in range(n)]

    # one array for every call about the same request
    e = engine()
    prompt = Tokens(toks)
    e.acquire(prompt, 100)
    for _ in range(5):
        e.prefetch(0, prompt)
    e.begin_prompt(0, prompt)
    arrays = {id(a) for _, a, _ in e.lib.seen}
    check("one array for acquire, 5 prefetches and begin_prompt", len(arrays) == 1, f"({len(e.lib.seen)} calls)")
    arr = e.lib.seen[0][1]
    check("the array holds the tokens", len(arr) == n and list(arr[:4]) == toks[:4] and arr[n - 1] == toks[-1])
    check("the length passed is the prompt's", all(c[2] == n for c in e.lib.seen))

    # a plain list still works (built per call), and a changed length is noticed
    e2 = engine()
    e2.prefetch(0, toks[:10])
    e2.prefetch(0, toks[:10])
    check("a plain list gives a correct array each time",
          all(list(a) == toks[:10] for _, a, _ in e2.lib.seen) and e2.lib.seen[0][1] is not e2.lib.seen[1][1])
    t = Tokens(toks[:10])
    first = c_array(t)
    t.append(5)
    second = c_array(t)
    check("a Tokens list that changed length is rebuilt", first is not second and len(second) == 11 and second[10] == 5)
    check("an empty prompt works", len(c_array(Tokens())) == 0 and len(c_array([])) == 0)

    # what it costs
    build = min(_time(lambda: c_array(list(toks))) for _ in range(3))
    first_call = _time(lambda: engine().prefetch(0, Tokens(toks)))
    e3, p3 = engine(), Tokens(toks)
    e3.prefetch(0, p3)
    later = min(_time(lambda: e3.prefetch(0, p3)) for _ in range(20))
    legacy = min(_time(lambda: (ctypes.c_int32 * n)(*toks)) for _ in range(3))
    print(f"100k tokens: first call {first_call * 1e3:.2f} ms, later calls {later * 1e3:.3f} ms "
          f"(the old per-call rebuild: {legacy * 1e3:.2f} ms; building from a list: {build * 1e3:.2f} ms)")
    check("a later call is at least 20x cheaper than the old rebuild", later * 20 < legacy)

    # the event loop stays free while a scheduler-like thread polls a long prompt
    e4, p4 = engine(), Tokens(toks)
    stop = threading.Event()

    def poll():  # _poll_loading: prefetch every pass, a few ms of other work between
        while not stop.is_set():
            e4.prefetch(0, p4)
            time.sleep(0.0015)

    async def lateness(duration=2.0):
        late, end = [], time.perf_counter() + duration
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            await asyncio.sleep(0.002)
            late.append((time.perf_counter() - t0 - 0.002) * 1e3)
        return late

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    late = asyncio.run(lateness())
    stop.set()
    th.join()
    med = statistics.median(late)
    print(f"event-loop lateness while polling a 100k-token prompt: median {med:.2f} ms, max {max(late):.2f} ms "
          "(4.7 ms median before the prompt buffer was cached)")
    check("the event loop is not held by the polling (median lateness under 2 ms)", med < 2.0)

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    sys.exit(main())
