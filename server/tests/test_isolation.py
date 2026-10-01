#!/usr/bin/env python3
"""Failure isolation (no GPU, no checkpoint): a request the engine cannot admit fails alone. Before,
Engine.acquire ran outside the admission try, so its exception reached the scheduler loop's
catch-all, which fails every request in flight.

  python3 server/tests/test_isolation.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fixtures import FakeEngine  # noqa: E402
from qwserve.scheduler import Request, Scheduler  # noqa: E402


class Client:
    """What the API handler sees of one request: its events."""

    def __init__(self, name, prompt, max_tokens):
        self.name, self.events, self.done = name, [], threading.Event()
        self.req = Request(prompt, {"max_tokens": max_tokens}, self.emit, set())
        self.req.rid = name

    def emit(self, kind, payload=None):
        self.events.append((kind, payload))
        if kind in ("end", "error"):
            self.done.set()

    def count(self, kind):
        return sum(1 for k, _ in self.events if k == kind)


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    e = FakeEngine(capacity=(8192, 8192), step_s=0.004)
    e.poison = {666}
    s = Scheduler(e)

    a = Client("a", [1] * 100, max_tokens=300)  # decodes for ~1.2 s
    s.submit(a.req)
    deadline = time.time() + 5
    while a.count("token") < 20 and time.time() < deadline:
        time.sleep(0.01)
    check("the long request is decoding", a.count("token") >= 20, f"({a.count('token')} tokens)")

    bad = Client("bad", [666] + [1] * 50, max_tokens=10)
    good = Client("good", [2] * 100, max_tokens=5)
    s.submit(bad.req)
    s.submit(good.req)
    bad.done.wait(5)
    good.done.wait(5)
    check("the request the engine cannot admit gets an error and an end",
          [k for k, _ in bad.events][-2:] == ["error", "end"] and bad.events[-1:] == [("end", "abort")],
          f"({bad.events[-2:]})")
    check("it did not hold a slot", e.busy.count(True) == 1, f"(busy {e.busy})")
    check("a request admitted next to it still finishes normally", good.events[-1:] == [("end", "length")],
          f"({good.events[-1:]})")
    tokens_then = a.count("token")
    check("the long request was not failed", a.count("error") == 0 and not a.done.is_set())
    a.done.wait(10)
    check("and runs to its length", a.events[-1:] == [("end", "length")] and a.count("error") == 0,
          f"({a.count('token')} tokens after {tokens_then} when the bad one failed)")
    s.shutdown(2)
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
