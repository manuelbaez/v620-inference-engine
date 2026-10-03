#!/usr/bin/env python3
"""The request log line says where the time before the first token went (no GPU, no checkpoint): the
wait for a slot, the disk load of a cached prompt, restoring it, the engine's prefill calls (with the saves
to the prefix cache and the wait for pinned buffers inside them) and what others' work took in between.

  python3 server/tests/test_request_log.py
"""

import contextlib
import io
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fixtures import FakeEngine  # noqa: E402
from qwserve.scheduler import Request, Scheduler  # noqa: E402


class LoadingEngine(FakeEngine):
    """A prompt whose cache is on disk: prefetch reports a background load for `load_s` seconds."""

    def __init__(self, load_s):
        super().__init__()
        self.load_s, self.load_from = load_s, None

    def prefetch(self, slot, tokens, media=None):
        if self.load_from is None:
            self.load_from = time.time()
        return time.time() - self.load_from < self.load_s


def run(engine, name, max_tokens=3):
    done = threading.Event()
    req = Request([1] * 50, {"max_tokens": max_tokens}, lambda kind, payload=None: kind in ("end", "error") and done.set(), set())
    req.rid = name
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sched = Scheduler(engine)
        sched.submit(req)
        done.wait(10)
        time.sleep(0.1)
        sched.shutdown()
    return [line for line in out.getvalue().splitlines() if line.startswith(f"request {name}:")]


def field(line, name):
    m = re.search(rf"{name} ([0-9.]+) s", line)
    return float(m.group(1)) if m else None


def main():
    fails = 0

    def check(what, ok):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'}")

    lines = run(FakeEngine(), "plain")
    check("one request line", len(lines) == 1)
    line = lines[0] if lines else ""
    print(line)
    check("the line keeps its old fields first (ttft, decode, slot)", re.search(r"ttft [0-9.]+ s, decode .*, slot \d+ \| queue", line) is not None)
    for name in ("queue", "load", "restore", "prefill", "saves", "pin wait", "interleaved"):
        check(f"the line has {name}", field(line, name) is not None)
    check("no disk load: load 0.0", field(line, "load") == 0.0)

    lines = run(LoadingEngine(0.4), "disk")
    line = lines[0] if lines else ""
    print(line)
    load = field(line, "load")
    check("a disk load of 0.4 s shows as load", load is not None and 0.35 <= load <= 0.9)
    print("all ok" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
