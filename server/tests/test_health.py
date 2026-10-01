#!/usr/bin/env python3
"""Health (no GPU, no checkpoint): /health says so when the engine can no longer serve, and the
watchdog exits the process so its supervisor restarts it. A failed rank or a collective timeout
is permanent (every later job fails the same way) and a wedged GPU hangs the scheduler thread in
one engine call; both used to leave a server that answered /health "ok" and failed every request.

  python3 server/tests/test_health.py
"""

import asyncio
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fixtures import FakeEngine  # noqa: E402
from qwserve.api import Server  # noqa: E402
from qwserve.engine import Engine  # noqa: E402
from qwserve.scheduler import Request, Scheduler  # noqa: E402

Scheduler.WATCH_INTERVAL = 0.05  # read when a scheduler starts its watchdog


def wait_for(cond, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    class Exited(BaseException):
        """What os._exit does to the watchdog thread: nothing runs after it."""

    class FakeLib:
        """qw_engine_failure as the C library implements it: the length of the message, copied into buf."""
        message = b""

        def qw_engine_failure(self, h, buf, n):
            buf.value = self.message[:n - 1]
            return len(self.message)

    binding = Engine.__new__(Engine)
    binding.h, binding.lib = 1, FakeLib()
    check("the binding reports no failure", binding.failure() == "")
    binding.lib.message = "hip: an illegal memory access was encountered".encode()
    check("and the engine's message when it failed", binding.failure() == "hip: an illegal memory access was encountered")

    exits = []
    real_exit, real_hook = os._exit, threading.excepthook

    def fake_exit(code):
        exits.append(code)
        raise Exited

    os._exit = fake_exit  # the watchdog's exit, recorded instead
    threading.excepthook = lambda a: None if a.exc_type is Exited else real_hook(a)
    try:
        # healthy, then a failed engine, then (a test double only) recovered
        e = FakeEngine()
        s = Scheduler(e)
        s.EXIT_GRACE = 0.4
        check("a working engine is healthy", s.health() == (True, "ok"))
        e.failed_with = "collective timed out: rank 1 timed out waiting for rank 3 at collective offset 12"
        ok, why = s.health()
        check("a failed engine is not, and says why", not ok and "collective timed out" in why, f"({why})")

        # /health: 503 with the reason, 200 otherwise
        srv = Server.__new__(Server)
        srv.sched = s

        def get_health():
            resp = asyncio.run(srv.health(None))
            return resp.status, json.loads(resp.text)
        status, body = get_health()
        check("/health answers 503 with the reason", status == 503 and body["status"] == "unhealthy"
              and "collective timed out" in body["reason"], f"({status}, {body['status']})")

        # unhealthy for longer than the grace: the watchdog exits (code 3)
        check("the watchdog exits after the grace", wait_for(lambda: exits == [3], 3.0), f"({exits})")
        s.shutdown(1)

        # recovering inside the grace cancels the exit
        exits.clear()
        e2 = FakeEngine()
        s2 = Scheduler(e2)
        s2.EXIT_GRACE = 1.0
        e2.failed_with = "hip: transient"
        time.sleep(0.3)
        e2.failed_with = ""
        time.sleep(1.3)
        check("a recovery inside the grace does not exit", exits == [] and s2.health()[0], f"({exits})")
        status, body = (lambda r: (r.status, json.loads(r.text)))(asyncio.run(Server.health(
            type("S", (), {"sched": s2})(), None)))
        check("/health answers 200 and ok again", status == 200 and body["status"] == "ok")
        s2.shutdown(1)

        # a wedged GPU: an engine call that does not return
        exits.clear()
        e3 = FakeEngine(step_s=0)
        e3.generate_block = threading.Event()
        s3 = Scheduler(e3)
        s3.STUCK_SECONDS = 0.3
        s3.EXIT_GRACE = 0.4
        got = []
        r = Request([1] * 50, {"max_tokens": 5}, lambda kind, payload=None: got.append(kind), set())
        r.rid = "stuck"
        s3.submit(r)
        check("the scheduler is inside generate()", wait_for(lambda: s3.call and s3.call[0] == "generate"),
              f"({s3.call})")
        check("a short call is not stuck", s3.health()[0])
        time.sleep(0.5)
        ok, why = s3.health()
        check("an engine call that runs too long is reported", not ok and "generate" in why, f"({why})")
        check("and the watchdog exits for it", wait_for(lambda: exits == [3], 3.0), f"({exits})")
        e3.generate_block.set()  # unblock; the request finishes and the engine is healthy again
        check("it recovers when the call returns", wait_for(lambda: s3.health()[0]))
        s3.shutdown(2)

        # a scheduler thread that died is reported
        s4 = Scheduler(FakeEngine())
        s4.WATCHDOG_EXIT = False
        s4.shutdown(1)
        s4.stopping = False  # (it is stopped and no longer stopping: what a crashed loop looks like)
        ok, why = s4.health()
        check("a dead scheduler thread is reported", not ok and "died" in why, f"({why})")
    finally:
        os._exit, threading.excepthook = real_exit, real_hook

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
