#!/usr/bin/env python3
"""How a prefill pass is shared (no GPU): long prompts waiting together prefill in parallel, in equal pieces of the
pass; short prompts go first and whole, and when they are there the long ones keep four of the pass's eight prompts
and half of its tokens, after a pass of their own; QW_PREFILL_LONG_PARALLEL=0 is the old rule, one long prompt at a time.

  python3 server/tests/test_prefill_parallel.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from qwserve.scheduler import Request, Scheduler  # noqa: E402
from test_prefill_order import FakeEngine  # noqa: E402


def request(name, n):
    # every prompt its own tokens: prompts with a common start wait for each other at admission (_shares_prefill)
    r = Request([2 + sum(map(ord, name)) % 1000] * n, {"max_tokens": 100}, lambda *a: None, set())
    r.name = name
    return r


def sched(names_sizes, slots=16):
    e = FakeEngine()
    e.capacity = [262144] * slots
    e.busy = [False] * slots
    s = Scheduler(e)
    s.waiting = [request(n, k) for n, k in names_sizes]
    s._admit()
    return s, e


def main():
    fails = 0

    def check(what, got, want):
        nonlocal fails
        fails += got != want
        print(f"{what}: {got} {'ok' if got == want else f'FAIL (want {want})'}")

    def shape(s):
        return [(r.name, n) for r, n in s._batch()]

    s, _ = sched([(f"L{i}", 30000) for i in range(3)])
    check("3 long prompts, nothing else", shape(s), [("L0", 1280), ("L1", 1280), ("L2", 1280)])
    s.shutdown(1)

    s, _ = sched([(f"L{i}", 30000) for i in range(10)])
    check("10 long prompts: the first 8 share the pass", shape(s), [(f"L{i}", 512) for i in range(8)])
    s.shutdown(1)

    s, _ = sched([(f"L{i}", 30000) for i in range(6)] + [(f"s{i}", 100) for i in range(5)])
    check("6 long, 5 short: first a pass for the short ones alone", shape(s), [(f"s{i}", 100) for i in range(5)])
    check("... and if they are still there in the next pass: 4 of them whole, 4 long ones share the rest", shape(s),
          [(f"s{i}", 100) for i in range(4)] + [(f"L{i}", 768) for i in range(4)])
    check("... then the short ones' own pass again", shape(s), [(f"s{i}", 100) for i in range(5)])
    s.shutdown(1)

    s, _ = sched([(f"L{i}", 30000) for i in range(4)] + [(f"m{i}", 1500) for i in range(3)])
    check("3 short prompts of 1,500 fit their own pass only in part", shape(s), [("m0", 1500), ("m1", 1500)])
    check("... together with long ones they leave them half the pass", shape(s),
          [("m0", 1500)] + [(f"L{i}", 512) for i in range(4)])
    s.shutdown(1)

    s, _ = sched([("L0", 30000), ("L1", 6000)])
    s.active = [request("decoding", 5)]  # others decode: a pass is QW_PREFILL_PIECE (2048) tokens
    check("while others decode the piece is shared", shape(s), [("L0", 1024), ("L1", 1024)])
    s.active = []
    s.shutdown(1)

    s, _ = sched([(f"L{i}", 30000) for i in range(3)] + [("s0", 100)])
    s.LONG_PARALLEL = 0
    check("QW_PREFILL_LONG_PARALLEL=0: the short one alone", shape(s), [("s0", 100)])
    s.prefilling = [r for r in s.prefilling if r.name != "s0"]
    check("... then one long prompt at a time", shape(s), [("L0", 4096)])
    s.shutdown(1)

    # end to end: 5 long prompts of different lengths all get in, the passes never exceed the chunk or 8 prompts
    s, e = sched([(f"L{i}", 9000 + 3000 * i) for i in range(5)])
    while s.prefilling:
        s._prefill()
    check("every pass within a chunk and 8 prompts", all(sum(n for _, n in c) <= 4096 and len(c) <= 8 for c in e.calls), True)
    check("all five decoding", sorted(r.name for r in s.active), [f"L{i}" for i in range(5)])
    check("tokens prefilled", sum(min(n, 10 ** 9) for c in e.calls for _, n in c) >= sum(9000 + 3000 * i for i in range(5)), True)
    check("the first passes take all five", [len(c) for c in e.calls[:3]], [5, 5, 5])
    s.shutdown(1)

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
