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


class SuspendEngine(FakeEngine):
    """FakeEngine with the prefix cache's part in a suspension: suspend() keeps how far the slot's prompt is, and
    begin_prompt gives that back to the same prompt."""

    def __init__(self):
        super().__init__()
        self.total, self.key, self.saved, self.suspends = {}, {}, {}, []

    def begin_prompt(self, slot, tokens, media=None):
        key = (tokens[0], len(tokens))
        cached = self.saved.get(key, 0)
        self.total[slot], self.key[slot] = len(tokens), key
        self.left[slot] = len(tokens) - cached
        return cached, None

    def suspend(self, slot):
        done = self.total[slot] - self.left[slot]
        self.saved[self.key[slot]] = done
        self.suspends.append(slot)
        return done


def sched(names_sizes, slots=16, engine=FakeEngine):
    e = engine()
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
    check("10 long prompts share the pass", shape(s), [(f"L{i}", 256) for i in range(10)])
    s.shutdown(1)

    s, _ = sched([(f"L{i}", 30000) for i in range(6)] + [(f"s{i}", 100) for i in range(5)])
    check("6 long, 5 short: first a pass for the short ones alone", shape(s), [(f"s{i}", 100) for i in range(5)])
    check("... and if they are still there in the next pass: together, the long ones sharing the rest", shape(s),
          [(f"s{i}", 100) for i in range(5)] + [(f"L{i}", 512) for i in range(6)])
    check("... then the short ones' own pass again", shape(s), [(f"s{i}", 100) for i in range(5)])
    s.shutdown(1)

    s, _ = sched([(f"L{i}", 30000) for i in range(14)] + [(f"s{i}", 50) for i in range(15)], slots=32)
    shape(s)  # the short ones' own pass
    check("14 long, 15 short: the long ones keep 4 of the 16 prompts of a pass", [n for n, _ in shape(s)],
          [f"s{i}" for i in range(12)] + [f"L{i}" for i in range(4)])
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

    # KV spill (production: 18,432 tokens of each slot in VRAM, a staging buffer of 243,712 spilled positions): long
    # prompts share a pass only as far as their spilled KV fits the buffer (the rest would read host RAM, half speed)
    def spill_sched(done):
        s, e = sched([(f"L{i}", 95000) for i in range(8)])
        e.vram_tokens, e.stage_tokens = [18432] * 16, 243712
        for r in s.prefilling:
            r.left = 95000 - done
        return s
    for done, n in ((0, 8), (40000, 8), (60000, 5), (90000, 3)):
        s = spill_sched(done)
        b = s._batch()
        check(f"spill: 8 prompts {done} tokens in: {n} share the pass", len(b), n)
        check(f"... and their spilled KV fits the buffer", sum(s._spilled(r, k) for r, k in b) <= 243712, True)
        s.shutdown(1)
    # a short continuation of a 200k conversation (cached but the last 1,000 tokens) fills most of the buffer: the
    # long prompts still get one place in the pass, never none
    s, e = sched([("c", 200000)] + [(f"L{i}", 95000) for i in range(8)])
    e.vram_tokens, e.stage_tokens = [18432] * 16, 243712
    for r in s.prefilling:
        r.left = 1000 if r.name == "c" else 95000 - 60000
    s._short_pass = True  # (the pass after the short ones' own)
    check("spill: a deep short continuation and 8 long prompts", [r.name for r, _ in s._batch()], ["c", "L0"])
    s.shutdown(1)

    # QW_SHORT_PASSES=4: a prompt of up to four passes is a short one
    Scheduler.SHORT_PASSES = 4
    s, _ = sched([(f"L{i}", 30000) for i in range(3)] + [("m0", 10000), ("m1", 3000)])
    check("SHORT_PASSES=4: prompts of 10,000 and 3,000 tokens are short: their own pass, the first a piece", shape(s),
          [("m0", 4096)])
    check("... then with the long ones, which keep half the pass", shape(s),
          [("m0", 2048), ("L0", 512), ("L1", 512), ("L2", 512)])
    for r in s.prefilling:
        if r.name == "m0":
            r.left = 500
    check("... and whole once they fit", shape(s), [("m0", 500), ("m1", 3000)])
    s.shutdown(1)
    s, e = sched([(f"L{i}", 30000) for i in range(16)], engine=SuspendEngine)
    s.waiting.append(request("m", 12000))
    s._admit()
    check("SHORT_PASSES=4: a 12,000-token request takes a long prompt's slot", ("m" in [r.name for r in s.prefilling], [r.name for r in s.waiting]),
          (True, ["L15"]))
    s.shutdown(1)
    Scheduler.SHORT_PASSES = 1

    # sixteen long prompts and sixteen slots: a short request takes the slot of the long prompt admitted last
    s, e = sched([(f"L{i}", 30000) for i in range(16)], engine=SuspendEngine)
    check("16 long prompts, 16 slots: all prefill", (len(s.prefilling), len(s.waiting)), (16, 0))
    for r in s.prefilling:  # three passes of 256 tokens each have gone in
        r.left -= 768
        e.left[r.slot] -= 768
    starts = []
    victim = s.prefilling[-1]
    victim.emit = lambda kind, payload=None: starts.append(kind)
    short = request("short", 50)
    s.waiting.append(short)
    s._admit()
    check("a short request arrives: the last long prompt gives it its slot and waits first in line",
          ([r.name for r in s.waiting], "short" in [r.name for r in s.prefilling], len(s.prefilling)), (["L15"], True, 16))
    check("what it had prefilled went to the prefix cache", e.saved[(victim.prompt[0], 30000)], 768)
    check("the short request's pass is its own", shape(s), [("short", 50)])
    s.prefilling.remove(short)  # it is prefilled and decoded
    s._finish(short, "stop")
    s._admit()
    check("when the short request is done the long prompt continues where it was",
          ([r.name for r in s.waiting], victim.left, victim.suspended, starts.count("start")), ([], 30000 - 768, 1, 0))
    for r in list(s.prefilling[4:]):  # only four long prompts left prefilling, the other slots decode
        s.prefilling.remove(r)
        s.active.append(r)
    s.waiting.append(request("short2", 50))
    s._admit()
    check("four long prompts keep their slots: the next short request waits", [r.name for r in s.waiting], ["short2"])
    s.active = []
    s.shutdown(1)

    # with slots kept free instead (QW_SHORT_SLOTS=4, no preemption): four of sixteen long prompts wait
    s, e = sched([(f"L{i}", 30000) for i in range(16)])
    s.shutdown(1)
    Scheduler.SHORT_SLOTS = 4
    s, e = sched([(f"L{i}", 30000) for i in range(16)])
    check("16 long prompts, 16 slots: 12 prefill, 4 wait for a slot", (len(s.prefilling), len(s.waiting)), (12, 4))
    s.waiting.append(request("short", 50))
    s._admit()
    check("a short request sent after them gets a slot at once", "short" in [r.name for r in s.prefilling], True)
    check("... and the first pass is its own", shape(s), [("short", 50)])
    for r in [r for r in s.prefilling if r.name == "short"]:  # the short request is done
        s.prefilling.remove(r)
        e.release(r.slot)
    done = s.prefilling[0]
    done.left = 100  # one long prompt nearly in: its place among the long ones is free
    s._admit()
    check("when a long prompt is nearly in, the next one starts", (len(s.prefilling), len(s.waiting)), (13, 3))
    s.SHORT_SLOTS = 0
    s._admit()
    check("QW_SHORT_SLOTS=0: long prompts take every slot", (len(s.prefilling), len(s.waiting)), (16, 0))
    s.shutdown(1)
    Scheduler.SHORT_SLOTS = 0

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
