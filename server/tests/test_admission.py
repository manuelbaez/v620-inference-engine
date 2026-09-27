#!/usr/bin/env python3
"""Slot admission order (no GPU): a request waiting for a big slot does not
hold up later ones that fit the free slots, and gets the big slot first once
it frees.

  python3 server/tests/test_admission.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve.scheduler import Request, Scheduler  # noqa: E402


class FakeEngine:
    """Slot bookkeeping like Session::acquire (the smallest free slot that fits)."""
    has_mtp = False
    prefill_chunk = 8192

    def __init__(self, capacity):
        self.capacity = capacity
        self.max_tokens = max(capacity)
        self.busy = [False] * len(capacity)

    def cache_stats(self):
        return None

    def acquire(self, tokens, max_new, media=None):
        fits = [i for i, c in enumerate(self.capacity) if not self.busy[i] and c >= len(tokens) + max_new]
        if not fits:
            return -1
        slot = min(fits, key=lambda i: self.capacity[i])
        self.busy[slot] = True
        return slot

    def prefetch(self, slot, tokens, media=None):
        return False  # nothing on disk

    def set_stop_tokens(self, slot, ids):
        pass

    def begin_prompt(self, slot, tokens, media=None):
        return 0, None


def request(name, n):
    r = Request([1] * n, {"max_tokens": 1000}, lambda *a: None, set())
    r.name = name
    return r


def main():
    fails = 0

    def check(what, got, want):
        nonlocal fails
        fails += got != want
        print(f"{what}: {got} {'ok' if got == want else f'FAIL (want {want})'}")

    e = FakeEngine([262144, 131072, 32768, 32768])
    s = Scheduler(e)
    s.waiting = [request("big1", 240000), request("big2", 240000), request("small1", 1000),
                 request("mid", 100000), request("small2", 1000), request("small3", 1000)]
    s._admit()
    check("admitted (slot sizes)", {r.name: e.capacity[r.slot] for r in s.prefilling},
          {"big1": 262144, "small1": 32768, "mid": 131072, "small2": 32768})
    check("waiting", [r.name for r in s.waiting], ["big2", "small3"])

    # big1 ends: its slot goes to big2, the oldest waiting request that fits it
    big1 = s.prefilling.pop(0)
    e.busy[big1.slot] = False
    s._admit()
    check("big2 gets the 256k slot", [(r.name, e.capacity[r.slot]) for r in s.prefilling if r.name == "big2"],
          [("big2", 262144)])
    check("still waiting", [r.name for r in s.waiting], ["small3"])
    s.shutdown(1)
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
