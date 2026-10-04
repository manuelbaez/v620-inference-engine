#!/usr/bin/env python3
"""Slot admission order (no GPU): a request waiting for a big slot does not
hold up later ones that fit the free slots, and gets the big slot first once
it frees. A request that found no slot is not offered to the engine again
until a slot has been released (acquire hands the engine the whole prompt).

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
        self.acquire_calls = 0

    def cache_stats(self):
        return None

    def release(self, slot):
        self.busy[slot] = False

    def acquire(self, tokens, max_new, media=None):
        self.acquire_calls += 1
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


def request(name, n, prefix=None):
    """n tokens; requests share no prefix (their first token differs) unless given the same `prefix` tokens."""
    first = [2 + sum(map(ord, name)) % 30000]
    tokens = (list(prefix) + first + [1] * n)[:n] if prefix else (first + [1] * n)[:n]
    r = Request(tokens, {"max_tokens": 1000}, lambda *a: None, set())
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

    # nothing was released: the two that found no slot are not offered again
    calls = e.acquire_calls
    for _ in range(5):
        s._admit()
    check("acquire calls while nothing is released", e.acquire_calls - calls, 0)

    # big1 ends: its slot goes to big2, the oldest waiting request that fits it
    big1 = s.prefilling.pop(0)
    s._finish(big1, "stop")
    calls = e.acquire_calls
    s._admit()
    check("acquire calls after a release (both waiting are offered)", e.acquire_calls - calls, 2)
    check("big2 gets the 256k slot", [(r.name, e.capacity[r.slot]) for r in s.prefilling if r.name == "big2"],
          [("big2", 262144)])
    check("still waiting", [r.name for r in s.waiting], ["small3"])
    s.shutdown(1)

    # shared prefix: a request whose prompt starts like one being prefilled waits until that one is past the shared
    # part (it then restores it from the block store), while a request with another prompt is admitted at once
    e = FakeEngine([65536, 65536, 65536, 65536])
    s = Scheduler(e)
    system = [7] * 12000
    s.waiting = [request("a", 13000, system), request("b", 12500, system), request("other", 12500),
                 request("short", 13000, [7] * 1000)]
    s._admit()
    check("shared 12k prefix: first and unrelated ones admitted, a 1k share is not worth waiting",
          sorted(r.name for r in s.prefilling), ["a", "other", "short"])
    check("the second with the same system prompt waits", [r.name for r in s.waiting], ["b"])
    a = next(r for r in s.prefilling if r.name == "a")
    a.left = 13000 - 8192  # a's prefill is at 8,192 of the 12,000 shared tokens
    s._admit()
    check("still waiting inside the shared part", [r.name for r in s.waiting], ["b"])
    a.left = 13000 - 11000  # 1,000 shared tokens to go: fewer than SHARE_MIN, but it already waits for them
    s._admit()
    check("waits until the first is past the whole shared part", [r.name for r in s.waiting], ["b"])
    a.left = 13000 - 12288  # past the shared part
    s._admit()
    check("admitted once the first is past it", sorted(r.name for r in s.prefilling), ["a", "b", "other", "short"])
    s.shutdown(1)
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
