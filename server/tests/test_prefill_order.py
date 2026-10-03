#!/usr/bin/env python3
"""Prefill order (no GPU): a long prompt prefilling with nothing decoding goes chunk by
chunk and yields as soon as a request arrives, and the short newcomer is prefilled before
the rest of the long prompt.

  python3 server/tests/test_prefill_order.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve.scheduler import Request, Scheduler  # noqa: E402


class FakeEngine:
    """Records prefill calls; each slot's prompt is done once all its tokens went in."""
    has_mtp = False
    prefill_chunk = 4096

    def __init__(self):
        self.capacity = [262144, 32768]
        self.max_tokens = max(self.capacity)
        self.busy = [False] * len(self.capacity)
        self.left = {}
        self.calls = []
        self.on_call = None  # hook run after each prefill call

    def cache_stats(self):
        return None

    def slot_timing(self, slot):
        return {"restore_s": 0.0, "prefill_s": 0.0, "save_s": 0.0, "pin_wait_s": 0.0}

    def acquire(self, tokens, max_new, media=None):
        fits = [i for i, c in enumerate(self.capacity) if not self.busy[i] and c >= len(tokens) + max_new]
        if not fits:
            return -1
        slot = min(fits, key=lambda i: self.capacity[i])
        self.busy[slot] = True
        return slot

    def prefetch(self, slot, tokens, media=None):
        return False

    def set_stop_tokens(self, slot, ids):
        pass

    def begin_prompt(self, slot, tokens, media=None):
        self.left[slot] = len(tokens)
        return 0, None

    def prefill_some(self, slot, n):
        return self.prefill_batch([(slot, n)])[0]

    def prefill_batch(self, pairs):
        self.calls.append(list(pairs))
        done = []
        for slot, n in pairs:
            self.left[slot] -= min(n, self.left[slot])
            done.append(self.left[slot] == 0)
        if self.on_call:
            self.on_call()
        return done

    def top_logprobs_prompt(self, slot, k):
        return None

    def sample_prompt(self, slot, sampling):
        return 7, 0.0

    def release(self, slot):
        self.busy[slot] = False


def request(name, n):
    r = Request([1] * n, {"max_tokens": 100}, lambda *a: None, set())
    r.name = name
    return r


def main():
    fails = 0

    def check(what, got, want):
        nonlocal fails
        fails += got != want
        print(f"{what}: {got} {'ok' if got == want else f'FAIL (want {want})'}")

    e = FakeEngine()
    s = Scheduler(e)
    long_r, short_r = request("long", 65536), request("short", 11)
    s.waiting = [long_r]
    s._admit()

    def arrive():  # the short request arrives during the long prompt's second chunk
        if len(e.calls) == 2:
            s.queue.append(short_r)
    e.on_call = arrive
    s._prefill()
    check("chunks before yielding", [c[0][1] for c in e.calls], [4096, 4096])
    check("long prompt still prefilling", [r.name for r in s.prefilling], ["long"])

    s.waiting += s.queue
    s.queue = []
    s._admit()
    e.on_call = None
    n = len(e.calls)
    s._prefill()  # the short one goes first, then the long one continues between its steps
    check("next pass", [(long_r.slot, short_r.slot).index(slot) for slot, _ in e.calls[n]], [1])
    check("short request decoding", [r.name for r in s.active], ["short"])
    check("then returns so it decodes (no further prefill call)", len(e.calls) - n, 1)
    s.shutdown(1)
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
