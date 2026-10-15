#!/usr/bin/env python3
"""Long prefills with nothing decoding keep the dashboard and the stats line current (no GPU): _prefill runs pass after
pass to the end in that case, and before 2026-10-15 the scheduler sampled the dashboard and printed its stats line only
after it returned, so eight 95k-token prompts showed nothing moving for six minutes.

  python3 server/tests/test_prefill_observe.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fixtures import FakeEngine  # noqa: E402
from qwserve.scheduler import Request, Scheduler  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += not ok


def main():
    e = FakeEngine(capacity=(262144,) * 4, prefill_s=0.05)  # 0.05 s per pass
    s = Scheduler(e)
    s.waiting = [Request([3 + i] * (4096 * 30), {"max_tokens": 10}, lambda *a: None, set()) for i in range(2)]
    s._admit()
    samples = []
    sample = s.metrics.sample
    s.metrics.sample = lambda *a, **k: (samples.append(a[1]), sample(*a, **k))
    s.LOG_INTERVAL = 1.0
    passes = e.prefills
    spent = s._prefill()  # nothing decoding, nothing arriving: to the end, ~60 passes, ~3 s
    # (it returns when the first prompt is done: that one then decodes)
    check(f"the prompts prefilled in one call until one was done ({e.prefills - passes} passes, {spent:.1f} s)",
          s.active and spent >= 2.5)
    check(f"the dashboard was sampled during it, about once a second ({len(samples)} samples)", len(samples) >= 2)
    check("those samples saw the prompts prefilling", samples and samples[0] == 2)
    w = s.metrics._win
    check(f"the time sampled counts as prefill time ({w['prefill_s']:.1f} s of {spent:.1f})",
          0 < w["prefill_s"] <= spent + 0.01)
    check("what is left for the main loop to count is the rest of the call", 0 <= spent - s._observed_s < 1.1)
    s.shutdown(1)
    print("all ok" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
