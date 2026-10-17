#!/usr/bin/env python3
"""The dashboard's PCIe traffic per card (no GPU, no checkpoint): the engine's byte counts become GB/s over each
sample's interval, per card and kind, with both directions summed per card for the chart and the spilled KV's kinds
summed over the cards; the first sample and an idle stretch give no rate.

  python3 server/tests/test_dashboard_pcie.py
"""

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve.dashboard.metrics import SAMPLE_SECONDS, Metrics  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += not ok


def main():
    counts = [[0] * 9 for _ in range(4)]
    m = Metrics()
    m.pcie_source = lambda: [tuple(c) for c in counts]
    now = [1000.0]
    with mock.patch("qwserve.dashboard.metrics.time.time", lambda: now[0]):
        m._win = m._new_window(now[0])
        now[0] += SAMPLE_SECONDS
        m.sample(1, 0, 0, 10, 100)
        check("the first sample has no rate", "pcie0" not in m.history[-1] and m.pcie == [])

        counts[0] = [1e9, 2e9, 3e9, 4e9, 5e9, 0, 0, 0, 0]  # over 5 s: 0.2, 0.4, 0.6, 0.8, 1.0 GB/s
        counts[2][3] = 10e9
        counts[1][4:8] = [5e9, 1e9, 2e9, 0.5e9]  # spilled KV on another card: 1.0, 0.2, 0.4, 0.1 GB/s
        counts[3][5] = 1e9
        counts[3][8] = 1.5e9  # prefill's direct reads: 0.3 GB/s
        now[0] += SAMPLE_SECONDS
        m.sample(1, 0, 0, 10, 100)
        h = m.history[-1]
        check("per kind, GB/s over the interval", m.pcie[0] == [0.2, 0.4, 0.6, 0.8, 1.0, 0, 0, 0, 0])
        check("the chart's value is both directions together", h["pcie0"] == 3.0 and h["pcie2"] == 2.0)
        check("the spilled KV counts in its card's total", h["pcie1"] == 1.7 and h["pcie3"] == 0.5)
        check("the spill chart sums the cards per kind",
              h["kv_read"] == 2.0 and h["kv_write"] == 0.4 and h["kv_stage_in"] == 0.4 and h["kv_stage_out"] == 0.1
              and h["kv_prefill_read"] == 0.3)
        check("... and its total is all of them", h["kv_total"] == 3.2)
        check("the snapshot carries the kinds", m.snapshot()["pcie"][2][3] == 2.0)

        now[0] += 1
        m.idle()
        now[0] += 600  # idle: no traffic; the next rate starts at wake, not at the idle sample
        m.wake()
        counts[1][0] = 5e9
        now[0] += SAMPLE_SECONDS
        m.sample(1, 0, 0, 10, 100)
        check("after an idle stretch the rate covers only the time awake", m.history[-1]["pcie1"] == 1.0)
        check("... and the spill chart reads zero", m.history[-1]["kv_total"] == 0)

    m2 = Metrics()  # an engine without the counts
    m2.sample(0, 0, 0, 0, 1, force=True)
    check("no source: no PCIe keys",
          "pcie0" not in m2.history[-1] and "kv_total" not in m2.history[-1] and m2.snapshot()["pcie"] == [])

    def broken():
        raise RuntimeError("engine gone")
    m3 = Metrics()
    m3.pcie_source = broken
    m3.sample(0, 0, 0, 0, 1, force=True)
    check("a failing source does not fail the sample", len(m3.history) == 1)
    print("all ok" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
