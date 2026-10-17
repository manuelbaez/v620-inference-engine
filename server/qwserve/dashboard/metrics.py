"""Serving metrics collected by the scheduler thread: cumulative counters, a
history of throughput and queue samples, and the recent requests. Throughput
comes two ways: speed, the sum of the rates of all the requests in that phase
(tokens over the wall time during which at least one request was prefilling,
or decoding), and demand (tokens over all wall time, so it drops when the
server is idle). PCIe traffic per card comes from the engine's own byte
counts (pcie_source), as GB/s over each sample's interval."""

import collections
import time

HISTORY_SECONDS = 3600  # one sample per SAMPLE_SECONDS for the last hour
SAMPLE_SECONDS = 5
RECENT_REQUESTS = 100


class Metrics:
    def __init__(self):
        self.started = time.time()
        self.totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "computed_prompt_tokens": 0,
                       "generated_tokens": 0, "errors": 0}
        self.history = collections.deque(maxlen=HISTORY_SECONDS // SAMPLE_SECONDS)
        self.requests = collections.deque(maxlen=RECENT_REQUESTS)
        self.live = {}  # running / prefilling / waiting counts, KV tokens used
        self._win = self._new_window(time.time())
        self._prefilled = self._generated = False  # during the current loop pass
        self.pcie_source = None  # engine.pcie_bytes, called on the scheduler thread (the engine is not thread-safe)
        self.pcie = []  # per card, GB/s by kind over the last sample's interval
        self._pcie_last = None  # (time, the counts then)

    def _pcie_rates(self, now):
        """Per card, GB/s by kind since the previous call (None the first time, or without a source)."""
        if not self.pcie_source:
            return None
        try:
            counts = self.pcie_source()
        except Exception:  # noqa: BLE001  a statistic: never fail the scheduler for it
            return None
        last, self._pcie_last = self._pcie_last, (now, counts)
        if not last or now <= last[0]:
            return None
        dt = now - last[0]
        return [[round((c - l) / dt / 1e9, 3) for c, l in zip(card, prev)] for card, prev in zip(counts, last[1])]

    @staticmethod
    def _spill_totals(pcie):
        """All cards together, GB/s: decode's reads of spilled rows, the rows decode and prefill write to host RAM,
        the prefill staging's copies in and back, and their sum (engine.PCIE_KINDS 4-7)."""
        out = {k: round(sum(c[i] for c in pcie if len(c) > i), 4)
               for k, i in (("kv_read", 4), ("kv_write", 5), ("kv_stage_in", 6), ("kv_stage_out", 7))}
        out["kv_total"] = round(sum(out.values()), 4)
        return out

    @staticmethod
    def _new_window(t):
        return {"t": t, "prefill": 0, "gen": 0, "prefill_s": 0.0, "gen_s": 0.0}

    # --- scheduler side
    def add_prefill(self, n):
        self.totals["computed_prompt_tokens"] += n
        self._win["prefill"] += n
        self._prefilled = True

    def add_generated(self, n=1):
        self.totals["generated_tokens"] += n
        self._win["gen"] += n
        self._generated = True

    def tick(self, seconds, prefilling, running):
        """One pass of the scheduler loop took `seconds`: it counts as prefill time if a request
        was prefilling (tokens computed, or requests still in that phase), as decode time likewise."""
        if self._prefilled or prefilling:
            self._win["prefill_s"] += seconds
        if self._generated or running:
            self._win["gen_s"] += seconds
        self._prefilled = self._generated = False

    def add_admitted(self, prompt, cached):
        self.totals["prompt_tokens"] += prompt
        self.totals["cached_tokens"] += cached

    def add_request(self, rid, reason, prompt, cached, generated, t_arrive, t_admit, t_first, t_end, slot):
        self.totals["requests"] += 1
        if reason in ("error", "abort"):
            self.totals["errors"] += reason == "error"
        self.requests.append({
            "id": rid, "finish": reason, "prompt": prompt, "cached": cached, "generated": generated, "slot": slot,
            "end": t_end,
            "queue_s": round(t_admit - t_arrive, 1) if t_admit and t_arrive else None,
            "ttft_s": round(t_first - t_arrive, 1) if t_first and t_arrive else None,
            "total_s": round(t_end - t_arrive, 1) if t_arrive else None,
            "decode_tps": round((generated - 1) / (t_end - t_first), 1) if t_first and generated > 1 and t_end > t_first
            else None,
        })

    def sample(self, running, prefilling, waiting, kv_used, kv_capacity, force=False, slots=None):
        """Records a history point every SAMPLE_SECONDS (called often by the scheduler loop). slots: per slot,
        the tokens it holds and whether a request uses it (None: as last reported, none in use)."""
        now = time.time()
        if slots is None:
            slots = [{"tokens": s["tokens"], "busy": False} for s in self.live.get("slots", [])]
        self.live = {"running": running, "prefilling": prefilling, "waiting": waiting, "kv_used_tokens": kv_used,
                     "kv_capacity_tokens": kv_capacity, "slots": slots}
        dt = now - self._win["t"]
        if dt < SAMPLE_SECONDS and not force:
            return
        w = self._win
        pcie = self._pcie_rates(now)
        if pcie is not None:
            self.pcie = pcie
        self.history.append({
            "t": round(now, 1),
            # all requests together while in that phase; None when none was (a gap in the chart)
            "prompt_tps": round(w["prefill"] / w["prefill_s"], 1) if w["prefill_s"] > 0 else None,
            "gen_tps": round(w["gen"] / w["gen_s"], 1) if w["gen_s"] > 0 else None,
            "prompt_demand_tps": round(w["prefill"] / dt, 1), "gen_demand_tps": round(w["gen"] / dt, 1),
            "running": running, "waiting": waiting + prefilling,
            "kv_pct": round(100 * kv_used / max(1, kv_capacity), 1),
            # both directions per card (GB/s), and the spilled KV's traffic with host RAM summed over the cards
            # (GB/s) by kind; the kinds per card are in snapshot()["pcie"]
            **({f"pcie{r}": round(sum(c), 3) for r, c in enumerate(pcie)} if pcie else {}),
            **(self._spill_totals(pcie) if pcie else {})})
        self._win = self._new_window(now)

    def idle(self):
        """The loop is about to sleep: close the current window, restart it on wake."""
        self.sample(0, 0, 0, self.live.get("kv_used_tokens", 0), self.live.get("kv_capacity_tokens", 1), force=True)

    def wake(self):
        self._win = self._new_window(time.time())
        self._pcie_rates(time.time())  # the next sample's rate starts now, not at the idle sample

    # --- HTTP side
    def snapshot(self):
        reqs = list(self.requests)
        done = [r for r in reqs if r["finish"] in ("stop", "length", "tool_calls")]

        def avg(key):
            v = [r[key] for r in done if r.get(key) is not None]
            return round(sum(v) / len(v), 1) if v else None
        return {"uptime_s": round(time.time() - self.started), "totals": dict(self.totals), "live": dict(self.live),
                "averages": {"requests": len(done), "ttft_s": avg("ttft_s"), "total_s": avg("total_s"),
                             "decode_tps": avg("decode_tps"), "queue_s": avg("queue_s")},
                "history": list(self.history), "recent": reqs[-30:][::-1], "pcie": self.pcie}
