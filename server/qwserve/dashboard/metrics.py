"""Serving metrics collected by the scheduler thread: cumulative counters, a
history of throughput and queue samples, and the recent requests. Throughput
comes two ways: speed, the sum of the rates of all the requests in that phase
(tokens over the wall time during which at least one request was prefilling,
or decoding), and demand (tokens over all wall time, so it drops when the
server is idle)."""

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

    def sample(self, running, prefilling, waiting, kv_used, kv_capacity, force=False):
        """Records a history point every SAMPLE_SECONDS (called often by the scheduler loop)."""
        now = time.time()
        self.live = {"running": running, "prefilling": prefilling, "waiting": waiting, "kv_used_tokens": kv_used,
                     "kv_capacity_tokens": kv_capacity}
        dt = now - self._win["t"]
        if dt < SAMPLE_SECONDS and not force:
            return
        w = self._win
        self.history.append({
            "t": round(now, 1),
            # all requests together while in that phase; None when none was (a gap in the chart)
            "prompt_tps": round(w["prefill"] / w["prefill_s"], 1) if w["prefill_s"] > 0 else None,
            "gen_tps": round(w["gen"] / w["gen_s"], 1) if w["gen_s"] > 0 else None,
            "prompt_demand_tps": round(w["prefill"] / dt, 1), "gen_demand_tps": round(w["gen"] / dt, 1),
            "running": running, "waiting": waiting + prefilling,
            "kv_pct": round(100 * kv_used / max(1, kv_capacity), 1)})
        self._win = self._new_window(now)

    def idle(self):
        """The loop is about to sleep: close the current window, restart it on wake."""
        self.sample(0, 0, 0, self.live.get("kv_used_tokens", 0), self.live.get("kv_capacity_tokens", 1), force=True)

    def wake(self):
        self._win = self._new_window(time.time())

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
                "history": list(self.history), "recent": reqs[-30:][::-1]}
