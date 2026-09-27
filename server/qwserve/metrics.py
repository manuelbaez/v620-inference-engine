"""Serving metrics for the dashboard (GET /) and its JSON (GET /metrics.json):
cumulative counters, a history of throughput and queue samples, and the recent
requests. Written by the scheduler thread, read by the HTTP handlers (whole
values are replaced, never mutated in place, so a reader sees a consistent
snapshot without locks)."""

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
        self._win = {"t": time.time(), "prefill": 0, "gen": 0}

    # --- scheduler side
    def add_prefill(self, n):
        self.totals["computed_prompt_tokens"] += n
        self._win["prefill"] += n

    def add_generated(self, n=1):
        self.totals["generated_tokens"] += n
        self._win["gen"] += n

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
            "queue_s": round(t_admit - t_arrive, 3) if t_admit and t_arrive else None,
            "ttft_s": round(t_first - t_arrive, 3) if t_first and t_arrive else None,
            "total_s": round(t_end - t_arrive, 3) if t_arrive else None,
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
        self.history.append({"t": round(now, 1), "prompt_tps": round(self._win["prefill"] / dt, 1),
                             "gen_tps": round(self._win["gen"] / dt, 1), "running": running,
                             "waiting": waiting + prefilling, "kv_pct": round(100 * kv_used / max(1, kv_capacity), 2)})
        self._win = {"t": now, "prefill": 0, "gen": 0}

    def idle(self):
        """The loop is about to sleep: close the current window, restart it on wake."""
        self.sample(0, 0, 0, self.live.get("kv_used_tokens", 0), self.live.get("kv_capacity_tokens", 1), force=True)

    def wake(self):
        self._win = {"t": time.time(), "prefill": 0, "gen": 0}

    # --- HTTP side
    def snapshot(self):
        reqs = list(self.requests)
        done = [r for r in reqs if r["finish"] in ("stop", "length", "tool_calls")]

        def avg(key):
            v = [r[key] for r in done if r.get(key) is not None]
            return round(sum(v) / len(v), 3) if v else None
        return {"uptime_s": round(time.time() - self.started), "totals": dict(self.totals), "live": dict(self.live),
                "averages": {"requests": len(done), "ttft_s": avg("ttft_s"), "total_s": avg("total_s"),
                             "decode_tps": avg("decode_tps"), "queue_s": avg("queue_s")},
                "history": list(self.history), "recent": reqs[-30:][::-1]}
