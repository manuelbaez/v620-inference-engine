"""Continuous batching over the engine's sequence slots."""

import os
import sys
import threading
import time

from .engine import Sampling


class Request:
    def __init__(self, prompt_ids, body, emit, eos_ids, media=None):
        self.prompt = prompt_ids
        self.media = media or []  # vision.Media items of the prompt
        self.emit = emit
        self.cancelled = threading.Event()
        mt = body.get("max_completion_tokens") or body.get("max_tokens")
        self.max_new = int(mt) if mt else None
        s = Sampling()
        s.temperature = float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0)
        s.top_p = float(body.get("top_p") or 1.0)
        s.top_k = int(body.get("top_k") or 0)
        s.min_p = float(body.get("min_p") or 0.0)
        s.presence_penalty = float(body.get("presence_penalty") or 0.0)
        s.frequency_penalty = float(body.get("frequency_penalty") or 0.0)
        s.repetition_penalty = float(body.get("repetition_penalty") or 1.0)
        s.seed = int(body.get("seed") or 0)
        self.sampling = s
        self.want_top = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        self.end_ids = set(body.get("stop_token_ids") or [])
        if not body.get("ignore_eos"):
            self.end_ids |= eos_ids
        self.slot = -1
        self.limit = 0      # max new tokens in the slot it got
        self.generated = 0
        self.next = None    # sampled token not yet fed to the engine
        self.cached = 0     # prompt tokens reused from the prefix cache
        self.keep = None    # buffers the engine reads while the prompt is prefilled
        self.t_admit = self.t_first = None  # prefill start, first token (time.time())

    def timings(self):
        """llama.cpp-style per-request timings (llama-swap's activity log reads these)."""
        now = time.time()
        prompt_n = len(self.prompt) - self.cached
        prompt_ms = (self.t_first - self.t_admit) * 1e3
        predicted_ms = (now - self.t_first) * 1e3
        decoded = max(self.generated - 1, 0)  # the first token comes from the prompt's logits
        return {
            "cache_n": self.cached,
            "prompt_n": prompt_n,
            "prompt_ms": round(prompt_ms, 3),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else 0.0,
            "prompt_per_second": round(prompt_n / prompt_ms * 1e3, 2) if prompt_n and prompt_ms > 0 else 0.0,
            "predicted_n": self.generated,
            "predicted_ms": round(predicted_ms, 3),
            "predicted_per_token_ms": round(predicted_ms / decoded, 3) if decoded else 0.0,
            "predicted_per_second": round(decoded / predicted_ms * 1e3, 2) if decoded and predicted_ms > 0 else 0.0,
        }


class Scheduler:
    """Continuous batching on one thread: admits queued requests into free
    slots, prefills them, then advances every active request by one token per
    batched engine step. While requests are decoding, prefills advance one
    piece (QW_PREFILL_PIECE tokens, default 2048) at a time, and between pieces
    the running requests decode for QW_DECODE_SHARE (default 0.25) of the
    piece's time (at least one step): a long prompt never stalls the others for
    more than one piece, and they keep a share of the GPUs while it goes in."""

    DEFAULT_RESERVE = 2048  # room reserved when the request sets no max_tokens
    PREFILL_PIECE = int(os.environ.get("QW_PREFILL_PIECE", "2048"))
    DECODE_SHARE = float(os.environ.get("QW_DECODE_SHARE", "0.25"))

    def __init__(self, engine, mtp_drafts=3):
        self.e = engine
        self.k = mtp_drafts if engine.has_mtp else 0
        self.stats = {"steps": 0, "rows": 0, "tokens": 0, "time": 0.0}  # decode steps, request-steps, emitted tokens
        self.cache_stats = engine.cache_stats()  # refreshed by the scheduler thread (the engine is not thread-safe)
        self.cv = threading.Condition()
        self.queue = []    # submitted, guarded by cv
        self.waiting = []  # admitted in FIFO order by the scheduler thread
        self.prefilling = []  # in a slot, prompt partly prefilled
        self.active = []
        self.stopping = False
        self.thread = threading.Thread(target=self._loop, daemon=True, name="qw-scheduler")
        self.thread.start()

    def shutdown(self, timeout=30):
        """Stops after the current step; in-flight requests end with an error."""
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        self.thread.join(timeout)
        for r in self.waiting + self.prefilling + self.active + self.queue:
            r.emit("error", "server shutting down")

    def submit(self, req):
        n = len(req.prompt)
        if n >= self.e.max_tokens:
            raise ValueError(f"prompt is {n} tokens; the context holds {self.e.max_tokens}")
        with self.cv:
            self.queue.append(req)
            self.cv.notify()

    def _finish(self, r, reason):
        if r.t_first is not None:
            r.emit("timings", r.timings())
        r.emit("end", reason)
        if r.slot >= 0:
            self.e.release(r.slot)
            r.slot = -1

    def _took(self, r, tid, lp, top):
        """Handles a sampled token; returns False when the request is done."""
        r.generated += 1
        if tid in r.end_ids:
            r.emit("eos", tid)  # counted in completion_tokens, like vLLM
            self._finish(r, "stop")
            return False
        r.emit("token", (tid, lp, top))
        if r.generated >= r.limit:
            self._finish(r, "length")
            return False
        r.next = tid
        return True

    def _admit(self):
        while self.waiting:
            r = self.waiting[0]
            if r.cancelled.is_set():
                self.waiting.pop(0)
                r.emit("end", "abort")
                continue
            n = len(r.prompt)
            need = r.max_new if r.max_new else self.DEFAULT_RESERVE
            need = max(1, min(need, self.e.max_tokens - n))
            slot = self.e.acquire(r.prompt, need, r.media)
            if slot < 0:
                return  # FIFO: wait for a slot to free up
            self.waiting.pop(0)
            r.slot = slot
            try:
                room = self.e.capacity[slot] - n
                r.limit = min(r.max_new, room) if r.max_new else room
                self.e.set_stop_tokens(slot, sorted(r.end_ids))
                r.t_admit = time.time()
                r.cached, r.keep = self.e.begin_prompt(slot, r.prompt, r.media)
                self.cache_stats = self.e.cache_stats()
                r.emit("start", r.cached)
            except Exception as ex:  # noqa: BLE001
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            self.prefilling.append(r)

    def _prefill(self):
        """Advances the oldest prefill: by one piece while others decode (then
        returns so they get steps), to the end when nothing is decoding.
        Returns the seconds spent."""
        t0 = time.time()
        while self.prefilling:
            r = self.prefilling[0]
            if r.cancelled.is_set():
                self.prefilling.pop(0)
                self._finish(r, "abort")
                continue
            try:
                done = self.e.prefill_some(r.slot, self.PREFILL_PIECE if self.active else 1 << 40)
                if not done:
                    return time.time() - t0
                self.prefilling.pop(0)
                self.cache_stats = self.e.cache_stats()
                top = self.e.top_logprobs(-1, r.want_top) if r.want_top else None
                tid, lp = self.e.sample_prompt(r.slot, r.sampling)
                r.t_first = time.time()
                r.keep = None
            except Exception as ex:  # noqa: BLE001
                if self.prefilling and self.prefilling[0] is r:
                    self.prefilling.pop(0)
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            if self._took(r, tid, lp, top):
                self.active.append(r)
            if self.active:
                return time.time() - t0  # let the running requests take steps before the next prefill
        return time.time() - t0

    def _step(self):
        for r in [r for r in self.active if r.cancelled.is_set()]:
            self.active.remove(r)
            self._finish(r, "abort")
        if not self.active:
            return
        batch = self.active[:16]
        self.active = self.active[16:] + batch  # round-robin past 16 requests (at most num_slots anyway)
        t0 = time.time()
        res = self.e.generate([(r.slot, r.next, r.limit - r.generated, r.sampling) for r in batch], self.k)
        st = self.stats
        st["time"] += time.time() - t0
        st["steps"] += 1
        st["rows"] += len(batch)
        st["tokens"] += sum(len(t) for t, _, _, _ in res)
        for r, (toks, lps, first, _) in zip(batch, res):
            for j, (tid, lp) in enumerate(zip(toks, lps)):
                top = self.e.top_logprobs(first + j, r.want_top) if r.want_top else None
                if not self._took(r, tid, lp, top):  # the engine stopped at the same token
                    self.active.remove(r)
                    break

    def _loop(self):
        while True:
            with self.cv:
                while (not self.queue and not self.waiting and not self.prefilling and not self.active
                       and not self.stopping):
                    self.cv.wait()
                if self.stopping:
                    return
                self.waiting += self.queue  # the queue is shared; waiting is this thread's
                self.queue = []
            try:
                self._admit()
                spent = self._prefill()
                t0 = time.time()
                self._step()
                while self.prefilling and self.active and time.time() - t0 < self.DECODE_SHARE * spent:
                    self._step()
            except Exception as ex:  # noqa: BLE001  engine failure: fail everything in flight
                print(f"scheduler error: {ex}", file=sys.stderr, flush=True)
                for r in self.prefilling + self.active:
                    r.emit("error", str(ex))
                    try:
                        self._finish(r, "abort")
                    except Exception:  # noqa: BLE001
                        pass
                self.prefilling = []
                self.active = []
