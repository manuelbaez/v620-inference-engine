"""Continuous batching over the engine's sequence slots."""

import sys
import threading
import time

from .engine import Sampling


class Request:
    def __init__(self, prompt_ids, body, emit, eos_ids):
        self.prompt = prompt_ids
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


class Scheduler:
    """Continuous batching on one thread: admits queued requests into free
    slots (prefill + first token), then advances every active request by one
    token per batched engine step."""

    DEFAULT_RESERVE = 2048  # room reserved when the request sets no max_tokens

    def __init__(self, engine, mtp_drafts=3):
        self.e = engine
        self.k = mtp_drafts if engine.has_mtp else 0
        self.stats = {"steps": 0, "rows": 0, "tokens": 0, "time": 0.0}  # decode steps, request-steps, emitted tokens
        self.cv = threading.Condition()
        self.queue = []    # submitted, guarded by cv
        self.waiting = []  # admitted in FIFO order by the scheduler thread
        self.active = []
        threading.Thread(target=self._loop, daemon=True, name="qw-scheduler").start()

    def submit(self, req):
        n = len(req.prompt)
        if n >= self.e.max_tokens:
            raise ValueError(f"prompt is {n} tokens; the context holds {self.e.max_tokens}")
        with self.cv:
            self.queue.append(req)
            self.cv.notify()

    def _finish(self, r, reason):
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
            slot = self.e.acquire(r.prompt, need)
            if slot < 0:
                return  # FIFO: wait for a slot to free up
            self.waiting.pop(0)
            r.slot = slot
            try:
                room = self.e.capacity[slot] - n
                r.limit = min(r.max_new, room) if r.max_new else room
                self.e.set_stop_tokens(slot, sorted(r.end_ids))
                cached = self.e.set_prompt(slot, r.prompt)
                r.emit("start", cached)
                top = self.e.top_logprobs(-1, r.want_top) if r.want_top else None
                tid, lp = self.e.sample_prompt(slot, r.sampling)
            except Exception as ex:  # noqa: BLE001
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            if self._took(r, tid, lp, top):
                self.active.append(r)

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
                while not self.queue and not self.waiting and not self.active:
                    self.cv.wait()
                self.waiting += self.queue  # the queue is shared; waiting is this thread's
                self.queue = []
            try:
                self._admit()
                self._step()
            except Exception as ex:  # noqa: BLE001  engine failure: fail everything in flight
                print(f"scheduler error: {ex}", file=sys.stderr, flush=True)
                for r in self.active:
                    r.emit("error", str(ex))
                    try:
                        self._finish(r, "abort")
                    except Exception:  # noqa: BLE001
                        pass
                self.active = []
