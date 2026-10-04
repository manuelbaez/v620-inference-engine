"""Continuous batching over the engine's sequence slots."""

import os
import sys
import threading
import time
import traceback

import numpy as np

from .engine import Sampling, Tokens, c_array
from .dashboard.metrics import Metrics


class _Watched:
    """The engine as the scheduler thread calls it: records the call in progress and since when
    (Scheduler.call), which the watchdog reads to tell a wedged GPU from a long job."""

    def __init__(self, engine, owner):
        self._engine, self._owner = engine, owner

    def __getattr__(self, name):
        attr = getattr(self._engine, name)
        if not callable(attr):
            return attr
        owner = self._owner

        def call(*args, **kwargs):
            owner.call = (name, time.time())
            try:
                return attr(*args, **kwargs)
            finally:
                owner.call = None
        return call


class Request:
    def __init__(self, prompt_ids, body, emit, eos_ids, media=None):
        self.prompt = Tokens(prompt_ids)  # its int32 array is built once, not on every pass
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
        self.no_slot_at = None  # Scheduler.slot_epoch when no slot could take it (retried once one frees)
        self.limit = 0      # max new tokens in the slot it got
        self.generated = 0
        self.next = None    # sampled token not yet fed to the engine
        self.cached = 0     # prompt tokens reused from the prefix cache
        self.left = 0       # prompt tokens still to prefill
        self.think_left = None   # thinking tokens still allowed (None: no cap / not thinking)
        self.think_close = None  # (</think> id, forced closing ids) when capped
        self.forced = []         # tokens to emit next instead of sampling (closing a capped thinking section)
        self.rid = "-"           # request id for the logs (set by the API)
        self.t_arrive = time.time()
        self.keep = None    # buffers the engine reads while the prompt is prefilled
        self.t_admit = self.t_first = None  # slot taken, first token (time.time())
        self.t_begin = None  # the prompt's cached part started to be restored (after any disk load)
        self.timing = None   # the engine's slot_timing() at the first token: restore, prefill (saves, pin wait)
        self.share = {}      # id() of a prefilling request -> length of the prefix this prompt shares with it
        self.t_share = None  # when it first waited for such a request's prefill (Scheduler._shares_prefill)
        self.share_on = None  # id() of the request it is waiting for

    def prompt_np(self):
        """The prompt as an int32 array (a view of the C API's array: no copy)."""
        return np.frombuffer(c_array(self.prompt), dtype=np.int32) if len(self.prompt) else np.zeros(0, np.int32)

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
    more than one piece, and they keep a share of the GPUs while it goes in.
    Prompts waiting together go into one prefill pass while they fit a piece
    (a prefill chunk when nothing decodes), so a burst of short prompts pays
    the pass's fixed cost once (QW_PREFILL_BATCH=0 turns that off). Prompts
    that fit whole in a piece go before a long one already under way, and a
    long prefill with nothing decoding still returns after each chunk when new
    requests have arrived, so they are admitted instead of waiting it out."""

    DEFAULT_RESERVE = 2048  # room reserved when the request sets no max_tokens
    LOG_INTERVAL = float(os.environ.get("QW_LOG_INTERVAL", "10"))  # seconds between stats lines (0: off)
    PREFILL_PIECE = int(os.environ.get("QW_PREFILL_PIECE", "2048"))
    DECODE_SHARE = float(os.environ.get("QW_DECODE_SHARE", "0.25"))
    # A waiting request whose prompt starts like one being prefilled waits until that one is past the shared part,
    # then takes it from the block store (QW_SHARE_WAIT=0: off; QW_SHARE_MIN: the least shared tokens still to prefill
    # that are worth waiting for).
    SHARE_WAIT = os.environ.get("QW_SHARE_WAIT", "1") != "0"
    SHARE_MIN = int(os.environ.get("QW_SHARE_MIN", "2048"))
    BATCH_PREFILL = os.environ.get("QW_PREFILL_BATCH", "1") != "0"  # several prompts per prefill pass
    # A prefill chunk or decode step takes seconds at most: an engine call running this long
    # (QW_STUCK_SECONDS) is a wedged GPU. Unhealthy for EXIT_GRACE seconds (QW_EXIT_GRACE), the
    # process exits so the supervisor restarts it (QW_WATCHDOG_EXIT=0: only /health says so).
    STUCK_SECONDS = float(os.environ.get("QW_STUCK_SECONDS", "300"))
    WATCHDOG_EXIT = os.environ.get("QW_WATCHDOG_EXIT", "1") != "0"
    EXIT_GRACE = float(os.environ.get("QW_EXIT_GRACE", "30"))
    WATCH_INTERVAL = 5.0

    def __init__(self, engine, mtp_drafts=3):
        self.raw_engine = engine  # for calls from other threads (failure()); the scheduler thread uses self.e
        self.call = None          # (engine method, time.time()) of the engine call in progress
        self.e = _Watched(engine, self)
        self.k = mtp_drafts if engine.has_mtp else 0
        self.stats = {"steps": 0, "rows": 0, "tokens": 0, "time": 0.0}  # decode steps, request-steps, emitted tokens
        self.cache_stats = engine.cache_stats()  # refreshed by the scheduler thread (the engine is not thread-safe)
        self.cv = threading.Condition()
        self.queue = []    # submitted, guarded by cv
        self.waiting = []  # admitted in FIFO order by the scheduler thread
        self.slot_epoch = 0   # counts slot releases: a waiting request that found no slot is retried after one
        self.loading = []     # in a slot, its cached prompt still loading from disk (background)
        self.prefilling = []  # in a slot, prompt partly prefilled
        self.active = []
        self.stopping = False
        # stats since the last log line: prompt tokens computed, generated tokens, prompt tokens
        # admitted and how many of those came from the prefix cache
        self.win = {"t": time.time(), "prefill": 0, "gen": 0, "prompt": 0, "cached": 0, "steps": 0, "step_time": 0.0,
                    "rows": 0}
        self.metrics = Metrics()  # for the dashboard
        self.thread = threading.Thread(target=self._loop, daemon=True, name="qw-scheduler")
        self.thread.start()
        if self.WATCHDOG_EXIT:
            threading.Thread(target=self._watchdog, daemon=True, name="qw-watchdog").start()

    def health(self):
        """(ok, reason): whether this engine can still serve. Not when the scheduler thread died, an
        engine call has run for over STUCK_SECONDS (a wedged GPU), or the engine reports that a rank
        failed or a collective timed out (permanent: every later job fails the same way)."""
        if not self.thread.is_alive() and not self.stopping:
            return False, "the scheduler thread died"
        call = self.call
        if call and time.time() - call[1] > self.STUCK_SECONDS:
            return False, f"engine call {call[0]} has been running for {time.time() - call[1]:.0f} s"
        failure = getattr(self.raw_engine, "failure", None)
        why = failure() if failure else ""
        if why:
            return False, f"engine failed: {why}"
        return True, "ok"

    def _watchdog(self):
        """Exits the process once health() has failed for EXIT_GRACE seconds: a failed or wedged
        engine does not recover, and while the process stays up its supervisor sees a running
        server that fails every request."""
        bad_since = None
        while not self.stopping:
            time.sleep(self.WATCH_INTERVAL)
            ok, why = self.health()
            if ok:
                bad_since = None
                continue
            bad_since = bad_since or time.time()
            if time.time() - bad_since >= self.EXIT_GRACE:
                print(f"watchdog: unhealthy for {time.time() - bad_since:.0f} s ({why}); exiting so the supervisor "
                      "restarts the engine", file=sys.stderr, flush=True)
                os._exit(3)

    def shutdown(self, timeout=30):
        """Stops after the current step; in-flight requests end with an error."""
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        self.thread.join(timeout)
        for r in self.waiting + self.loading + self.prefilling + self.active + self.queue:
            r.emit("error", "server shutting down")

    def submit(self, req):
        n = len(req.prompt)
        if n >= self.e.max_tokens:
            raise ValueError(f"prompt is {n} tokens; the context holds {self.e.max_tokens}")
        with self.cv:
            self.queue.append(req)
            self.cv.notify()

    def _log_request(self, r, reason):
        now = time.time()
        ttft = f"{r.t_first - r.t_admit:.2f} s" if r.t_first and r.t_admit else "-"
        decode = (f"{(r.generated - 1) / (now - r.t_first):.1f} tok/s"
                  if r.t_first and r.generated > 1 and now > r.t_first else "-")
        print(f"request {r.rid}: {reason}, prompt {len(r.prompt)} ({r.cached} cached), generated {r.generated}, "
              f"ttft {ttft}, decode {decode}, slot {r.slot}{self._phases(r)}", flush=True)

    @staticmethod
    def _phases(r):
        """Where the time before the first token went, appended to the request line: waiting for a slot
        (queue), the disk load of its cached prompt (load), restoring it (restore), the prefill calls of its
        own (prefill, which includes the saves to the prefix cache and, in them, the wait for pinned
        buffers) and the rest of the time from the start of the prefill to the first token, which others'
        decode steps and prefills took (interleaved). The ttft above counts from the slot being taken."""
        parts = [f"queue {r.t_admit - r.t_arrive:.2f} s"]
        if r.t_begin is not None:
            parts.append(f"load {r.t_begin - r.t_admit:.2f} s")
        t = r.timing
        if t and r.t_first and r.t_begin is not None:
            inter = max(0.0, r.t_first - r.t_begin - t["restore_s"] - t["prefill_s"])
            parts += [f"restore {t['restore_s']:.2f} s",
                      f"prefill {t['prefill_s']:.2f} s (saves {t['save_s']:.2f} s, pin wait {t['pin_wait_s']:.2f} s)",
                      f"interleaved {inter:.2f} s"]
        return " | " + ", ".join(parts)

    def _stats(self, force=False):
        """The periodic stats line (like vLLM's): throughput since the last line and the queue."""
        w, now = self.win, time.time()
        dt = now - w["t"]
        if self.LOG_INTERVAL <= 0 or (dt < self.LOG_INTERVAL and not force):
            return
        busy = self.active or self.prefilling or self.waiting or self.queue
        if w["prefill"] or w["gen"] or busy:
            cap = sum(self.e.capacity)
            used = sum(len(r.prompt) + r.generated for r in self.active + self.prefilling)
            reuse = f"{100 * w['cached'] / w['prompt']:.0f}%" if w["prompt"] else "-"
            step = f", {1e3 * w['step_time'] / w['steps']:.1f} ms/step, {w['gen'] / w['rows']:.2f} tok/row" if w["steps"] else ""
            print(f"stats: prompt {w['prefill'] / dt:.0f} tok/s, generation {w['gen'] / dt:.1f} tok/s{step} | "
                  f"running {len(self.active)}, prefilling {len(self.prefilling)}, loading {len(self.loading)}, waiting "
                  f"{len(self.waiting) + len(self.queue)} | slots {len(self.active) + len(self.prefilling) + len(self.loading)}/"
                  f"{len(self.e.capacity)}, KV {100 * used / cap:.1f}% | prompt tokens from cache {reuse}", flush=True)
        self.win = {"t": now, "prefill": 0, "gen": 0, "prompt": 0, "cached": 0, "steps": 0, "step_time": 0.0, "rows": 0}

    def _finish(self, r, reason):
        if r.t_admit is not None:
            self._log_request(r, reason)
        self.metrics.add_request(r.rid, reason, len(r.prompt), r.cached, r.generated, r.t_arrive, r.t_admit,
                                 r.t_first, time.time(), r.slot)
        if r.t_first is not None:
            r.emit("timings", r.timings())
        r.emit("end", reason)
        if r.slot >= 0:
            self.e.release(r.slot)
            r.slot = -1
            self.slot_epoch += 1

    def _took(self, r, tid, lp, top):
        """Handles a sampled token; returns False when the request is done."""
        r.generated += 1
        self.win["gen"] += 1
        self.metrics.add_generated()
        if tid in r.end_ids:
            r.emit("eos", tid)  # counted in completion_tokens, like vLLM
            self._finish(r, "stop")
            return False
        r.emit("token", (tid, lp, top))
        if r.generated >= r.limit:
            self._finish(r, "length")
            return False
        r.next = tid
        if r.think_left is not None and not r.forced:
            end_id, close = r.think_close
            if tid == end_id:
                r.think_left = None  # the model closed its thinking itself
            else:
                r.think_left -= 1
                if r.think_left <= 0:  # budget spent: close the thinking section for it
                    r.forced = list(close)
                    r.think_left = None
        return True

    def _admit(self):
        """Gives waiting requests slots in arrival order. One that no free slot fits keeps
        waiting while the ones after it may take the slots that are free (a request waiting for
        the big slot does not hold up short ones). It is not starved: every pass offers free
        slots in arrival order, so it gets the first one it fits. A request that found no slot is
        not offered again until some slot has been released (acquire hands the whole prompt to
        the engine, so asking every pass for every waiting request is not free). Returns whether
        any request left the waiting list."""
        i = 0
        moved = False
        while i < len(self.waiting):
            r = self.waiting[i]
            if r.cancelled.is_set():
                self.waiting.pop(i)
                r.emit("end", "abort")
                moved = True
                continue
            if r.no_slot_at == self.slot_epoch:  # no slot was released since it found none
                i += 1
                continue
            if self._shares_prefill(r):  # another request is prefilling the start of this prompt: wait for it
                i += 1
                continue
            n = len(r.prompt)
            need = r.max_new if r.max_new else self.DEFAULT_RESERVE
            need = max(1, min(need, self.e.max_tokens - n))
            try:
                slot = self.e.acquire(r.prompt, need, r.media)
            except Exception as ex:  # noqa: BLE001  a malformed request (media that do not match its prompt) fails alone
                self.waiting.pop(i)
                moved = True
                print(f"request {r.rid}: admission error: {ex}", file=sys.stderr, flush=True)
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            if slot < 0:
                r.no_slot_at = self.slot_epoch
                i += 1  # keeps waiting; later ones may fit the free slots
                continue
            self.waiting.pop(i)
            moved = True
            r.slot = slot
            try:
                room = self.e.capacity[slot] - n
                r.limit = min(r.max_new, room) if r.max_new else room
                self.e.set_stop_tokens(slot, sorted(r.end_ids))
                r.t_admit = time.time()
                if self.e.prefetch(slot, r.prompt, r.media):  # cached entries on disk: load them in the background
                    self.loading.append(r)
                    continue
            except Exception as ex:  # noqa: BLE001
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            self._begin(r)
        return moved

    def _shares_prefill(self, r):
        """True while a request being prefilled still has SHARE_MIN or more tokens to go of a prefix it shares with r
        (the same system prompt, or the same prompt sent twice): r then waits, and restores that prefix from the block
        store (the snapshot at a message boundary or chunk end in it) instead of computing it a second time next to
        the first. QW_SHARE_WAIT=0 turns it off."""
        if not self.SHARE_WAIT or r.media:
            return False
        for p in self.prefilling:
            if p.media or p.cancelled.is_set():
                continue
            common = r.share.get(id(p))
            if common is None:
                a, b = r.prompt_np(), p.prompt_np()
                m = min(len(a), len(b))
                diff = np.nonzero(a[:m] != b[:m])[0]
                common = r.share[id(p)] = int(diff[0]) if len(diff) else m
            # worth waiting for when SHARE_MIN or more shared tokens are still to come; once it waits, it waits until
            # the other request is past the whole shared part (only then is the snapshot at its end in the store:
            # released earlier, it found nothing to restore and prefilled everything itself, 1.7 s later)
            left = common - (len(p.prompt) - p.left)
            if left >= self.SHARE_MIN or (left > 0 and r.share_on == id(p)):
                r.share_on = id(p)
                if r.t_share is None:
                    r.t_share = time.time()
                return True
        r.share_on = None
        return False

    def _poll_loading(self):
        """Requests whose cached prompt was loading from disk: begin those whose load is done."""
        for r in list(self.loading):
            if r.cancelled.is_set():
                self.loading.remove(r)
                self._finish(r, "abort")
                continue
            try:
                if self.e.prefetch(r.slot, r.prompt, r.media):
                    continue
            except Exception as ex:  # noqa: BLE001
                self.loading.remove(r)
                r.emit("error", str(ex))
                self._finish(r, "abort")
                continue
            self.loading.remove(r)
            self._begin(r)

    def _begin(self, r):
        """Restores what the caches hold of r's prompt into its slot and queues the rest for prefill."""
        r.t_begin = time.time()
        try:
            r.cached, r.keep = self.e.begin_prompt(r.slot, r.prompt, r.media)
            r.left = len(r.prompt) - r.cached
            self.win["prompt"] += len(r.prompt)
            self.win["cached"] += r.cached
            self.metrics.add_admitted(len(r.prompt), r.cached)
            self.cache_stats = self.e.cache_stats()
            r.emit("start", r.cached)
        except Exception as ex:  # noqa: BLE001
            r.emit("error", str(ex))
            self._finish(r, "abort")
            return
        self.prefilling.append(r)

    def _batch(self):
        """The prefills to advance in one engine pass: prompts that fit whole go first (in
        arrival order, so a short request does not wait behind a long prompt), then the oldest
        of the rest; several while their whole prompts fit (QW_PREFILL_PIECE tokens while
        others decode, else one prefill chunk); only the first may go in partly. [(request, tokens)]."""
        room = min(self.PREFILL_PIECE, self.e.prefill_chunk) if self.active else self.e.prefill_chunk
        short = [r for r in self.prefilling if r.left <= room and not r.media]
        batch = []
        for r in short + [r for r in self.prefilling if r not in short]:
            if r.media or room <= 0 or (batch and (r.left > room or not self.BATCH_PREFILL)):
                break  # media prompts go alone (their vision tokens are encoded per slot)
            take = min(r.left, room)
            batch.append((r, take))
            room -= take
        if not batch:  # a media prompt first in line
            batch = [(self.prefilling[0], self.PREFILL_PIECE if self.active else self.e.prefill_chunk)]
        return batch

    def _prefill(self):
        """Advances the prefills (see _batch): by one piece while others decode, then returns so
        they get steps; chunk by chunk to the end when nothing is decoding, but returning as
        soon as new requests arrive so they are admitted. Returns the seconds spent."""
        t0 = time.time()
        while self.prefilling:
            for r in [r for r in self.prefilling if r.cancelled.is_set()]:
                self.prefilling.remove(r)
                self._finish(r, "abort")
            if not self.prefilling:
                break
            batch = self._batch()
            computed = sum(min(n, r.left) for r, n in batch)
            self.win["prefill"] += computed
            try:
                if len(batch) == 1:
                    done = [self.e.prefill_some(batch[0][0].slot, batch[0][1])]
                else:
                    done = self.e.prefill_batch([(r.slot, n) for r, n in batch])
                self.metrics.add_prefill(computed)
            except Exception as ex:  # noqa: BLE001
                print(f"prefill error ({', '.join(r.rid for r, _ in batch)}): {ex}\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
                for r, _ in batch:
                    self.prefilling.remove(r)
                    r.emit("error", str(ex))
                    self._finish(r, "abort")
                continue
            self.cache_stats = self.e.cache_stats()
            for (r, n), d in zip(batch, done):
                if not d:
                    r.left -= n
                    continue
                self.prefilling.remove(r)
                try:
                    top = self.e.top_logprobs_prompt(r.slot, r.want_top) if r.want_top else None
                    tid, lp = self.e.sample_prompt(r.slot, r.sampling)
                    r.t_first = time.time()
                    r.timing = self.e.slot_timing(r.slot)
                    r.keep = None
                except Exception as ex:  # noqa: BLE001
                    r.emit("error", str(ex))
                    self._finish(r, "abort")
                    continue
                if self._took(r, tid, lp, top):
                    self.active.append(r)
            if self.active or self.queue:
                return time.time() - t0  # running requests take steps, new ones are admitted
        return time.time() - t0

    def _step(self):
        for r in [r for r in self.active if r.cancelled.is_set()]:
            self.active.remove(r)
            self._finish(r, "abort")
        if not self.active:
            return
        batch = self.active[:16]
        self.active = self.active[16:] + batch  # round-robin past 16 requests (at most num_slots anyway)
        def budget(r):  # tokens the engine may emit for r this step
            if r.forced:
                return 1  # no drafts: its sampled token is replaced by the forced one
            b = r.limit - r.generated
            return max(1, min(b, r.think_left)) if r.think_left is not None else b
        t0 = time.time()
        res = self.e.generate([(r.slot, r.next, budget(r), r.sampling, r.want_top > 0) for r in batch], self.k)
        st = self.stats
        st["time"] += time.time() - t0
        self.win["steps"] += 1
        self.win["step_time"] += time.time() - t0
        self.win["rows"] += len(batch)
        st["steps"] += 1
        st["rows"] += len(batch)
        st["tokens"] += sum(len(t) for t, _, _, _ in res)
        for r, (toks, lps, first, _) in zip(batch, res):
            if r.forced:  # emit the next forced token; the engine's sample for this row is dropped
                if not self._took(r, r.forced.pop(0), 0.0, None):
                    self.active.remove(r)
                continue
            for j, (tid, lp) in enumerate(zip(toks, lps)):
                top = self.e.top_logprobs(first + j, r.want_top) if r.want_top else None
                if not self._took(r, tid, lp, top):  # the engine stopped at the same token
                    self.active.remove(r)
                    break

    def _loop(self):
        while True:
            with self.cv:
                while (not self.queue and not self.waiting and not self.prefilling and not self.active
                       and not self.loading and not self.stopping):
                    self._stats(force=True)  # the last window before going idle
                    self.metrics.idle()
                    self.cv.wait()
                    now = time.time()  # windows restart on wake: idle time is not averaged in
                    self.win["t"] = now
                    self.metrics.wake()
                if self.stopping:
                    return
                self.waiting += self.queue  # the queue is shared; waiting is this thread's
                self.queue = []
            t_pass = time.time()
            try:
                moved = self._admit()
                self._poll_loading()
                if not (self.active or self.prefilling or self.queue or moved):
                    time.sleep(0.02)  # only disk loads in flight, or requests waiting for a slot: do not spin
                spent = self._prefill()
                t0 = time.time()
                self._step()
                while self.prefilling and self.active and time.time() - t0 < self.DECODE_SHARE * spent:
                    self._step()
                self._stats()
                self.metrics.tick(time.time() - t_pass, len(self.prefilling), len(self.active))
                self.metrics.sample(len(self.active), len(self.prefilling),
                                    len(self.waiting) + len(self.queue) + len(self.loading),
                                    sum(len(r.prompt) + r.generated for r in self.active + self.prefilling),
                                    sum(self.e.capacity))
            except Exception as ex:  # noqa: BLE001  engine failure: fail everything in flight
                print(f"scheduler error ({len(self.prefilling) + len(self.active)} requests failed): {ex}\n"
                      f"{traceback.format_exc()}", file=sys.stderr, flush=True)
                for r in self.loading + self.prefilling + self.active:
                    r.emit("error", str(ex))
                    try:
                        self._finish(r, "abort")
                    except Exception:  # noqa: BLE001
                        pass
                self.loading = []
                self.prefilling = []
                self.active = []
