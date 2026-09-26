"""ctypes binding of the engine's C API (include/qw/capi.h)."""

import ctypes
import json


class Sampling(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("min_p", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


class StepReq(ctypes.Structure):
    _fields_ = [
        ("slot", ctypes.c_int32),
        ("pending", ctypes.c_int32),
        ("budget", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("sampling", Sampling),
    ]


class MediaStruct(ctypes.Structure):
    _fields_ = [
        ("hash", ctypes.c_uint64),
        ("video", ctypes.c_int32),
        ("t", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("w", ctypes.c_int32),
        ("patches", ctypes.POINTER(ctypes.c_float)),
        ("starts", ctypes.POINTER(ctypes.c_int64)),
    ]


class CacheStats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "hits", "tokens_restored", "snapshots_saved", "ram_bytes", "disk_bytes", "blocks", "snapshots",
        "prompt_tokens", "reused_tokens", "blend_candidate_tokens")]


class Engine:
    """ctypes wrapper of the slot C API. Not thread-safe: only the scheduler
    thread calls it."""

    def __init__(self, lib_path, options):
        lib = ctypes.CDLL(lib_path)
        P, I32P, FP = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
        sig = {
            "qw_open": (P, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
            "qw_error": (ctypes.c_char_p, [P]),
            "qw_num_slots": (ctypes.c_int, [P]),
            "qw_slot_capacity": (ctypes.c_int64, [P, ctypes.c_int]),
            "qw_acquire": (ctypes.c_int, [P, I32P, ctypes.c_int64, ctypes.c_int64]),
            "qw_release": (ctypes.c_int, [P, ctypes.c_int]),
            "qw_set_prompt": (ctypes.c_int64, [P, ctypes.c_int, I32P, ctypes.c_int64]),
            "qw_set_prompt_media": (ctypes.c_int64, [P, ctypes.c_int, I32P, ctypes.c_int64,
                                                      ctypes.POINTER(MediaStruct), ctypes.c_int]),
            "qw_has_vision": (ctypes.c_int, [P]),
            "qw_begin_prompt": (ctypes.c_int64, [P, ctypes.c_int, I32P, ctypes.c_int64,
                                                 ctypes.POINTER(MediaStruct), ctypes.c_int]),
            "qw_prefill_some": (ctypes.c_int, [P, ctypes.c_int, ctypes.c_int64]),
            "qw_acquire_media": (ctypes.c_int, [P, I32P, ctypes.c_int64, ctypes.c_int64,
                                                 ctypes.POINTER(MediaStruct), ctypes.c_int]),
            "qw_sample_prompt": (ctypes.c_int32, [P, ctypes.c_int, ctypes.POINTER(Sampling), FP]),
            "qw_decode": (ctypes.c_int, [P, ctypes.c_int, I32P, I32P]),
            "qw_sample_row": (ctypes.c_int32, [P, ctypes.c_int, ctypes.POINTER(Sampling), FP]),
            "qw_top_logprobs": (ctypes.c_int, [P, ctypes.c_int, ctypes.c_int, I32P, FP]),
            "qw_has_mtp": (ctypes.c_int, [P]),
            "qw_persist": (ctypes.c_int, [P]),
            "qw_set_boundary_token": (ctypes.c_int, [P, ctypes.c_int32]),
            "qw_get_cache_stats": (ctypes.c_int, [P, ctypes.POINTER(CacheStats)]),
            "qw_set_stop_tokens": (ctypes.c_int, [P, ctypes.c_int, I32P, ctypes.c_int]),
            "qw_generate": (ctypes.c_int, [P, ctypes.c_int, ctypes.POINTER(StepReq), ctypes.c_int, I32P, FP, I32P,
                                           I32P, I32P]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        err = ctypes.create_string_buffer(4096)
        self.h = lib.qw_open(json.dumps(options).encode(), err, len(err))
        if not self.h:
            raise RuntimeError("engine failed to start: " + err.value.decode(errors="replace"))
        self.lib = lib
        self.capacity = [lib.qw_slot_capacity(self.h, i) for i in range(lib.qw_num_slots(self.h))]
        self.max_tokens = max(self.capacity)
        self.has_mtp = bool(lib.qw_has_mtp(self.h))
        self.has_vision = bool(lib.qw_has_vision(self.h))

    def _err(self):
        return RuntimeError(self.lib.qw_error(self.h).decode(errors="replace"))

    def _check(self, r):
        if r < 0:
            raise self._err()
        return r

    @staticmethod
    def _media_array(media):
        """ctypes array of the media, and the numpy buffers it points into (keep them alive)."""
        import numpy as np
        keep, items = [], []
        for m in media:
            patches = np.ascontiguousarray(m.patches, dtype=np.float32)
            starts = np.asarray([s for s, _ in m.spans], dtype=np.int64)
            keep += [patches, starts]
            t, h, w = m.grid
            items.append(MediaStruct(int.from_bytes(m.hash, "little"), int(m.kind == "video"), t, h, w,
                                     patches.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))))
        return (MediaStruct * len(items))(*items), keep

    def acquire(self, tokens, max_new, media=None):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        if media:
            arr_m, keep = self._media_array(media)
            r = self.lib.qw_acquire_media(self.h, arr, len(tokens), max_new, arr_m, len(media))
        else:
            r = self.lib.qw_acquire(self.h, arr, len(tokens), max_new)
        if r < 0 and self.lib.qw_error(self.h):
            raise self._err()
        return r  # -1: no free slot fits right now

    def release(self, slot):
        self._check(self.lib.qw_release(self.h, slot))

    def set_prompt(self, slot, tokens, media=None):
        """media: vision.Media items whose spans are set (their pads are in `tokens`)."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        if not media:
            return self._check(self.lib.qw_set_prompt(self.h, slot, arr, len(tokens)))
        arr_m, keep = self._media_array(media)
        return self._check(self.lib.qw_set_prompt_media(self.h, slot, arr, len(tokens), arr_m, len(media)))

    def begin_prompt(self, slot, tokens, media=None):
        """Restores what the caches hold of the prompt; returns (reused tokens, buffers to keep
        alive until prefill_some has put the rest in)."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        arr_m, keep = self._media_array(media or [])
        reused = self._check(self.lib.qw_begin_prompt(self.h, slot, arr, len(tokens), arr_m, len(media or [])))
        return reused, keep  # keep: buffers the C side reads until the prompt is in

    def prefill_some(self, slot, max_tokens):
        """Prefills up to max_tokens more of the slot's prompt; True once it is all in."""
        return bool(self._check(self.lib.qw_prefill_some(self.h, slot, max_tokens)))

    def sample_prompt(self, slot, s):
        lp = ctypes.c_float()
        return self._check(self.lib.qw_sample_prompt(self.h, slot, ctypes.byref(s), ctypes.byref(lp))), lp.value

    def decode(self, slots, tokens):
        n = len(slots)
        self._check(self.lib.qw_decode(self.h, n, (ctypes.c_int32 * n)(*slots), (ctypes.c_int32 * n)(*tokens)))

    def sample_row(self, row, s):
        lp = ctypes.c_float()
        return self._check(self.lib.qw_sample_row(self.h, row, ctypes.byref(s), ctypes.byref(lp))), lp.value

    def top_logprobs(self, row, k):
        """row < 0: the distribution after the last set_prompt."""
        ids = (ctypes.c_int32 * k)()
        lps = (ctypes.c_float * k)()
        n = self._check(self.lib.qw_top_logprobs(self.h, row, k, ids, lps))
        return list(zip(ids[:n], lps[:n]))

    def set_stop_tokens(self, slot, ids):
        ids = list(ids)
        self._check(self.lib.qw_set_stop_tokens(self.h, slot, (ctypes.c_int32 * max(1, len(ids)))(*ids), len(ids)))

    def generate(self, reqs, k):
        """reqs: (slot, pending, budget, Sampling). Returns per request
        (tokens, logprobs, first_row, stopped); token j came from row first_row + j."""
        n, w = len(reqs), k + 1
        arr = (StepReq * n)(*[StepReq(s, p, b, 0, smp) for s, p, b, smp in reqs])
        toks, lps = (ctypes.c_int32 * (n * w))(), (ctypes.c_float * (n * w))()
        counts, firsts, stopped = (ctypes.c_int32 * n)(), (ctypes.c_int32 * n)(), (ctypes.c_int32 * n)()
        self._check(self.lib.qw_generate(self.h, n, arr, k, toks, lps, counts, firsts, stopped))
        return [(list(toks[i * w:i * w + counts[i]]), list(lps[i * w:i * w + counts[i]]), firsts[i], bool(stopped[i]))
                for i in range(n)]

    def set_boundary_token(self, token_id):
        """Token that starts a chat message: prefill snapshots the state before it."""
        self._check(self.lib.qw_set_boundary_token(self.h, token_id))

    def cache_stats(self):
        s = CacheStats()
        self._check(self.lib.qw_get_cache_stats(self.h, ctypes.byref(s)))
        return {name: getattr(s, name) for name, _ in CacheStats._fields_}

    def persist(self):
        """Saves the slots' conversations to the disk prefix cache."""
        self._check(self.lib.qw_persist(self.h))
