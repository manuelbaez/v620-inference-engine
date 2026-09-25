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
            "qw_sample_prompt": (ctypes.c_int32, [P, ctypes.c_int, ctypes.POINTER(Sampling), FP]),
            "qw_decode": (ctypes.c_int, [P, ctypes.c_int, I32P, I32P]),
            "qw_sample_row": (ctypes.c_int32, [P, ctypes.c_int, ctypes.POINTER(Sampling), FP]),
            "qw_top_logprobs": (ctypes.c_int, [P, ctypes.c_int, ctypes.c_int, I32P, FP]),
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

    def _err(self):
        return RuntimeError(self.lib.qw_error(self.h).decode(errors="replace"))

    def _check(self, r):
        if r < 0:
            raise self._err()
        return r

    def acquire(self, tokens, max_new):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        r = self.lib.qw_acquire(self.h, arr, len(tokens), max_new)
        if r < 0 and self.lib.qw_error(self.h):
            raise self._err()
        return r  # -1: no free slot fits right now

    def release(self, slot):
        self._check(self.lib.qw_release(self.h, slot))

    def set_prompt(self, slot, tokens):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        return self._check(self.lib.qw_set_prompt(self.h, slot, arr, len(tokens)))

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
