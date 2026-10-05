#!/usr/bin/env python3
"""Sampling defaults (no GPU, no model): a request gets the model's generation_config.json values
for the sampling settings it does not send; the ones it sends win, and 0 still means off.

  python3 server/tests/test_sampling_defaults.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve.scheduler import Request, sampling_defaults  # noqa: E402


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {what} {detail}")

    def sampling(body, defaults):
        s = Request([1, 2, 3], body, lambda *a: None, set(), defaults=defaults).sampling
        return round(s.temperature, 3), round(s.top_p, 3), s.top_k, round(s.repetition_penalty, 3)

    with tempfile.TemporaryDirectory() as d:
        check("no generation_config.json: no defaults", sampling_defaults(d) == {})
        with open(os.path.join(d, "generation_config.json"), "w") as f:
            json.dump({"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "eos_token_id": [5, 6],
                       "bos_token_id": 5}, f)
        dflt = sampling_defaults(d)
        check("only the sampling values are read", dflt == {"temperature": 1.0, "top_k": 20, "top_p": 0.95}, f"({dflt})")
        with open(os.path.join(d, "generation_config.json"), "w") as f:
            f.write("{not json")
        check("an unreadable file: no defaults", sampling_defaults(d) == {})

    check("a request that sets nothing gets the model's", sampling({}, dflt) == (1.0, 0.95, 20, 1.0))
    check("null counts as not set", sampling({"temperature": None, "top_p": None}, dflt) == (1.0, 0.95, 20, 1.0))
    check("the request's values win", sampling({"temperature": 0.6, "top_p": 0.8, "top_k": 40}, dflt) == (0.6, 0.8, 40, 1.0))
    check("temperature 0 stays greedy", sampling({"temperature": 0}, dflt)[0] == 0.0)
    check("top_k 0 and top_p 0 turn them off", sampling({"top_k": 0, "top_p": 0}, dflt)[1:3] == (1.0, 0))
    check("without defaults: as before", sampling({}, None) == (1.0, 1.0, 0, 1.0)
          and sampling({"repetition_penalty": 0}, {}) == (1.0, 1.0, 0, 1.0))
    print("all ok" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
