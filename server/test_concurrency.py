#!/usr/bin/env python3
"""Concurrent-request check (stdlib only): greedy requests sent in parallel
must produce the same text as the same requests sent one at a time, and the
aggregate generation rate is reported.

  test_concurrency.py --url http://host:port --model NAME [-n 4] [--tokens 256]
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

TOPICS = ["the history of the printing press", "how rainbows form", "the rules of chess", "why the sky is blue",
          "how vaccines work", "the water cycle", "black holes", "how compilers work"]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/v1/chat/completions"
    bodies = [{"model": a.model, "temperature": 0, "max_tokens": a.tokens, "ignore_eos": True,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": f"Write a detailed essay about {TOPICS[i % len(TOPICS)]}."}]}
              for i in range(a.n)]

    seq, t_seq = [], 0.0
    for b in bodies:
        r, dt = post(url, b)
        seq.append(r)
        t_seq += dt
    n_tok = sum(r["usage"]["completion_tokens"] for r in seq)
    print(f"sequential: {n_tok} tokens in {t_seq:.1f} s = {n_tok / t_seq:.1f} tok/s")

    par = [None] * a.n

    def run(i):
        par[i] = post(url, bodies[i])[0]

    t0 = time.time()
    th = [threading.Thread(target=run, args=(i,)) for i in range(a.n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    t_par = time.time() - t0
    n_tok = sum(r["usage"]["completion_tokens"] for r in par)
    print(f"parallel x{a.n}: {n_tok} tokens in {t_par:.1f} s = {n_tok / t_par:.1f} tok/s")

    bad = 0
    for i, (s, p) in enumerate(zip(seq, par)):
        cs, cp = s["choices"][0]["message"]["content"], p["choices"][0]["message"]["content"]
        if cs != cp:
            bad += 1
            k = next((j for j in range(min(len(cs), len(cp))) if cs[j] != cp[j]), min(len(cs), len(cp)))
            print(f"request {i}: DIFFERENT at char {k}: seq {cs[k:k + 40]!r} vs par {cp[k:k + 40]!r}")
        print(f"request {i}: cached {p['usage']['prompt_tokens_details']['cached_tokens']}"
              f"/{p['usage']['prompt_tokens']} prompt tokens")
    print("all ok" if not bad else f"{bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
