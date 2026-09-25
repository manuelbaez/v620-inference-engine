#!/usr/bin/env python3
"""Fetch ground truth from a running vLLM (stdlib only).

  vllm_logprobs.py tokenize "text"             -> comma-separated token ids
  vllm_logprobs.py logprobs 1,2,3 OUT [--gen N] -> writes "<pos> <tok> <logprob>" lines
                                                 and prints vLLM's greedy continuation
  vllm_logprobs.py compare REF VLLM             -> diff two logprob files
"""
import json
import sys
import urllib.request

BASE = "http://llm-backend-amd.local.net:8080"
MODEL = "vllm/qwen3.8-flash-next-rdna2"


def post(path, body):
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    cmd = sys.argv[1]
    if cmd == "tokenize":
        r = post(f"/upstream/{MODEL}/tokenize", {"model": MODEL, "prompt": sys.argv[2]})
        print(",".join(map(str, r["tokens"])))
    elif cmd == "logprobs":
        ids = [int(x) for x in sys.argv[2].split(",")]
        gen = int(sys.argv[sys.argv.index("--gen") + 1]) if "--gen" in sys.argv else 8
        r = post(
            f"/upstream/{MODEL}/v1/completions",
            {
                "model": MODEL,
                "prompt": ids,
                "max_tokens": gen,
                "temperature": 0,
                "prompt_logprobs": 0,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            },
        )
        ch = r["choices"][0]
        with open(sys.argv[3], "w") as f:
            for pos, entry in enumerate(ch["prompt_logprobs"]):
                if entry is None:
                    continue
                tok = ids[pos]
                lp = entry[str(tok)]["logprob"]
                f.write(f"{pos} {tok} {lp:.5f}\n")
        toks = [t.split(":")[1] for t in ch["logprobs"]["tokens"]]
        print("generated: " + " ".join(toks))
    elif cmd == "compare":
        ref = {}
        for line in open(sys.argv[2]):
            p = line.split()
            ref[int(p[0])] = (int(p[1]), float(p[2]), int(p[3]))
        diffs, agree, n = [], 0, 0
        for line in open(sys.argv[3]):
            p = line.split()
            pos, tok, lp = int(p[0]), int(p[1]), float(p[2])
            if pos not in ref:
                continue
            n += 1
            diffs.append(abs(ref[pos][1] - lp))
            agree += ref[pos][2] == tok
        diffs.sort()
        print(f"positions {n}  mean |dlogprob| {sum(diffs)/n:.4f}  "
              f"median {diffs[n//2]:.4f}  max {diffs[-1]:.4f}  "
              f"ref-top1==actual-next {agree}/{n}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
