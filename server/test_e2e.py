#!/usr/bin/env python3
"""End-to-end OpenAI API checks (stdlib only). Runs fixed greedy scenarios
against a server and saves or compares the results.

  test_e2e.py --url http://host:port --model NAME --save ref.json
  test_e2e.py --url http://host:port --model NAME --compare ref.json
"""

import argparse
import json
import sys
import time
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
NOTHINK = {"enable_thinking": False}


def post(url, body, stream=False):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not stream:
            return json.load(r), time.time() - t0, None
        content, reasoning, calls, usage, first = "", "", [], None, None
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning") or d.get("reasoning_content")) and first is None:
                    first = time.time() - t0
                content += d.get("content") or ""
                reasoning += d.get("reasoning") or d.get("reasoning_content") or ""
                calls += d.get("tool_calls") or []
        msg = {"content": content, "reasoning": reasoning, "tool_calls": calls}
        return {"choices": [{"message": msg}], "usage": usage}, time.time() - t0, first


def scenarios(model):
    base = {"model": model, "temperature": 0}
    return {
        "short": dict(base, messages=[{"role": "user", "content": "What is the capital of France? One sentence."}],
                      max_tokens=64, chat_template_kwargs=NOTHINK),
        "think": dict(base, messages=[{"role": "user", "content": "What is 17*23? Answer with just the number."}],
                      max_tokens=512, chat_template_kwargs={"reasoning_effort": "low"}),
        "tool": dict(base, messages=[{"role": "user", "content": "What's the weather in Paris for the next 3 days?"}],
                     tools=TOOLS, max_tokens=256, chat_template_kwargs=NOTHINK),
        "stream": dict(base, messages=[{"role": "user", "content": "Name three primary colors, comma separated."}],
                       max_tokens=64, stream=True, stream_options={"include_usage": True},
                       chat_template_kwargs=NOTHINK),
    }


def summarize(resp):
    m = resp["choices"][0]["message"]
    calls = [[c["function"]["name"], json.loads(c["function"]["arguments"])] for c in (m.get("tool_calls") or [])]
    return {"content": (m.get("content") or "").strip(), "reasoning": (m.get("reasoning") or m.get("reasoning_content") or "").strip(),
            "tool_calls": calls, "usage": resp.get("usage")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()
    out = {}
    for name, body in scenarios(a.model).items():
        r, dt, first = post(a.url + "/v1/chat/completions", body, stream=body.get("stream", False))
        out[name] = summarize(r)
        print(f"{name:7s} {dt:6.2f}s{'' if first is None else f' (first token {first:.2f}s)'}: "
              f"{out[name]['content'][:80]!r} calls={out[name]['tool_calls']} usage={out[name]['usage']}")
    # two turns: the second extends the first, so its prompt should be mostly cached
    msgs = [{"role": "user", "content": "Write a haiku about GPUs."}]
    body = {"model": a.model, "temperature": 0, "max_tokens": 64, "messages": msgs, "chat_template_kwargs": NOTHINK}
    r1, dt1, _ = post(a.url + "/v1/chat/completions", body)
    msgs = msgs + [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
                   {"role": "user", "content": "Now one about CPUs."}]
    r2, dt2, _ = post(a.url + "/v1/chat/completions", dict(body, messages=msgs))
    out["turn1"], out["turn2"] = summarize(r1), summarize(r2)
    print(f"turn1   {dt1:6.2f}s usage={r1.get('usage')}")
    print(f"turn2   {dt2:6.2f}s usage={r2.get('usage')}")
    if a.save:
        json.dump(out, open(a.save, "w"), indent=1, ensure_ascii=False)
    if a.compare:
        ref = json.load(open(a.compare))
        for k in ref:
            same = (ref[k]["content"] == out[k]["content"] and ref[k]["tool_calls"] == out[k]["tool_calls"])
            print(f"{k:7s} {'SAME' if same else 'DIFFERENT'}")
            if not same:
                print("   ref :", repr(ref[k]["content"][:200]), ref[k]["tool_calls"])
                print("   ours:", repr(out[k]["content"][:200]), out[k]["tool_calls"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
