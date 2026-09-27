#!/usr/bin/env python3
"""Checks the server's text layer without the engine:
  1. chat template + tokenizer produce the same token ids as production vLLM's
     /tokenize for a set of conversations (plain, tools, tool calls, reasoning);
  2. the output parser splits reasoning / content / tool calls like vLLM's
     qwen3 parsers, on token streams built from text.
Usage: test_text.py [--vllm URL]"""

import argparse
import json
import sys
import urllib.request

import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwserve.output_parser import OutputParser  # noqa: E402
from qwserve.prompt import ChatPrompt  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["city"]}}}]

CONVERSATIONS = [
    {"messages": [{"role": "user", "content": "Hello there"}]},
    {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"}]},
    {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}},
    {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS},
    {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather",
                                                                        "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"},
        {"role": "user", "content": "Thanks, and tomorrow?"}], "tools": TOOLS},
    {"messages": [{"role": "user", "content": "Explain ünïcödé 日本語 and emoji 🎉"},
                  {"role": "assistant", "content": "Sure.", "reasoning_content": "easy"},
                  {"role": "user", "content": "More?"}]},
]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://llm-backend-amd.local.net:8080/upstream/vllm/qwen3.8-flash-next-rdna2")
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    a = ap.parse_args()

    srv = ChatPrompt(a.model_dir)

    fails = 0
    for i, conv in enumerate(CONVERSATIONS):
        text, _, _, _ = srv.render(dict(conv))
        ours = srv.encode(text)
        body = dict(conv)
        body["model"] = "vllm/qwen3.8-flash-next-rdna2"
        body["add_generation_prompt"] = True
        try:
            theirs = post(a.vllm + "/tokenize", body)["tokens"]
        except Exception as e:  # noqa: BLE001
            print(f"conv {i}: vLLM /tokenize unavailable ({e}); ours has {len(ours)} tokens")
            continue
        ok = ours == theirs
        fails += not ok
        first = next((k for k, (x, y) in enumerate(zip(ours, theirs)) if x != y), min(len(ours), len(theirs)))
        print(f"conv {i}: {'ok' if ok else 'MISMATCH'} ({len(ours)} vs {len(theirs)} tokens" +
              ("" if ok else f", first diff at {first}: ours {srv.tok.decode(ours[first:first+8], skip_special_tokens=False)!r}"
               f" theirs {srv.tok.decode(theirs[first:first+8], skip_special_tokens=False)!r}") + ")")

    # parser on synthetic output
    def run(text, thinking=True, tools=TOOLS):
        p = OutputParser(srv.tok, thinking, tools, parse_tools=True)
        for t in srv.encode(text):
            p.feed(t)
        p.finish()
        return p

    p = run("Let me think.\n</think>\n\nThe answer is 4.")
    fails += not (p.reasoning == "Let me think.\n" and p.content == "\n\nThe answer is 4.")
    print("parser reasoning/content:", repr(p.reasoning), repr(p.content))
    p = run("I'll call it.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n"
            "</parameter>\n</function>\n</tool_call>")
    args = json.loads(p.tool_calls[0]["function"]["arguments"]) if p.tool_calls else None
    good = args == {"city": "Paris", "days": 3, "metric": True, "tags": ["a", "b"]}
    fails += not good
    print("parser tool call:", p.tool_calls[0]["function"]["name"] if p.tool_calls else None, args, "ok" if good else "FAIL")
    p = run("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>")
    fails += not (p.tool_calls and p.reasoning == "")
    print("parser tool call straight from reasoning:", [c["function"]["arguments"] for c in p.tool_calls])
    p = run("Plain answer.", thinking=False)
    fails += not (p.content == "Plain answer." and p.reasoning == "")
    print("parser no-thinking:", repr(p.content))
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
