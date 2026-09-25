#!/usr/bin/env python3
"""Cases for qw_blend_eval: agent-style prompts where a tool output (a source
file) is reused after a different prefix. Writes the case file to stdout.

  python3 tools/blend_cases.py --model-dir /mnt/llms/qwen3.8-flash-next-awq > cases.txt
"""
import argparse
import os

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    "block_store": ("src/session/block_store.cpp",
                    "In block_store.cpp, how does enforce_budgets choose what to evict from RAM? Answer in two sentences."),
    "disk_tier": ("src/session/disk_tier.cpp",
                  "In disk_tier.cpp, what magic bytes and file extensions are used for blocks and snapshots?"),
    "prefill": ("src/engine/prefill.hip", "In prefill.hip, why are the router logits kept in fp32?"),
    "qsa": ("src/kernels/qsa.hip", "In qsa.hip, what does qsa_compress_T_kernel compute for each group of tokens?"),
    "scheduler": ("server/qwserve/scheduler.py", "In scheduler.py, what is DEFAULT_RESERVE used for?"),
}
SYS = ("You are a careful coding agent working in the qw inference engine repository. You read files with tools and "
       "answer questions about them precisely.\n\n" + open(os.path.join(ROOT, "README.md")).read()[:6000])


def msg(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def tool(name):
    body = open(os.path.join(ROOT, FILES[name][0])).read()
    return msg("user", f"<tool_response>\n{FILES[name][0]}:\n{body}\n</tool_response>")


def query(name):
    return msg("user", FILES[name][1]) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    args = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    sys = msg("system", SYS)
    ask = msg("user", "Read the files I need for this refactor.")
    cases = [
        # the chunk moved: it followed another file, now follows a third one
        ("moved/block_store", sys + ask + tool("disk_tier"), sys + ask + tool("scheduler"), "block_store", "block_store"),
        ("moved/qsa", sys + ask + tool("prefill"), sys + ask + tool("block_store"), "qsa", "qsa"),
        # the chunk came first in the source, now after two files
        ("first->third/disk_tier", sys + ask, sys + ask + tool("qsa") + tool("scheduler"), "disk_tier", "disk_tier"),
        # an earlier message was edited (the agent's request reworded)
        ("edited/prefill", sys + ask,
         sys + msg("user", "Read the files I need for this refactor, then list anything risky."), "prefill", "prefill"),
        # the question is about the file before the chunk
        ("moved/ask-earlier", sys + ask + tool("disk_tier"), sys + ask + tool("scheduler"), "block_store", "scheduler"),
    ]
    for name, pre_src, pre_dst, chunk, q in cases:
        print("#" + name)
        for text in (pre_src, pre_dst, tool(chunk), query(q)):
            print(",".join(map(str, enc(text))))
        print()


if __name__ == "__main__":
    main()
