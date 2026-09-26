#!/usr/bin/env python3
"""Calibration sequences for GPTQ (tools/qw_calibrate): source code, docs and
chat-formatted text, tokenized, one sequence of comma-separated ids per line.
docs/DESIGN.md is left out: it is the evaluation text.

  python3 tools/gptq_calib_data.py --out /tmp/calib.txt --roots ~/inference-engine ~/vllm-qwen4exp
"""
import argparse
import os
import random

from tokenizers import Tokenizer

EXT = (".py", ".cpp", ".hpp", ".hip", ".cuh", ".h", ".c", ".md", ".rst", ".txt", ".sh", ".yaml", ".json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/calib.txt")
    ap.add_argument("--seqs", type=int, default=128)
    ap.add_argument("--len", type=int, default=2048)
    args = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    files = []
    for root in args.roots:
        for d, _, names in os.walk(os.path.expanduser(root)):
            if any(x in d for x in ("/.git", "/build", "__pycache__", "node_modules")):
                continue
            files += [os.path.join(d, n) for n in names if n.endswith(EXT) and n != "DESIGN.md"]
    rng = random.Random(1)
    rng.shuffle(files)
    out, fi = [], 0
    while len(out) < args.seqs and fi < len(files):
        try:
            text = open(files[fi], errors="replace").read()
        except OSError:
            text = ""
        fi += 1
        if len(text) < 2000:
            continue
        if len(out) % 3 == 0:  # chat-formatted: special tokens, a question, a thinking block
            text = ("<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\n"
                    f"Explain what this file does:\n\n{text[:6000]}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                    "Let me read the file carefully and summarize its purpose.\n</think>\n\nThis file")
        ids = tok.encode(text, add_special_tokens=False).ids[:args.len]
        if len(ids) >= 256:
            out.append(ids)
    with open(args.out, "w") as f:
        for ids in out:
            f.write(",".join(map(str, ids)) + "\n")
    print(f"{len(out)} sequences, {sum(map(len, out))} tokens -> {args.out}")


if __name__ == "__main__":
    main()
