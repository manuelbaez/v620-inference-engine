#!/usr/bin/env python3
"""Token ids of real text for benchmarks (prefill_bench --tokens-file): text_tokens.py TOKENIZER_JSON OUT FILE...
Needs the `tokenizers` package. Writes the ids of all files, one after the other, separated by commas."""
import sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
ids = []
for path in sys.argv[3:]:
    ids += tok.encode(open(path, errors="replace").read()).ids
open(sys.argv[2], "w").write(",".join(map(str, ids)))
print(f"{len(ids)} token ids from {len(sys.argv) - 3} files -> {sys.argv[2]}")
