"""Paired comparison of two bench_tasks.py result files: accuracy per task and McNemar's exact test."""
import json, sys
from math import comb
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
na, nb = sys.argv[3] if len(sys.argv) > 3 else "A", sys.argv[4] if len(sys.argv) > 4 else "B"
for t in ("gsm8k", "mmlu", "arc", "all"):
    keys = [k for k in A if k in B and (t == "all" or k.startswith(t + ":"))]
    a = sum(A[k] for k in keys); b = sum(B[k] for k in keys)
    only_a = sum(A[k] and not B[k] for k in keys); only_b = sum(B[k] and not A[k] for k in keys)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    print(f"{t:6s} n={len(keys):5d}  {na} {100*a/len(keys):5.1f}%  {nb} {100*b/len(keys):5.1f}%  "
          f"({na} only {only_a}, {nb} only {only_b}; McNemar p={p:.3f})")
