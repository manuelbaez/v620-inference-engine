"""Places the routed experts on the cards so that each card gets the same work in every layer.

The engine puts expert e of every layer on card e // 128. A card whose experts get more tokens than the others'
finishes its routed-expert phase later and the others wait for it at the next exchange. This reads the work of
each expert measured on a workload (QW_MOE_STATS=1 QW_MOE_COUNTS_FILE=counts.txt: tiles per layer and expert) and
writes a placement for QW_EXPERT_PERM: per layer, 128 experts per card with the cards' sums as equal as it finds
(largest first onto the lightest card with room, then swaps between the heaviest and the lightest card).

  python3 tools/expert_balance.py counts.txt perm.txt [--check other_counts.txt]
      --check: also print the balance the placement gives on another workload's counts
"""
import sys

CARDS, PER = 4, 128


def read(path):
    return [[int(v) for v in line.split()] for line in open(path) if line.strip()]


def loads(row, perm):
    return [sum(row[e] for e in perm[c * PER:(c + 1) * PER]) for c in range(CARDS)]


def place(row):
    cards = [[] for _ in range(CARDS)]
    total = [0] * CARDS
    for e in sorted(range(len(row)), key=lambda e: -row[e]):
        c = min((c for c in range(CARDS) if len(cards[c]) < PER), key=lambda c: total[c])
        cards[c].append(e)
        total[c] += row[e]
    for _ in range(10000):  # swaps that narrow the gap between the heaviest and the lightest card
        hi, lo = max(range(CARDS), key=lambda c: total[c]), min(range(CARDS), key=lambda c: total[c])
        gap = total[hi] - total[lo]
        best = None
        for i, a in enumerate(cards[hi]):
            for j, b in enumerate(cards[lo]):
                d = row[a] - row[b]
                if 0 < d < gap and (best is None or abs(gap - 2 * d) < best[0]):
                    best = (abs(gap - 2 * d), i, j, d)
        if best is None or best[0] >= gap:
            break
        _, i, j, d = best
        cards[hi][i], cards[lo][j] = cards[lo][j], cards[hi][i]
        total[hi] -= d
        total[lo] += d
    return [e for c in cards for e in sorted(c)]


def factor(counts, perms):  # busiest card / mean, summed over the layers (what the waiting follows)
    mx = mean = 0
    for row, perm in zip(counts, perms):
        ld = loads(row, perm)
        mx += max(ld)
        mean += sum(ld) / CARDS
    return mx / max(mean, 1)


def main():
    args = sys.argv[1:]
    check = args.pop(args.index("--check") + 1) if "--check" in args else None
    if check:
        args.remove("--check")
    counts = read(args[0])
    n = len(counts[0])
    assert n == CARDS * PER, f"{n} experts per layer"
    ident = [list(range(n))] * len(counts)
    perms = [place(row) for row in counts[:-1]] if len(counts) % 2 else [place(row) for row in counts]
    main_layers = len(perms)  # (an odd number of rows: the MTP layer last, which keeps its placement)
    with open(args[1], "w") as f:
        for p in perms:
            f.write(" ".join(map(str, p)) + "\n")
    moved = sum(e // PER != i // PER for p in perms for i, e in enumerate(p))
    print(f"{main_layers} layers, {moved} of {main_layers * n} experts move to another card")
    print(f"busiest card / mean on these counts: {factor(counts[:main_layers], ident):.4f} -> {factor(counts[:main_layers], perms):.4f}")
    if check:
        other = read(check)[:main_layers]
        print(f"on {check}: {factor(other, ident):.4f} -> {factor(other, perms):.4f}")


if __name__ == "__main__":
    main()
