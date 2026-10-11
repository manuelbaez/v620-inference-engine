# Prefix-cache behaviour from the engine's request log (journalctl -t qw): per period, requests, prompt tokens, the share
# served from the cache, and "cold repeats": requests of 4,000+ prompt tokens with nothing cached whose length is
# within 5% above an earlier request's of the last 30 minutes (a follow-up turn that should have hit the cache).
#   journalctl -t qw --since ... --no-pager -o short-iso | cache_report.py [hours per period=6]
import re, sys, datetime, collections
H = float(sys.argv[1]) if len(sys.argv) > 1 else 6
pat = re.compile(r"^(\S+) .*request \w+: (\w+), prompt (\d+) \((\d+) cached\)")
per = collections.OrderedDict(); recent = []
for line in sys.stdin:
    m = pat.match(line)
    if not m: continue
    t = datetime.datetime.fromisoformat(m.group(1)); fin, p, c = m.group(2), int(m.group(3)), int(m.group(4))
    key = t.replace(minute=0, second=0, microsecond=0, hour=int(t.hour // H * H))
    d = per.setdefault(key, [0, 0, 0, 0, 0])
    d[0] += 1; d[1] += p; d[2] += c
    recent = [(tt, pp) for tt, pp in recent if (t - tt).total_seconds() < 1800]
    if p >= 4000 and fin != "abort":
        d[4] += 1
        if c == 0 and any(pp <= p <= pp * 1.05 + 400 and pp >= 3000 for tt, pp in recent): d[3] += 1
    recent.append((t, p))
print(f"{'period (UTC)':17s} {'requests':>8s} {'prompt tok':>11s} {'from cache':>10s} {'cold repeats / long requests':>30s}")
for k, d in per.items():
    print(f"{k:%m-%d %H:%M}       {d[0]:8d} {d[1]:11d} {100 * d[2] / max(d[1], 1):9.1f}% {d[3]:14d} / {d[4]}")
