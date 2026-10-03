# Tests and measurements still to do

Benchmarks that did not give a result, and measurements the docs name as open. Each entry says how to run it, what was
seen, and what to check first. Delete an entry when it has a result (and put the result in DESIGN.md).

## `decode_stall_bench`: does pinning stall decoding?

`build/decode_stall_bench --steps 500 --threads 8` (stage `stall` of `bench/run_suite.sh`, needs the 4 GPUs and
`QW_PLE_DIR`). Times single-stream decode steps one by one while 8 background threads pin and unpin 256 MB arenas
(default flags, non-coherent, each also one at a time), against no pinning. It answers the question the prefill results
leave open: pins slow a cold prefill through the cache (DESIGN.md "Saving to the prefix cache and the prefill
collectives"), but is a request that decodes meanwhile slowed too? Production saw 1-4 s steps without any load.

Seen (2026-10-03, stage `suite-1002-all/stall`): killed by the 1,800 s timeout with no output after
`vision tower: 0.90 GB per card`; the other benchmarks print `warmup done in N s` there, this one never did, so it
stopped inside the engine's warm-up, before the first measured row.

Check first: the benchmark makes one slot of 4,096 tokens (`opt.slot_tokens = {4096}`) and leaves the prefill chunk at
its default 8,192; the warm-up prefill is probably larger than the slot (the others export `QW_PREFILL_CHUNK=4096` or use
slots of 32k+). Set `opt.prefill_chunk = 4096` (or a slot of 8,192+) and run it with `--steps 50` first.

## `pin_bench --busy`: pins while every GPU computes

`build/pin_bench 256 3 --busy` (same stage). Pins arenas with 1, 4 and 8 threads while one thread per GPU keeps compute
kernels running, and a probe thread times small `mmap`/`munmap` calls.

Seen: killed by the 900 s timeout after 5 of the 12 rows. The rows it printed: `hipHostMalloc` 0.047 s per arena at 1
thread, 0.054 at 4, 0.091 at 8 (probe stalls ~45 ms, as on an idle box); non-coherent 0.043 / 0.061 s. The wall time per
row was 150-430 s for 3 rounds of 0.05 s pins, so almost all of it was something else.

Check first (a guess): `hipHostFree` waits for the device's streams, and the busy threads enqueue kernels back to back
without bounding the queue, so the backlog grows and every unpin waits for it. If so the benchmark, not the pool, is at
fault: bound the kernel queue (synchronize every few launches) before reading anything into the 0.047 s figures.

## Cold prefill with the n-gram table locked

Why a cold 100k prompt through the cache still costs +49% over no store in a bad host state (DESIGN.md). On the dev box
the table is page cache (`ulimit -l` 8 MB, "mlock refused"), production locks it. Run `cold_prefill_bench --tokens 100000
--configs nostore,old,nc+pairs` with `ulimit -l unlimited` (stage `pool4`'s command), once with `QW_PLE_PIN` on.

## Others named as open in DESIGN.md

A standing pinned reserve that refills only while the engine is idle (the Session knows); the QSA attention kernel (20.7%
of GPU time at 60k of context); the SSD for the disk tier.
