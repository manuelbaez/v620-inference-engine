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

## The spilled KV's PCIe counters (2026-10-17): `test_spill` and `test_pcie_traffic`

Built (commits 79ba602, 6d58a25) and in production since 2026-10-17 04:24 UTC, never run on the GPUs: the dev
container llm-experiments was stopped by a power outage. Run on the dev box (4 GPUs, a window; production unloaded):
`build/test_spill` and `QW_SPILL_STAGE=0 build/test_spill` (section 5: a 1,000-token prompt into the all-spilled slot
must count 12-13 x 500,500 rows of prefill direct reads without staging, 0 with it; everything else must stay
bit-identical, batched-prefill hash 287027c10cf4e970), and `build/test_pcie_traffic` (the new copy kinds count as
themselves only). If section 5 fails by a factor near 499,500 / 500,500, the dense lists exclude the query's own
position: fix the expected count, not the counter. Use the production checkpoint for it
(`QW_MODEL_DIR=QW_PLE_DIR=/mnt/llms/qwen3.8-flash-next-fp4 QW_DENSE_FP8=1 QW_F4_PACKED=1`): another checkpoint's 32 GB
table fills the host's page cache and the next production start places its spill pages badly (65%, prefill -45%).

## Many long streams on the all-in-RAM layout (2026-10-17)

Production runs `--slots 0 x 16` with `QW_SPILL_CACHE_MB=4864` (DESIGN "Slots with no VRAM part"); idle single
requests equal the 16,384-in-VRAM layout. Not measured: 8-16 streams decoding at 50-200k of context, where the larger
cache should pay. Run with `QW_SPILL_CACHE_STATS=1` (hit rates) on both layouts and compare decode tok/s and the
dashboard's summed spilled-KV reads.
