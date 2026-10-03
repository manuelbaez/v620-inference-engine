#!/usr/bin/env bash
# Runs one stage of the prefill suite on the dev box (4 GPUs, production unloaded and idle) and keeps every log.
#   bench/run_suite.sh correctness|pool|compute [outdir]
#
#   correctness  the unit and GPU tests of the pool, the replica sharing and the comm variants, with the switches
#                off and on (bit-exactness gates; ~20 min)
#   pool         pinning and collective microbenchmarks, then the cold long prefill A/B of the store settings
#                (64k tokens, then 100k for the main candidates; ~50 min)
#   pool2        the store settings again with the pool's final semantics and the combinations (~35 min)
#   pool3        a cache at its budget (arenas given back and pinned again), and the standing reserve (~40 min)
#   pool4        the 100k-token cold prefill again: the defaults, pins one at a time, a standing reserve (~30 min)
#   stall        pins while the GPUs compute, and decode steps timed while arenas are pinned (~10 min)
#   compute      the collective variant end to end, decode and determinism with it, the prefill input pipeline,
#                chunk size, the MoE balance, long context and a kernel trace of one chunk (~50 min)
#
# Run it from the build tree's parent (~/inference-engine). It never touches production: stop it first
# (/home/server/qw-tools/prod-idle-unload.sh) and bring production back afterwards (prod-smoke.sh).
set -u
cd "$(dirname "$0")/.."
STAGE=${1:-correctness}
OUT=${2:-$HOME/suite-$(date +%m%d-%H%M)-$STAGE}
mkdir -p "$OUT"
B=build
FP8=/mnt/llms/qwen3.8-flash-next-ple/ples_fp8
ALL="QW_POOL_ARENA=noncoherent QW_RESERVE_AHEAD=2 QW_KV_PAIRS=1 QW_KV_PAIR_CHECK=1"

run() {  # name timeout_s command...: the log is $OUT/name.log; the exit code and time go to the index
    local name=$1 t=$2
    shift 2
    echo "=== $name: $*" | tee -a "$OUT/index.txt"
    local t0
    t0=$(date +%s)
    timeout "$t" "$@" >"$OUT/$name.log" 2>&1
    echo "    exit $? in $(($(date +%s) - t0)) s" | tee -a "$OUT/index.txt"
}

{
    echo "stage $STAGE, $(date), host $(hostname)"
    git log --oneline -1 2>/dev/null
    free -g | head -2
    rocm-smi --showpower --showclocks --showtemp --showuse 2>/dev/null | head -60
} >"$OUT/environment.txt" 2>&1

case "$STAGE" in
correctness)
    run pinned_pool 300 $B/test_pinned_pool
    run comm_variants 600 $B/test_comm_variants
    run comm 600 $B/test_comm
    run kv_replicas 1200 $B/test_kv_replicas
    run block_store_default 1500 $B/test_block_store
    run block_store_pairs 1500 env QW_KV_PAIRS=1 QW_KV_PAIR_CHECK=1 $B/test_block_store
    run block_store_noncoherent 1500 env QW_POOL_ARENA=noncoherent QW_RESERVE_AHEAD=2 $B/test_block_store
    run block_store_all 1500 env $ALL $B/test_block_store
    run host_tier_all 1500 env $ALL $B/test_host_tier
    run disk_load_all 1500 env $ALL $B/test_disk_load
    run disk_load_all_threads 1500 env $ALL QW_LOAD_THREADS=4 $B/test_disk_load
    run snapshot_capture 1500 $B/test_snapshot_capture
    ;;
pool)
    run pin_bench 900 $B/pin_bench 256 4
    run comm_raw 900 $B/comm_bench raw
    run comm_engine 900 $B/comm_bench engine 200
    export QW_PLE_DIR=$FP8 QW_PREFILL_CHUNK=4096
    run cold_64k 5400 $B/cold_prefill_bench --tokens 65536 --reps 2 --host-gb 48 \
        --configs nostore,old,noncoherent,ahead2,pairs,pairs+check,reserve16,reserve16+ahead2,reserve10+pairs
    run cold_100k 5400 $B/cold_prefill_bench --tokens 100000 --reps 2 --host-gb 64 \
        --configs nostore,old,reserve16+ahead2,reserve10+pairs
    ;;
pool2)
    export QW_PLE_DIR=$FP8 QW_PREFILL_CHUNK=4096
    # pin cost varies a lot between runs on this host (the same configuration took 64.7 s once and 35.3 s next: 173 s
    # against 37 s of pinning), so four repetitions with a rotating start, and the spread is reported
    run cold_64k 7200 $B/cold_prefill_bench --tokens 65536 --reps 4 --host-gb 48 \
        --configs nostore,old,ahead2+pairs,nc+pairs,reserve10+nc+pairs,serial+ahead2+pairs
    run cold_100k 6000 $B/cold_prefill_bench --tokens 100000 --reps 2 --host-gb 64 \
        --configs nostore,old,ahead2+pairs,reserve10+nc+pairs
    # the same loads from disk, buffers taken on demand against buffers kept ready (thread time getting pinned buffers)
    run disk_load_default 1500 $B/test_disk_load
    run disk_load_reserve 1500 env QW_POOL_RESERVE_GB=8 QW_POOL_ARENA=noncoherent QW_KV_PAIRS=1 $B/test_disk_load
    ;;
pool3)
    # a cache at its budget: 64k tokens through a 6 GB store, so arenas empty as old chunks are evicted and the pool
    # either gives them back and pins new ones (churn) or keeps them
    export QW_PLE_DIR=$FP8 QW_PREFILL_CHUNK=4096
    run churn_64k 5400 $B/cold_prefill_bench --tokens 65536 --reps 4 --host-gb 6 \
        --configs nostore,old,ahead2+pairs
    # the standing reserve (the Session waits for it before the clock starts, as a server's first request would)
    run reserve_64k 5400 $B/cold_prefill_bench --tokens 65536 --reps 4 --host-gb 48 \
        --configs nostore,old,ahead2+pairs,reserve10+nc+pairs
    ;;
pool4)
    # production's long cold prompts: ahead2+pairs left +5.5% over no store at 100k tokens (pool2). nc+pairs is the
    # code's defaults (non-coherent arenas, 2 chunks ahead, pairs); pins one at a time and a standing reserve (the
    # Session waits for it) are the candidates for the rest
    export QW_PLE_DIR=$FP8 QW_PREFILL_CHUNK=4096
    run cold_100k_more 9000 $B/cold_prefill_bench --tokens 100000 --reps 3 --host-gb 64 \
        --configs nostore,nc+pairs,ahead2+pairs,serial+ahead2+pairs,reserve10+nc+pairs
    ;;
all)  # the stages after pool2, most valuable first
    "$0" compute "$OUT/compute"
    "$0" stall "$OUT/stall"
    "$0" pool3 "$OUT/pool3"
    "$0" pool4 "$OUT/pool4"
    ;;
final)  # the defaults as built, after a rebuild: the gates and the numbers the docs quote (~25 min)
    export QW_PLE_DIR=$FP8 QW_PREFILL_CHUNK=4096
    run f_pinned_pool 300 $B/test_pinned_pool
    run f_comm_variants 600 $B/test_comm_variants
    run f_pipeline_exact 1500 $B/test_prefill_pipeline
    run f_block_store 1500 $B/test_block_store
    run f_disk_load 1500 $B/test_disk_load
    run f_decode 1500 $B/test_batch_decode --a 760,6511,314,9338,369 --b 1,2,3,4,5,6,7,8 --gen 48
    run f_prefill 1500 env QW_PROFILE=1 $B/prefill_bench --reps 3 8192 16384
    run f_determinism 2400 $B/test_speculative --p 760,6511,314,9338,369 --p 1,2,3,4,5,6,7,8 --gen 256 --k 5 --repeat 6
    # real text for the MoE balance (routing depends on the tokens): the repo's docs and sources, tokenized
    run f_text_ids 300 "$HOME/qwenv/bin/python" tools/text_tokens.py /mnt/llms/qwen3.8-flash-next-awq/tokenizer.json "$OUT/text_ids.txt" \
        docs/ENGINE_GUIDE.md docs/DESIGN.md src/engine/prefill.hip src/engine/engine.hpp docs/MODEL.md
    run f_moe_text 1500 env QW_MOE_STATS=1 QW_PROFILE=1 $B/prefill_bench --reps 1 --tokens-file "$OUT/text_ids.txt" 16384
    ;;
disk)  # the disk tier's file formats through the tiers: v2 block files (the default), v1 with a buffer per rank (~10 min)
    export QW_PLE_DIR=$FP8
    run d_disk_tier 300 $B/test_disk_tier
    run d_block_store 1500 $B/test_block_store
    run d_host_tier 1500 $B/test_host_tier
    run d_disk_load 1500 $B/test_disk_load
    run d_host_tier_v1 1500 env QW_KV_PAIRS=0 $B/test_host_tier
    run d_disk_load_v1 1500 env QW_KV_PAIRS=0 $B/test_disk_load
    run d_kv_replicas 1500 $B/test_kv_replicas
    ;;
stall)
    export QW_PLE_DIR=$FP8
    run pin_busy 900 $B/pin_bench 256 3 --busy
    run decode_stall 1800 $B/decode_stall_bench --steps 500 --threads 8
    ;;
compute)
    # most valuable first, so that a window cut short still has the answers that decide defaults
    export QW_PLE_DIR=$FP8
    run prefill_pipeline_exact 1500 $B/test_prefill_pipeline
    # the settings meant to become defaults, through the tier tests
    NEWDEF="QW_POOL_ARENA=noncoherent QW_RESERVE_AHEAD=2 QW_KV_PAIRS=1 QW_KV_PAIR_CHECK=1 QW_COMM_PUSH2D=1"
    run gate_block_store 1500 env $NEWDEF $B/test_block_store
    run gate_host_tier 1500 env $NEWDEF $B/test_host_tier
    run gate_disk_load 1500 env $NEWDEF QW_LOAD_THREADS=2 $B/test_disk_load
    for rep in 1 2; do  # the 2D push of the big collectives, end to end (bit-identical results; chunk 4096 as in production)
        run comm_plain_$rep 1500 env QW_COMM_PUSH2D=0 QW_PREFILL_PIPELINE=0 QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 8192 16384
        run comm_push2d_$rep 1500 env QW_COMM_PUSH2D=1 QW_PREFILL_PIPELINE=0 QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 8192 16384
    done
    for rep in 1 2; do  # decode with and without it: the 2D push also serves multi-row decode collectives
        run decode_plain_$rep 1500 env QW_COMM_PUSH2D=0 $B/test_batch_decode --a 760,6511,314,9338,369 --b 1,2,3,4,5,6,7,8 --gen 48
        run decode_push2d_$rep 1500 env QW_COMM_PUSH2D=1 $B/test_batch_decode --a 760,6511,314,9338,369 --b 1,2,3,4,5,6,7,8 --gen 48
    done
    # the determinism check (a mismatch between repeats of a greedy run means wrong arithmetic), with the variant on
    run determinism_push2d 2400 env QW_COMM_PUSH2D=1 QW_PREFILL_CHUNK=4096 $B/test_speculative --p 760,6511,314,9338,369 --p 1,2,3,4,5,6,7,8 \
        --gen 256 --k 5 --repeat 6
    run determinism_plain 2400 env QW_COMM_PUSH2D=0 QW_PREFILL_CHUNK=4096 $B/test_speculative --p 760,6511,314,9338,369 --p 1,2,3,4,5,6,7,8 \
        --gen 256 --k 5 --repeat 6
    for rep in 1 2; do  # the next chunk's host inputs prepared while the GPUs run this one
        run pipeline_off_$rep 1500 env QW_COMM_PUSH2D=1 QW_PREFILL_PIPELINE=0 QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 16384
        run pipeline_on_$rep 1500 env QW_COMM_PUSH2D=1 QW_PROFILE=1 QW_PREFILL_PIPELINE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 16384
    done
    for c in 2048 4096 8192; do  # chunk (and so piece) size
        run chunk_$c 1500 env QW_COMM_PUSH2D=1 QW_PROFILE=1 QW_PREFILL_CHUNK=$c $B/prefill_bench --reps 3 2048 8192 16384
    done
    run moe_balance 1500 env QW_COMM_PUSH2D=1 QW_MOE_STATS=1 QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 2 16384
    run ctx_60k 2400 env QW_COMM_PUSH2D=1 QW_PROFILE=1 QW_MOE_STATS=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 60000 --reps 2 4096 8192
    run kernel_trace 2400 rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT/trace" -o chunk -- \
        env QW_COMM_PUSH2D=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 60000 --reps 1 4096
    ;;
*)
    echo "usage: $0 correctness|pool|pool2|pool3|pool4|stall|compute|final|disk|all [outdir]"
    exit 2
    ;;
esac
echo "done: $OUT" | tee -a "$OUT/index.txt"
