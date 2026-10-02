#!/usr/bin/env bash
# Runs one stage of the prefill suite on the dev box (4 GPUs, production unloaded and idle) and keeps every log.
#   bench/run_suite.sh correctness|pool|compute [outdir]
#
#   correctness  the unit and GPU tests of the pool, the replica sharing and the comm variants, with the switches
#                off and on (bit-exactness gates; ~20 min)
#   pool         pinning and collective microbenchmarks, then the cold long prefill A/B of the store settings
#                (64k tokens, then 100k for the main candidates; ~50 min)
#   pool2        the store settings again with the pool's final semantics and the combinations (~35 min)
#   stall        pins while the GPUs compute, and decode steps timed while arenas are pinned (~10 min)
#   compute      prefill speed by chunk size, at long context, the MoE balance, micro-batches off, the comm
#                variants end to end, a kernel trace of one chunk, and the determinism check (~60 min)
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
ALL="QW_POOL_ARENA=huge QW_RESERVE_AHEAD=2 QW_KV_PAIRS=1 QW_KV_PAIR_CHECK=1"

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
    run block_store_huge 1500 env QW_POOL_ARENA=huge QW_RESERVE_AHEAD=2 $B/test_block_store
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
        --configs nostore,old,noncoherent,huge,ahead2,pairs,pairs+check,reserve16,reserve16+ahead2,reserve10+pairs
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
stall)
    export QW_PLE_DIR=$FP8
    run pin_busy 900 $B/pin_bench 256 3 --busy
    run decode_stall 1800 $B/decode_stall_bench --steps 500 --threads 8
    ;;
compute)
    export QW_PLE_DIR=$FP8
    run prefill_pipeline_exact 1500 $B/test_prefill_pipeline
    for c in 1024 2048 4096 8192; do
        run chunk_$c 1500 env QW_PROFILE=1 QW_PREFILL_CHUNK=$c $B/prefill_bench --reps 3 2048 8192 16384
    done
    run chunk_4096_nosplit 1500 env QW_PROFILE=1 QW_NO_SPLIT=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 8192 16384
    run moe_balance 1500 env QW_MOE_STATS=1 QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 2 8192 16384
    run ctx_0 1500 env QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 4096 8192
    run ctx_30k 2400 env QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 30000 --reps 3 4096 8192
    run ctx_60k 2400 env QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 60000 --reps 3 4096 8192
    run ctx_120k 3600 env QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 120000 --reps 2 4096 8192
    run moe_balance_ctx60k 2400 env QW_MOE_STATS=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 60000 --reps 1 4096
    for rep in 1 2; do
        run pipeline_off_$rep 1500 env QW_PROFILE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 16384
        run pipeline_on_$rep 1500 env QW_PROFILE=1 QW_PREFILL_PIPELINE=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 16384
    done
    for rep in 1 2; do
        run comm_default_$rep 1500 env QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 8192 16384
        run comm_variants_$rep 1500 env QW_COMM_PUSH2D=1 QW_COMM_VECRECV=1 QW_PREFILL_CHUNK=4096 $B/prefill_bench --reps 3 8192 16384
    done
    for rep in 1 2; do  # decode with the collective variants: the 2D push also serves multi-row decode collectives
        run decode_plain_$rep 1500 env QW_COMM_PUSH2D=0 QW_COMM_VECRECV=0 $B/test_batch_decode --a 760,6511,314,9338,369 --b 1,2,3,4,5,6,7,8 --gen 48
        run decode_push2d_$rep 1500 env QW_COMM_PUSH2D=1 $B/test_batch_decode --a 760,6511,314,9338,369 --b 1,2,3,4,5,6,7,8 --gen 48
    done
    run kernel_trace 2400 rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT/trace" -o chunk -- \
        env QW_PREFILL_CHUNK=4096 $B/prefill_bench --ctx 60000 --reps 1 4096
    run determinism_default 2400 env QW_PREFILL_CHUNK=4096 $B/test_speculative --p 760,6511,314,9338,369 --p 1,2,3,4,5,6,7,8 \
        --gen 256 --k 5 --repeat 6
    run determinism_variants 2400 env QW_COMM_PUSH2D=1 QW_COMM_VECRECV=1 QW_PREFILL_CHUNK=4096 $B/test_speculative \
        --p 760,6511,314,9338,369 --p 1,2,3,4,5,6,7,8 --gen 256 --k 5 --repeat 6
    ;;
*)
    echo "usage: $0 correctness|pool|pool2|stall|compute [outdir]"
    exit 2
    ;;
esac
echo "done: $OUT" | tee -a "$OUT/index.txt"
