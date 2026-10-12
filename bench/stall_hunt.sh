#!/usr/bin/env bash
# Hunts the stall seen during the engine's warm-up (docs/DESIGN.md "Three experiments on the prefill pass"): starts
# the engine again and again in several configurations under gdb, and when a start does not finish in time takes the
# host threads' stacks, the cards' use and the log before killing it. Dev box, production unloaded.
#   bench/stall_hunt.sh OUTDIR MINUTES [GDB]      (GDB: default ~/ugdb/gdb; ptrace needs gdb to be the parent)
# BASE (default ~/ie-head/build) is the build to compare with. One start takes ~50 s.
set -u
cd "$(dirname "$0")/.."
O=$1 MIN=$2 GDB=${3:-$HOME/ugdb/gdb} BASE=${BASE:-$HOME/ie-head/build}
mkdir -p "$O"
export QW_PLE_DIR=/mnt/llms/qwen3.8-flash-next-ple/ples_fp8
END=$(($(date +%s) + MIN * 60))
one() {  # name limit_s build env...
    local name=$1 limit=$2 build=$3 log
    shift 3
    log="$O/$name.log"
    env "$@" "$GDB" -batch -ex "handle all nostop noprint pass" -ex "handle SIGINT stop print nopass" -ex run \
        -ex "thread apply all bt 16" --args "$build/prefill_bench" --reps 1 300 >"$log" 2>&1 &
    local pid=$! t0
    t0=$(date +%s)
    while kill -0 $pid 2>/dev/null && [ $(($(date +%s) - t0)) -lt "$limit" ]; do sleep 2; done
    if kill -0 $pid 2>/dev/null; then
        {
            echo "=== STALL after ${limit}s"
            rocm-smi --showuse 2>/dev/null | grep "GPU use"
            top -b -H -n1 | grep prefill_bench | head -12
        } >>"$log" 2>&1
        pkill -INT -x prefill_bench
        sleep 30
        pkill -9 -x prefill_bench
        pkill -9 -f "ugdb/root/usr/bin/gdb"
        echo "$(date +%T) $name STALL" | tee -a "$O/index.txt"
    else
        echo "$(date +%T) $name ok $(($(date +%s) - t0)) s $(grep -o 'warmup done in [0-9.]* s' "$log")" >>"$O/index.txt"
    fi
}
i=0
while [ "$(date +%s)" -lt "$END" ]; do
    i=$((i + 1))
    one "base4k_$i" 150 "$BASE" QW_PREFILL_CHUNK=4096
    one "new4k_$i" 150 build QW_PREFILL_CHUNK=4096
    one "check4k_$i" 150 build QW_PREFILL_CHUNK=4096 QW_ROUTER16=check
    one "mb3big_$i" 180 build QW_PREFILL_CHUNK=21504 QW_PREFILL_MB=3
    one "basebig_$i" 180 "$BASE" QW_PREFILL_CHUNK=21504
done
echo "done $(date +%T)" >>"$O/index.txt"
