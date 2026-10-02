#include "comm/comm.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include "core/common.hpp"

namespace qw {

namespace {

constexpr int FLAG_STRIDE = 16;  // uint32 per flag slot (64 bytes)
constexpr long SPIN_LIMIT = 1L << 26;
constexpr size_t BLOCK_BYTES = 64 * 1024;  // push payload per block
constexpr int MAX_BLOCKS = 64;

void ck(hipError_t e, const char *what) {
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
        fail(std::string("comm: ") + what + ": " + hipGetErrorString(e));
}

struct PushArgs {
    const uint8_t *src;
    int rows;
    size_t row16;            // uint4 per row per destination
    size_t src_row_stride;   // bytes
    size_t src_dest_offset;  // bytes between destinations' parts (0 = broadcast)
    size_t slot;
    uint8_t *recv[RANKS];    // each destination's receive buffer
    uint32_t *flags[RANKS];  // each destination's flags
    uint32_t *counter;       // this rank's per-destination completion counters
    const uint32_t *base;    // this rank's sequence base
    uint32_t k;
    int rank;
    int concurrent;
    int rows2d;  // strided rows: all threads copy a flat range of (row, column) pairs, not row by row
};

// grid.x = blocks. All blocks serve one destination at a time, in an order
// rotated by rank (rank r sends to r, r+1, r+2, r+3): at any moment each GPU
// feeds one peer and no two GPUs target the same one. Writing to all peers at
// once (12 concurrent cross-NUMA flows) measured ~6x slower on this box.
// The last block to finish a destination raises its flag; every block fences
// its own stores before counting.
__device__ void push_body(const PushArgs &a, int block) {
    const uint32_t seq = *a.base + a.k;
    const int par = int(seq & 1);
    const int row16 = int(a.row16);
    // Small payloads (latency-bound) go to all peers at once: block b serves
    // destination b. Large ones rotate as described above.
    const bool concurrent = a.concurrent != 0;
    for (int dd = concurrent ? block : 0; dd < (concurrent ? block + 1 : RANKS); ++dd) {
        const int d = (a.rank + dd) % RANKS;
        const int nblk = concurrent ? 1 : int(gridDim.x);
        const int blk = concurrent ? 0 : block;
        uint4 *dst = reinterpret_cast<uint4 *>(a.recv[d] + (size_t(par) * RANKS + a.rank) * a.slot);
        const uint8_t *src = a.src + a.src_dest_offset * d;
        if (a.rows == 1) {
            const uint4 *s0 = reinterpret_cast<const uint4 *>(src);
            for (int c = blk * blockDim.x + threadIdx.x; c < row16; c += nblk * blockDim.x) dst[c] = s0[c];
        } else if (a.rows2d) {
            // Strided rows (a reduce-scatter's slice of every row): a row is 80-300 uint4, so a loop over rows
            // keeps most of a block's 256 threads idle. One flat index over (row, column) pairs keeps them all busy
            // (32-bit divisions: the 64-bit ones are emulated).
            const int total = a.rows * row16;
            const int stride16 = int(a.src_row_stride / 16);
            const uint4 *s0 = reinterpret_cast<const uint4 *>(src);
            for (int i = blk * int(blockDim.x) + int(threadIdx.x); i < total; i += nblk * int(blockDim.x)) {
                const int row = i / row16, c = i - row * row16;
                dst[i] = s0[row * stride16 + c];
            }
        } else {  // 32-bit row loop: 64-bit divisions are emulated
            for (int row = blk; row < a.rows; row += nblk) {
                const uint4 *s0 = reinterpret_cast<const uint4 *>(src + size_t(row) * a.src_row_stride);
                uint4 *d0 = dst + size_t(row) * row16;
                for (int c = threadIdx.x; c < row16; c += blockDim.x) d0[c] = s0[c];
            }
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint32_t done = atomicAdd(&a.counter[d], 1u);
            if (done == uint32_t(nblk) - 1) {
                a.counter[d] = 0;  // next use of this counter is ordered after this kernel
                __threadfence_system();
                __hip_atomic_store(a.flags[d] + (par * RANKS + a.rank) * FLAG_STRIDE, seq, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    }
}

__global__ __launch_bounds__(256) void push_kernel(PushArgs a) {
    push_body(a, int(blockIdx.x));
}

struct RecvArgs {
    const uint8_t *recv;  // this rank's receive buffer [parity][src][slot]
    uint32_t *flags;      // this rank's flags [parity][src]
    size_t slot;
    int rows;
    size_t row_bytes;  // payload bytes per row per source
    const uint32_t *base;
    uint32_t k;
    int op;
    int wait;  // poll the flags here (else a preceding wait kernel did)
    int rank;
    void *dst;
    uint32_t *err;  // shared by all ranks (coherent host memory)
};

// Polling is done by a handful of threads only: many blocks spinning on
// uncached flags flood the receiver's memory with reads and slow the very P2P
// writes they are waiting for (measured: 10x on a 10 MB all-reduce).
__device__ void wait_flags(const RecvArgs &a, int par, uint32_t seq) {
    if (threadIdx.x < RANKS) {
        const uint32_t *f = a.flags + (par * RANKS + threadIdx.x) * FLAG_STRIDE;
        long spins = 0;
        while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(2);
            // after one timeout every wait gives up at once: the engine fails in
            // seconds instead of timing out collective after collective
            if ((++spins & 1023) == 0 && __hip_atomic_load(a.err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) break;
            if (spins > SPIN_LIMIT) {
                // first timeout wins: waiting rank, missing source, sequence offset k
                const uint32_t code = 0x80000000u | (uint32_t(a.rank) << 28) | (uint32_t(threadIdx.x) << 24) |
                                      (a.k & 0xffffffu);
                uint32_t expected = 0;
                __hip_atomic_compare_exchange_strong(a.err, &expected, code, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_SYSTEM);
                break;
            }
        }
    }
}

__global__ void wait_kernel(RecvArgs a) {
    const uint32_t seq = *a.base + a.k;
    wait_flags(a, int(seq & 1), seq);
}

// Reduces or concatenates the 4 slots into dst; `blk` of `nblk` blocks.
__device__ void recv_body(const RecvArgs &a, int blk, int nblk) {
    const uint32_t seq = *a.base + a.k;
    const int par = int(seq & 1);
    const uint8_t *slots = a.recv + size_t(par) * RANKS * a.slot;
    if (a.wait) wait_flags(a, par, seq);
    __syncthreads();
    const size_t stride = size_t(nblk) * blockDim.x;
    const size_t start = blk * size_t(blockDim.x) + threadIdx.x;
    if (a.op == 0) {  // sum fp32 -> fp32, same layout
        const size_t n = size_t(a.rows) * a.row_bytes / 4;
        const float *s0 = reinterpret_cast<const float *>(slots);
        float *dst = static_cast<float *>(a.dst);
        const size_t qs = a.slot / 4;
        for (size_t i = start; i < n; i += stride) {
            float v = 0.f;
#pragma unroll
            for (int q = 0; q < RANKS; ++q) v += s0[q * qs + i];
            dst[i] = v;
        }
    } else if (a.op == 1) {  // sum fp16 -> fp32
        const size_t n = size_t(a.rows) * a.row_bytes / 2;
        const __half *s0 = reinterpret_cast<const __half *>(slots);
        float *dst = static_cast<float *>(a.dst);
        const size_t qs = a.slot / 2;
        for (size_t i = start; i < n; i += stride) {
            float v = 0.f;
#pragma unroll
            for (int q = 0; q < RANKS; ++q) v += __half2float(s0[q * qs + i]);
            dst[i] = v;
        }
    } else {  // concat, interleaved per row: dst[t][q][row_bytes]
        const int r16 = int(a.row_bytes / 16);
        uint4 *dst = static_cast<uint4 *>(a.dst);
        for (int row = blk; row < a.rows; row += nblk)
            for (int c = threadIdx.x; c < r16 * RANKS; c += blockDim.x) {
                const int q = c / r16, cc = c - q * r16;
                dst[size_t(row) * r16 * RANKS + c] =
                    reinterpret_cast<const uint4 *>(slots + q * a.slot)[size_t(row) * r16 + cc];
            }
    }
}

__global__ __launch_bounds__(256) void recv_kernel(RecvArgs a) {
    recv_body(a, int(blockIdx.x), int(gridDim.x));
}

// A small collective in one kernel (one launch instead of two): blocks 0..3
// push to one destination each, the last block waits for the 4 slots and
// reduces them. All 5 blocks are resident at once, so the waiting block never
// holds up the pushes.
__global__ __launch_bounds__(256) void exchange_small_kernel(PushArgs pa, RecvArgs ra) {
    if (int(blockIdx.x) < RANKS)
        push_body(pa, int(blockIdx.x));
    else
        recv_body(ra, 0, 1);
}

}  // namespace

Comm::Tuning Comm::Tuning::from_env() {
    Tuning t;
    t.push2d = std::getenv("QW_COMM_PUSH2D") && std::atoi(std::getenv("QW_COMM_PUSH2D")) != 0;
    return t;
}

Comm::Comm(const std::array<int, RANKS> &devices, size_t slot_bytes)
    : dev_(devices), slot_(slot_bytes), tune_(Tuning::from_env()) {
    QW_CHECK(slot_ % 16 == 0, "comm: slot size must be a multiple of 16");
    for (int r = 0; r < RANKS; ++r) {
        ck(hipSetDevice(dev_[size_t(r)]), "set device");
        for (int q = 0; q < RANKS; ++q)
            if (q != r) ck(hipDeviceEnablePeerAccess(dev_[size_t(q)], 0), "enable peer access");
        (void)hipGetLastError();  // clear the sticky already-enabled error
        void *p;
        ck(hipExtMallocWithFlags(&p, 2 * RANKS * slot_, hipDeviceMallocUncached), "alloc recv");
        recv_[size_t(r)] = static_cast<uint8_t *>(p);
        ck(hipExtMallocWithFlags(&p, 2 * RANKS * FLAG_STRIDE * 4, hipDeviceMallocUncached), "alloc flags");
        flags_[size_t(r)] = static_cast<uint32_t *>(p);
        ck(hipMemset(p, 0, 2 * RANKS * FLAG_STRIDE * 4), "clear flags");
        ck(hipMalloc(&p, 64), "alloc counters");
        ck(hipMemset(p, 0, 64), "clear counters");
        counters_[size_t(r)] = static_cast<uint32_t *>(p);
        ck(hipMalloc(&p, 64), "alloc base");
        ck(hipMemset(p, 0, 64), "clear base");
        base_[size_t(r)] = static_cast<uint32_t *>(p);
        ck(hipHostMalloc(&p, 64, 0), "alloc base staging");
        base_host_[size_t(r)] = static_cast<uint32_t *>(p);
        *base_host_[size_t(r)] = 0;
    }
    void *e;
    ck(hipHostMalloc(&e, 64, hipHostMallocCoherent), "alloc err");
    err_ = static_cast<uint32_t *>(e);
    *err_ = 0;
}

Comm::~Comm() {
    for (int r = 0; r < RANKS; ++r) {
        (void)hipSetDevice(dev_[size_t(r)]);
        (void)hipFree(recv_[size_t(r)]);
        (void)hipFree(flags_[size_t(r)]);
        (void)hipFree(counters_[size_t(r)]);
        (void)hipFree(base_[size_t(r)]);
        (void)hipHostFree(base_host_[size_t(r)]);
    }
    (void)hipHostFree(err_);
}

std::string Comm::describe_error(uint32_t e) {
    if (!e) return "no error";
    return "rank " + std::to_string((e >> 28) & 7) + " timed out waiting for rank " + std::to_string((e >> 24) & 15) +
           " at collective offset " + std::to_string(e & 0xffffffu);
}

namespace {
__global__ void bump_base_kernel(uint32_t *base, uint32_t delta) {
    *base += delta;
}
}  // namespace

void Comm::bump_base(int r, uint32_t delta, hipStream_t s) {
    bump_base_kernel<<<1, 1, 0, s>>>(base_[size_t(r)], delta);
}

void Comm::set_base(int r, uint32_t base, hipStream_t s) {
    // The staging word is only rewritten after the previous step synchronized.
    *base_host_[size_t(r)] = base;
    ck(hipMemcpyAsync(base_[size_t(r)], base_host_[size_t(r)], 4, hipMemcpyHostToDevice, s), "set base");
}

void Comm::exchange(int r, const void *src, int rows, size_t row_bytes, size_t src_row_stride, size_t src_dest_offset,
                    Op op, void *dst, uint32_t k, hipStream_t s) {
    const size_t bytes = size_t(rows) * row_bytes;
    QW_CHECK(row_bytes % 16 == 0 && src_row_stride % 16 == 0 && bytes <= slot_, "comm: bad message size");
    QW_CHECK(k > 0, "comm: sequence offsets start at 1");
    PushArgs pa;
    pa.src = static_cast<const uint8_t *>(src);
    // Large payloads (prefill) whose rows are contiguous in the source and in
    // the slot are pushed as one flat copy: a block per 1-2 KB row leaves most
    // threads idle (prefill +4-10%). Small (decode) payloads keep row blocks.
    static const bool flat_ok = !std::getenv("QW_COMM_ROWWISE");
    static const size_t flat_min =
        std::getenv("QW_COMM_CONCURRENT_MAX") ? size_t(std::atol(std::getenv("QW_COMM_CONCURRENT_MAX"))) : 32768;
    const bool flat = flat_ok && rows > 1 && src_row_stride == row_bytes && bytes > flat_min;
    pa.rows2d = tune_.push2d && !flat && rows > 1 && (row_bytes / 16) < 256;
    pa.rows = flat ? 1 : rows;
    pa.row16 = (flat ? bytes : row_bytes) / 16;
    pa.src_row_stride = src_row_stride;
    pa.src_dest_offset = src_dest_offset;
    pa.slot = slot_;
    for (int d = 0; d < RANKS; ++d) {
        pa.recv[d] = recv_[size_t(d)];
        pa.flags[d] = flags_[size_t(d)];
    }
    pa.counter = counters_[size_t(r)];
    pa.base = base_[size_t(r)];
    pa.k = k;
    pa.rank = r;
    const int blocks = int(std::min<size_t>(pa.rows > 1 ? std::min<size_t>(MAX_BLOCKS, size_t(rows)) : MAX_BLOCKS,
                                            (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES));
    // Measured (tests/gpu/test_comm): all peers at once wins up to ~32 KB;
    // above that, rotating destinations is up to 2x faster.
    static const size_t concurrent_max =
        std::getenv("QW_COMM_CONCURRENT_MAX") ? size_t(std::atol(std::getenv("QW_COMM_CONCURRENT_MAX"))) : 32768;
    pa.concurrent = bytes <= concurrent_max;
    RecvArgs ra;
    ra.recv = recv_[size_t(r)];
    ra.flags = flags_[size_t(r)];
    ra.slot = slot_;
    ra.rows = rows;
    ra.row_bytes = row_bytes;
    ra.base = base_[size_t(r)];
    ra.k = k;
    ra.op = int(op);
    ra.rank = r;
    ra.dst = dst;
    ra.err = err_;
    static const size_t recv_bytes =
        std::getenv("QW_COMM_RECV_BYTES") ? size_t(std::atol(std::getenv("QW_COMM_RECV_BYTES"))) : 65536;
    const int rblocks = std::max(1, int(std::min<size_t>(256, (bytes * RANKS + recv_bytes - 1) / recv_bytes)));
    ra.wait = rblocks == 1;
    static const bool split = std::getenv("QW_COMM_SPLIT") != nullptr;  // two launches, as before
    if (pa.concurrent && ra.wait && !split) {
        exchange_small_kernel<<<RANKS + 1, 256, 0, s>>>(pa, ra);
        return;
    }
    push_kernel<<<pa.concurrent ? RANKS : std::max(blocks, 1), 256, 0, s>>>(pa);
    if (!ra.wait) wait_kernel<<<1, 64, 0, s>>>(ra);
    recv_kernel<<<rblocks, 256, 0, s>>>(ra);
}

void Comm::allreduce(int r, const void *src, float *dst, int n, uint32_t k, hipStream_t s, int rows, bool src_f16) {
    const size_t es = src_f16 ? 2 : 4;
    exchange(r, src, rows, size_t(n) * es, size_t(n) * es, 0, src_f16 ? SUM_F16 : SUM_F32, dst, k, s);
}

void Comm::reduce_scatter(int r, const void *src, float *dst, int n, uint32_t k, hipStream_t s, int rows,
                          bool src_f16) {
    const size_t es = src_f16 ? 2 : 4;
    exchange(r, src, rows, size_t(n) * es, size_t(n) * RANKS * es, size_t(n) * es, src_f16 ? SUM_F16 : SUM_F32, dst, k,
             s);
}

void Comm::allgather(int r, const void *src, void *dst, size_t bytes, uint32_t k, hipStream_t s, int rows) {
    exchange(r, src, rows, bytes, bytes, 0, CONCAT, dst, k, s);
}

}  // namespace qw
