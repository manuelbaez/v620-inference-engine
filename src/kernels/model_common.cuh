// Device functions shared by the prefill (_T) and batched decode (_B) kernels
// of each layer type, so both paths run the same math.
#pragma once

#include "core/shard.hpp"
#include "kernels/common.cuh"
#include "kernels/types.hpp"

namespace qw::gpu {

using namespace cfg;

// NeoX rotation of the first ROPE_DIM dims of buf (shared, n >= ROPE_DIM) in
// place, block-cooperative. The angle is reduced in double: pos * inv_freq is
// up to ~2.6e5 rad, far outside fast sin/cos's accurate range.
// Interleaved multimodal RoPE (the HF Qwen4Exp scheme for vision tokens):
// frequency i uses the height position if i % 3 == 1 and i < 33, the width
// position if i % 3 == 2 and i < 30 (sections 11 / 11 / 10), else the
// temporal one. Equal positions reduce to rope_shared.
__device__ inline void rope3_shared(float *buf, int64_t pt, int64_t ph, int64_t pw) {
    constexpr int half = ROPE_DIM / 2;
    __syncthreads();
    if (threadIdx.x < half) {
        const int i = threadIdx.x;
        const int64_t pos = (i % 3 == 1 && i < 33) ? ph : (i % 3 == 2 && i < 30) ? pw : pt;
        const double inv = pow(ROPE_THETA, -2.0 * i / ROPE_DIM);
        const double a = fmod(double(pos) * inv, 6.283185307179586);
        float sn, cs;
        sincosf(float(a), &sn, &cs);
        const float x1 = buf[i], x2 = buf[i + half];
        buf[i] = x1 * cs - x2 * sn;
        buf[i + half] = x2 * cs + x1 * sn;
    }
    __syncthreads();
}

__device__ inline void rope_shared(float *buf, int64_t pos) {
    constexpr int half = ROPE_DIM / 2;
    __syncthreads();
    if (threadIdx.x < half) {
        const int i = threadIdx.x;
        const double inv = pow(ROPE_THETA, -2.0 * i / ROPE_DIM);
        const double a = fmod(double(pos) * inv, 6.283185307179586);
        float sn, cs;
        sincosf(float(a), &sn, &cs);
        const float x1 = buf[i], x2 = buf[i + half];
        buf[i] = x1 * cs - x2 * sn;
        buf[i + half] = x2 * cs + x1 * sn;
    }
    __syncthreads();
}

// buf[0..n) = gemma_rmsnorm(src) with w1 = 1 + w. Block-cooperative.
__device__ inline void norm_shared(const float *src, const float *w1, float *buf, int n, float *red) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = src[i];
        ss += v * v;
    }
    ss = block_sum<8>(ss, red);
    const float r = rsqrtf(ss / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = src[i] * r * w1[i];
}

constexpr int QSA_LOCAL_HEADS = QH_L;  // 6

__device__ __forceinline__ uint32_t order_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive scan over the 1024 threads of a block.
__device__ inline int block_exscan_1024(int v, int *sh, int *total) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        int add = t >= off ? sh[t - off] : 0;
        __syncthreads();
        sh[t] += add;
        __syncthreads();
    }
    int incl = sh[t];
    if (total) *total = sh[1023];
    __syncthreads();
    return incl - v;
}

// One block of 1024 threads. Radix-select the 512th largest key, then an
// ordered compaction: everything above the threshold, plus ties by lowest
// group index, matching the reference's tie-break.
__device__ inline void select_body(const float *scores, int64_t pos, int32_t *list, int32_t *count) {
    const int nb = int((pos + 1) / IDX_RATIO);
    if (nb <= IDX_TOPK_BLOCKS) {  // dense: every token so far
        for (int i = threadIdx.x; i <= pos; i += blockDim.x) list[i] = i;
        if (threadIdx.x == 0) *count = int32_t(pos + 1);
        return;
    }
    __shared__ int hist[256];
    __shared__ int scan[1024];
    __shared__ uint32_t s_prefix;
    __shared__ int s_k;
    const int t = threadIdx.x;
    uint32_t prefix = 0, mask = 0;
    int k = IDX_TOPK_BLOCKS;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int i = t; i < nb; i += 1024) {
            const uint32_t key = order_key(scores[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xff], 1);
        }
        __syncthreads();
        if (t == 0) {
            int acc = 0, b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= k) break;
                acc += hist[b];
            }
            s_prefix = prefix | (uint32_t(b) << shift);
            s_k = k - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        k = s_k;
        mask |= 0xffu << shift;
        __syncthreads();
    }
    const uint32_t thr = prefix;  // the 512th largest key; take k ties
    const int per = (nb + 1023) / 1024;
    const int b0 = t * per, b1 = min(nb, b0 + per);
    int n_eq = 0;
    for (int i = b0; i < b1; ++i) n_eq += order_key(scores[i]) == thr;
    const int eq_before = block_exscan_1024(n_eq, scan, nullptr);
    int n_sel = 0, eq_seen = eq_before;
    for (int i = b0; i < b1; ++i) {
        const uint32_t key = order_key(scores[i]);
        if (key > thr)
            ++n_sel;
        else if (key == thr && eq_seen++ < k)
            ++n_sel;
    }
    const int out0 = block_exscan_1024(n_sel, scan, nullptr);
    int o = out0;
    eq_seen = eq_before;
    for (int i = b0; i < b1; ++i) {
        const uint32_t key = order_key(scores[i]);
        const bool take = key > thr || (key == thr && eq_seen++ < k);
        if (take) {
            for (int r = 0; r < IDX_RATIO; ++r) list[o * IDX_RATIO + r] = i * IDX_RATIO + r;
            ++o;
        }
    }
    if (t == 0) {
        int n = IDX_TOPK_BLOCKS * IDX_RATIO;
        for (int64_t p = int64_t(nb) * IDX_RATIO; p <= pos; ++p) list[n++] = int32_t(p);
        *count = n;
    }
}

constexpr int ATT_CHUNK = 64;
constexpr int ATT_BLOCKS = (IDX_BUDGET + IDX_RATIO - 1 + ATT_CHUNK - 1) / ATT_CHUNK;  // 33
constexpr int ATT_REC = HEAD_DIM + 2;                                                 // m, l, acc[256]
constexpr int ATT_PART = ATT_BLOCKS * QSA_LOCAL_HEADS * ATT_REC;                      // per query row

// Row `pos` of a per-position array that is split at vt: VRAM below, mapped host memory from vt on (KV spill,
// kernels/types.hpp). Unspilled slots have vt = NO_SPILL, so the branch is uniform and always the first.
template <typename T>
__device__ __forceinline__ T *kv_row(T *dev, T *host, int32_t vt, int64_t pos, int width) {
    return pos < vt ? dev + pos * width : host + (pos - vt) * width;
}

// Row of position pos in a slot's ring of raw indexer keys (kernels/types.hpp).
template <typename T>
__device__ __forceinline__ T *raw_row(T *ring, int32_t rows, int64_t pos) {
    return ring + (pos % rows) * IDX_DIM;
}

// Block b (256 threads) covers list entries [64b, 64b+64); writes one
// partial record (m, l, acc[256]) per head. K and V rows below vt are read from K / V, the others from Kh / Vh.
// With a SpillCache (c.tags non-null), spilled rows are read from it when cached, else from the host part and listed for
// the step's fill (slot and layer identify the rows' tags).
__device__ inline void attend_body(const uint4 *q, const uint4 *K, const uint4 *V, const uint4 *Kh, const uint4 *Vh,
                                   int32_t vt, const int32_t *list, int n, float *partial,
                                   const SpillCache &c = SpillCache{}, int slot = 0, int layer = 0) {
    QW_DCHECK(n >= 0 && n <= LIST_W);
    constexpr int NH = QSA_LOCAL_HEADS;
    __shared__ float sh[8][NH][ATT_REC];
    const int lane = threadIdx.x % WAVE, w = threadIdx.x / WAVE;
    const float scale = rsqrtf(float(HEAD_DIM));
    float qf[NH][8];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        uint4 v = q[h * (HEAD_DIM / 8) + lane];
        const __half *hv = reinterpret_cast<const __half *>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) qf[h][i] = __half2float(hv[i]) * scale;
    }
    float m[NH], l[NH], acc[NH][8];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[h][i] = 0.f;
    }
    const int base = blockIdx.x * ATT_CHUNK;
    const uint32_t epoch = c.tags && Kh ? c.epoch[slot] : 0;
    for (int i = base + w; i < min(n, base + ATT_CHUNK); i += 8) {
        const int tok = list[i];
        QW_DCHECK(tok >= 0 && tok < (1 << 20));
        const uint4 *kp = kv_row(K, Kh, vt, tok, HEAD_DIM / 8), *vp = kv_row(V, Vh, vt, tok, HEAD_DIM / 8);
        if (tok >= vt && epoch) {
            const uint32_t e = cache_entry(slot, layer, tok, c.entries);
            if (c.tags[e] == cache_tag(epoch, slot, layer, tok)) {
                kp = reinterpret_cast<const uint4 *>(c.K) + size_t(e) * (HEAD_DIM / 8);
                vp = reinterpret_cast<const uint4 *>(c.V) + size_t(e) * (HEAD_DIM / 8);
                if (c.stats && lane == 0) atomicAdd(c.stats, 1ull);
            } else if (lane == 0) {
                const uint32_t i = atomicAdd(c.miss_n, 1u);
                if (i < c.miss_cap)
                    c.miss[i] = make_uint2(e, uint32_t(tok) | (uint32_t(slot) << 25) | (uint32_t(layer) << 28));
                if (c.stats) atomicAdd(c.stats + 1, 1ull);
            }
        }
        uint4 kv = kp[lane];
        uint4 vv = vp[lane];
        const __half *kh = reinterpret_cast<const __half *>(&kv);
        const __half *vh = reinterpret_cast<const __half *>(&vv);
        float kf[8], vf[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            kf[d] = __half2float(kh[d]);
            vf[d] = __half2float(vh[d]);
        }
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            float s = 0.f;
#pragma unroll
            for (int d = 0; d < 8; ++d) s += qf[h][d] * kf[d];
            s = wave_sum(s);
            const float mn = fmaxf(m[h], s);
            const float c = __expf(m[h] - mn), e = __expf(s - mn);
            l[h] = l[h] * c + e;
#pragma unroll
            for (int d = 0; d < 8; ++d) acc[h][d] = acc[h][d] * c + e * vf[d];
            m[h] = mn;
        }
    }
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        if (lane == 0) {
            sh[w][h][0] = m[h];
            sh[w][h][1] = l[h];
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) sh[w][h][2 + lane * 8 + d] = acc[h][d];
    }
    __syncthreads();
    // combine the 8 waves: thread t handles (head, dim) pairs
    for (int idx = threadIdx.x; idx < NH * ATT_REC; idx += blockDim.x) {
        const int h = idx / ATT_REC, f = idx % ATT_REC;
        float M = -INFINITY;
        for (int ww = 0; ww < 8; ++ww) M = fmaxf(M, sh[ww][h][0]);
        float v = 0.f;
        if (f == 0) {
            v = M;
        } else {
            for (int ww = 0; ww < 8; ++ww)
                if (sh[ww][h][1] > 0.f) v += sh[ww][h][f] * __expf(sh[ww][h][0] - M);
        }
        partial[size_t(h) * ATT_REC + f] = v;
    }
}

// One block of HEAD_DIM threads per (query, head h): merge the ATT_BLOCKS
// partials of head h, apply the sigmoid gate. The per-block (m, l) pairs are
// staged in shared memory in one parallel read, so the accumulator loads are
// independent (no chain of dependent global reads).
template <typename G>
__device__ inline void combine_body(const float *partial, const G *gate, __half *out, int h) {
    constexpr int NH = QSA_LOCAL_HEADS;
    __shared__ float sm[ATT_BLOCKS], sl[ATT_BLOCKS];
    const int d = threadIdx.x;
    if (d < ATT_BLOCKS) {
        const float *r = partial + (size_t(d) * NH + h) * ATT_REC;
        sl[d] = r[1];
        sm[d] = r[1] > 0.f ? r[0] : -INFINITY;
    }
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int b = 0; b < ATT_BLOCKS; ++b) M = fmaxf(M, sm[b]);
    float L = 0.f, o = 0.f;
#pragma unroll
    for (int b = 0; b < ATT_BLOCKS; ++b)
        if (sl[b] > 0.f) {
            const float c = __expf(sm[b] - M);
            L += sl[b] * c;
            o += partial[(size_t(b) * NH + h) * ATT_REC + 2 + d] * c;
        }
    out[h * HEAD_DIM + d] = __float2half(o / L * sigmoid(float(gate[h * HEAD_DIM + d])));
}

// 512 threads (one per expert): softmax over the router logits, then the
// top-k by repeated argmax (ties: lowest index). top/topp are shared arrays
// of the caller; valid for all threads on return.
__device__ inline void route_body(const float *logits, int *top, float *topp) {
    __shared__ float red[16];
    __shared__ int redi[16];
    const int t = threadIdx.x, lane = t % WAVE, w = t / WAVE;
    const float x = logits[t];
    // softmax
    float mx = wave_max(x);
    if (lane == 0) red[w] = mx;
    __syncthreads();
    mx = red[0];
    for (int i = 1; i < 16; ++i) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float p = __expf(x - mx);
    const float sum = block_sum<16>(p, red);
    p /= sum;
    // top-k by repeated argmax (ties: lowest index)
    float cand = p;
    for (int k = 0; k < TOP_K; ++k) {
        float v = cand;
        int idx = t;
#pragma unroll
        for (int o = WAVE / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, WAVE);
            const int oi = __shfl_xor(idx, o, WAVE);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        __syncthreads();
        if (lane == 0) {
            red[w] = v;
            redi[w] = idx;
        }
        __syncthreads();
        if (t == 0) {
            float bv = red[0];
            int bi = redi[0];
            for (int i = 1; i < 16; ++i)
                if (red[i] > bv || (red[i] == bv && redi[i] < bi)) {
                    bv = red[i];
                    bi = redi[i];
                }
            top[k] = bi;
            topp[k] = bv;
        }
        __syncthreads();
        if (t == top[k]) cand = -1.f;
    }
    __syncthreads();
}

}  // namespace qw::gpu
