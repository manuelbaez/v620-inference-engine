#include "kernels/moe.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

__global__ __launch_bounds__(512) void moe_route_T_kernel(const float *logits, int first, int32_t *tok_e, float *tok_w,
                                                          int32_t *counts, unsigned long long *use) {
    __shared__ int top[TOP_K];
    __shared__ float topp[TOP_K];
    const int t = blockIdx.x;
    route_body(logits + size_t(t) * N_EXPERTS, top, topp);
    if (threadIdx.x < TOP_K) {
        float sum = 0.f;
        for (int k = 0; k < TOP_K; ++k) sum += topp[k];
        const int k = threadIdx.x, e = top[k];
        const bool local = e >= first && e < first + EXP_L;
        tok_e[t * TOP_K + k] = local ? e - first : -1;
        tok_w[t * TOP_K + k] = topp[k] / sum;
        if (local) atomicAdd(&counts[e - first], 1);
        if (local && use) atomicAdd(use, 1ull);
    }
}

void moe_route_T(const float *logits, int T, int first, int32_t *tok_e, float *tok_w, int32_t *counts, hipStream_t s,
                 unsigned long long *use) {
    moe_route_T_kernel<<<T, N_EXPERTS, 0, s>>>(logits, first, tok_e, tok_w, counts, use);
}

__global__ void moe_offsets_kernel(const int32_t *counts, int32_t *offsets, int32_t *fill) {
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < EXP_L; ++e) {
            offsets[e] = acc;
            acc += counts[e];
            fill[e] = 0;
        }
        offsets[EXP_L] = acc;
    }
}

__global__ void moe_scatter_kernel(const int32_t *tok_e, const int32_t *offsets, int32_t *fill, int T,
                                   int32_t *tok_slot, int32_t *pair_tok) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * TOP_K) return;
    const int e = tok_e[i];
    if (e < 0) {
        tok_slot[i] = -1;
        return;
    }
    const int slot = offsets[e] + atomicAdd(&fill[e], 1);
    tok_slot[i] = slot;
    pair_tok[slot] = i / TOP_K;
}

void moe_scatter_T(const int32_t *tok_e, const int32_t *counts, int T, int32_t *offsets, int32_t *fill,
                   int32_t *tok_slot, int32_t *pair_tok, hipStream_t s) {
    moe_offsets_kernel<<<1, 64, 0, s>>>(counts, offsets, fill);
    moe_scatter_kernel<<<(T * TOP_K + 255) / 256, 256, 0, s>>>(tok_e, offsets, fill, T, tok_slot, pair_tok);
}

// Decode rows (M <= 32), one block: wave w routes row w (top-10 of its 512
// logits by repeated wave argmax, ties to the lowest index; weights are the
// softmax renormalized over the top-10, so the full softmax cancels out),
// then the block counts pairs per local expert, scans the offsets and gives
// each pair a deterministic slot (row-major order within an expert).
__global__ __launch_bounds__(1024) void moe_route_B_kernel(const float *logits, int M, int first, int32_t *counts,
                                                           int32_t *offsets, float *tok_w, int32_t *tok_slot,
                                                           int32_t *pair_tok, unsigned long long *use) {
    constexpr int PER = N_EXPERTS / WAVE;  // 16 logits per lane
    __shared__ int s_e[32 * TOP_K];
    __shared__ int s_cnt[EXP_L], s_off[EXP_L + 1];
    const int t = threadIdx.x, lane = t % WAVE, w = t / WAVE;
    if (t < EXP_L) s_cnt[t] = 0;
    if (w < M) {
        float v[PER];
        const float *l = logits + size_t(w) * N_EXPERTS;
#pragma unroll
        for (int i = 0; i < PER; ++i) v[i] = l[lane + WAVE * i];
        float topv = 0.f, lmax = 0.f, sum = 0.f;
        int topi = 0;
        for (int k = 0; k < TOP_K; ++k) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < PER; ++i)
                if (v[i] > bv) {  // ascending index per lane: the first max wins
                    bv = v[i];
                    bi = lane + WAVE * i;
                }
#pragma unroll
            for (int o = WAVE / 2; o > 0; o >>= 1) {
                const float ov = __shfl_xor(bv, o, WAVE);
                const int oi = __shfl_xor(bi, o, WAVE);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == bi % WAVE) {
#pragma unroll
                for (int i = 0; i < PER; ++i)
                    if (i == bi / WAVE) v[i] = -INFINITY;
            }
            if (k == 0) lmax = bv;
            const float e = __expf(bv - lmax);
            sum += e;
            if (lane == k) {
                topv = e;
                topi = bi;
            }
        }
        if (lane < TOP_K) {
            const bool local = topi >= first && topi < first + EXP_L;
            s_e[w * TOP_K + lane] = local ? topi - first : -1;
            tok_w[w * TOP_K + lane] = topv / sum;
        }
    }
    __syncthreads();
    const int n_pairs = M * TOP_K;
    if (t < n_pairs && s_e[t] >= 0) atomicAdd(&s_cnt[s_e[t]], 1);
    __syncthreads();
    if (w == 0) {  // exclusive scan of 128 counts: 4 per lane
        int c[EXP_L / WAVE], run = 0;
#pragma unroll
        for (int i = 0; i < EXP_L / WAVE; ++i) {
            c[i] = s_cnt[lane * (EXP_L / WAVE) + i];
            run += c[i];
        }
        int incl = run;
#pragma unroll
        for (int o = 1; o < WAVE; o <<= 1) {
            const int n = __shfl_up(incl, o, WAVE);
            if (lane >= o) incl += n;
        }
        int acc = incl - run;
#pragma unroll
        for (int i = 0; i < EXP_L / WAVE; ++i) {
            const int e = lane * (EXP_L / WAVE) + i;
            s_off[e] = acc;
            offsets[e] = acc;
            counts[e] = c[i];
            acc += c[i];
        }
        if (lane == WAVE - 1) {
            s_off[EXP_L] = acc;
            offsets[EXP_L] = acc;
            if (use) atomicAdd(use, static_cast<unsigned long long>(acc));
        }
    }
    __syncthreads();
    if (t < n_pairs) {
        const int e = s_e[t];
        if (e < 0) {
            tok_slot[t] = -1;
        } else {
            int before = 0;
            for (int j = 0; j < t; ++j) before += s_e[j] == e;
            const int slot = s_off[e] + before;
            tok_slot[t] = slot;
            pair_tok[slot] = t / TOP_K;
        }
    }
}

void moe_route_B(const float *logits, int M, int first, int32_t *counts, int32_t *offsets, float *tok_w,
                 int32_t *tok_slot, int32_t *pair_tok, hipStream_t s, unsigned long long *use) {
    moe_route_B_kernel<<<1, 1024, 0, s>>>(logits, M, first, counts, offsets, tok_w, tok_slot, pair_tok, use);
}

__global__ void moe_combine_T_kernel(const __half *yp, const int32_t *tok_slot, const float *tok_w,
                                     const __half *shared, __half *out) {
    const int t = blockIdx.x;
    for (int n = threadIdx.x; n < H; n += blockDim.x) {
        float acc = __half2float(shared[size_t(t) * H + n]);
        for (int k = 0; k < TOP_K; ++k) {
            const int slot = tok_slot[t * TOP_K + k];
            if (slot >= 0) acc += tok_w[t * TOP_K + k] * __half2float(yp[size_t(slot) * H + n]);
        }
        out[size_t(t) * H + n] = __float2half(acc);
    }
}

void moe_combine_T(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const uint16_t *shared, int T,
                   uint16_t *out, hipStream_t s) {
    moe_combine_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(yp), tok_slot, tok_w,
                                           reinterpret_cast<const __half *>(shared), reinterpret_cast<__half *>(out));
}

__global__ void swiglu_T_kernel(const float *v, int ldv, int n, int T, __half *h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    const int t = i / n, j = i % n;
    const float *row = v + size_t(t) * ldv;
    h[i] = __float2half(silu(row[j]) * row[n + j] * sigmoid(row[2 * n]));
}

void swiglu_T(const float *v, int ldv, int n, int T, uint16_t *h, hipStream_t s) {
    swiglu_T_kernel<<<(T * n + 255) / 256, 256, 0, s>>>(v, ldv, n, T, reinterpret_cast<__half *>(h));
}

// ---- batched decode rows
// grid (M, H / 256): one thread per output column.
__global__ __launch_bounds__(256) void moe_combine_B_kernel(const __half *yp, const int32_t *tok_slot,
                                                            const float *tok_w, const float *shared, float *out) {
    const int r = blockIdx.x, n = blockIdx.y * blockDim.x + threadIdx.x;
    int slot[TOP_K];
    float wt[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        slot[k] = tok_slot[r * TOP_K + k];
        wt[k] = tok_w[r * TOP_K + k];
    }
    float acc = shared[size_t(r) * H + n];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        QW_DCHECK(slot[k] < 32 * TOP_K);
        if (slot[k] >= 0) acc += wt[k] * __half2float(yp[size_t(slot[k]) * H + n]);
    }
    out[size_t(r) * H + n] = acc;
}

void moe_combine_B(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const float *shared, int M,
                   float *out, hipStream_t s) {
    moe_combine_B_kernel<<<dim3(M, H / 256), 256, 0, s>>>(reinterpret_cast<const __half *>(yp), tok_slot, tok_w, shared,
                                                          out);
}

}  // namespace qw::gpu
