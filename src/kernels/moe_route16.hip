// Prefill routing from fp16 router logits. rocBLAS computes the router's GEMM about four times faster with fp16
// output than with fp32 output, but fp16 logits flip near-tie expert choices; here the fp16 logits only pick
// ROUTE_CAND candidates per token, whose logits are then computed again in fp32 from the router's weights, and the
// top-10 and their weights come from those. A token whose 10th exact logit is not clear of every expert left out
// (by more than the fp16 logits can be off) gets all 512 computed.
#include "kernels/moe.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

constexpr int ROUTE_CAND = 16;  // one per wave of the block
static_assert(ROUTE_CAND * WAVE == N_EXPERTS && H % (2 * WAVE) == 0);

// One round of a block-wide argmax over the threads' `mine` (ties: lowest index): the winner goes to *out_i and
// *out_v and is taken out of the running. Valid for all threads on return.
__device__ __forceinline__ void take_max(float &mine, float *red, int *redi, int *out_i, float *out_v) {
    const int t = threadIdx.x, lane = t % WAVE, w = t / WAVE;
    float v = mine;
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
        *out_i = bi;
        *out_v = bv;
    }
    __syncthreads();
    if (t == *out_i) mine = -INFINITY;
}

// This wave's fp32 logit of expert e for the token with input x (both fp16 pairs), valid on every lane.
__device__ __forceinline__ float exact_logit(const uint32_t *W, const uint32_t *x, int e) {
    const uint32_t *w = W + size_t(e) * (H / 2);
    const int lane = threadIdx.x % WAVE;
    float acc = 0.f;
    for (int i = lane; i < H / 2; i += WAVE) acc = dot2(w[i], x[i], acc);
    return wave_sum(acc);
}

__global__ __launch_bounds__(512) void moe_route16_T_kernel(const __half *l16, const uint32_t *W, const uint32_t *X,
                                                            int ldx2, int first, int32_t *tok_e, float *tok_w,
                                                            int32_t *counts, unsigned long long *use, unsigned *wide) {
    __shared__ float red[16];
    __shared__ int redi[16];
    __shared__ int cand[ROUTE_CAND + 1];
    __shared__ float cand16[ROUTE_CAND + 1];
    __shared__ float exact[N_EXPERTS];
    __shared__ int top[TOP_K];
    __shared__ float topv[TOP_K];
    __shared__ int all;
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid % WAVE, w = tid / WAVE;
    const uint32_t *x = X + size_t(t) * ldx2;
    float mine = __half2float(l16[size_t(t) * N_EXPERTS + tid]);
    for (int k = 0; k <= ROUTE_CAND; ++k) take_max(mine, red, redi, &cand[k], &cand16[k]);  // (the last: best left out)
    const float e = exact_logit(W, x, cand[w]);
    if (lane == 0) exact[w] = e;
    __syncthreads();
    if (tid == 0) {
        bool used[ROUTE_CAND] = {};
        for (int k = 0; k < TOP_K; ++k) {
            int b = -1;
            for (int c = 0; c < ROUTE_CAND; ++c)
                if (!used[c] && (b < 0 || exact[c] > exact[b] || (exact[c] == exact[b] && cand[c] < cand[b]))) b = c;
            used[b] = true;
            top[k] = cand[b];
            topv[k] = exact[b];
        }
        // a fp16 logit is within half a unit in its last place (2^-11 of its size) of the fp32 one; four times that
        const float rest = cand16[ROUTE_CAND];
        all = !(rest + fabsf(rest) * (1.f / 512) + 1e-3f < topv[TOP_K - 1]);
    }
    __syncthreads();
    if (all) {
        for (int r = 0; r < WAVE; ++r) {
            const float v = exact_logit(W, x, r * ROUTE_CAND + w);
            if (lane == 0) exact[r * ROUTE_CAND + w] = v;
        }
        __syncthreads();
        mine = exact[tid];
        for (int k = 0; k < TOP_K; ++k) take_max(mine, red, redi, &top[k], &topv[k]);
        if (tid == 0 && wide) atomicAdd(wide, 1u);
    }
    if (tid < TOP_K) {  // softmax over the top-10 (the full softmax cancels out)
        float sum = 0.f;
        for (int k = 0; k < TOP_K; ++k) sum += __expf(topv[k] - topv[0]);
        const int k = tid, ex = top[k];
        const bool local = ex >= first && ex < first + EXP_L;
        tok_e[t * TOP_K + k] = local ? ex - first : -1;
        tok_w[t * TOP_K + k] = __expf(topv[k] - topv[0]) / sum;
        if (local) atomicAdd(&counts[ex - first], 1);
        if (local && use) atomicAdd(use, 1ull);
    }
}

void moe_route16_T(const uint16_t *logits16, const uint16_t *W, const uint16_t *X, int ldx, int T, int first,
                   int32_t *tok_e, float *tok_w, int32_t *counts, hipStream_t s, unsigned long long *use,
                   unsigned *wide) {
    moe_route16_T_kernel<<<T, N_EXPERTS, 0, s>>>(reinterpret_cast<const __half *>(logits16),
                                                 reinterpret_cast<const uint32_t *>(W),
                                                 reinterpret_cast<const uint32_t *>(X), ldx / 2, first, tok_e, tok_w,
                                                 counts, use, wide);
}

// QW_ROUTER16=check: per token, whether the two routings chose the same experts, and the largest weight difference.
// stats: [0] tokens, [1] tokens with another expert list, [2] bits of the largest |weight difference|.
__global__ void moe_route_diff_kernel(const int32_t *ea, const float *wa, const int32_t *eb, const float *wb, int T,
                                      unsigned *stats) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    bool same = true;
    float d = 0.f;
    for (int k = 0; k < TOP_K; ++k) {
        same = same && ea[t * TOP_K + k] == eb[t * TOP_K + k];
        d = fmaxf(d, fabsf(wa[t * TOP_K + k] - wb[t * TOP_K + k]));
    }
    atomicAdd(&stats[0], 1u);
    if (!same) atomicAdd(&stats[1], 1u);
    else atomicMax(&stats[2], __float_as_uint(d));
}

void moe_route_diff_T(const int32_t *ea, const float *wa, const int32_t *eb, const float *wb, int T, unsigned *stats,
                      hipStream_t s) {
    moe_route_diff_kernel<<<(T + 255) / 256, 256, 0, s>>>(ea, wa, eb, wb, T, stats);
}

}  // namespace qw::gpu
