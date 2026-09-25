#include "kernels/moe.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

__global__ __launch_bounds__(512) void moe_route_T_kernel(const float *logits, int first, int32_t *tok_e, float *tok_w,
                                                          int32_t *counts) {
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
    }
}

void moe_route_T(const float *logits, int T, int first, int32_t *tok_e, float *tok_w, int32_t *counts, hipStream_t s) {
    moe_route_T_kernel<<<T, N_EXPERTS, 0, s>>>(logits, first, tok_e, tok_w, counts);
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
__global__ void moe_combine_B_kernel(const __half *yp, const int32_t *tok_slot, const float *tok_w, const float *shared,
                                     float *out) {
    const int r = blockIdx.x;
    for (int n = threadIdx.x; n < H; n += blockDim.x) {
        float acc = shared[size_t(r) * H + n];
        for (int k = 0; k < TOP_K; ++k) {
            const int slot = tok_slot[r * TOP_K + k];
            if (slot >= 0) acc += tok_w[r * TOP_K + k] * __half2float(yp[size_t(slot) * H + n]);
        }
        out[size_t(r) * H + n] = acc;
    }
}

void moe_combine_B(const uint16_t *yp, const int32_t *tok_slot, const float *tok_w, const float *shared, int M,
                   float *out, hipStream_t s) {
    moe_combine_B_kernel<<<M, 256, 0, s>>>(reinterpret_cast<const __half *>(yp), tok_slot, tok_w, shared, out);
}

}  // namespace qw::gpu
