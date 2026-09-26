#include "kernels/common.cuh"
#include "kernels/sampling.hpp"
#include "kernels/types.hpp"

#include <algorithm>

namespace qw::gpu {

namespace {

__device__ __forceinline__ float penalized(float x, int all, int gen, const SampleRow &p) {
    if (all > 0 && p.repetition != 1.f) x = x > 0.f ? x / p.repetition : x * p.repetition;
    if (gen > 0) x -= p.presence + p.frequency * float(gen);
    return x;
}

__global__ __launch_bounds__(256) void penalize_kernel(const float *raw, float *L, const SampleRow *rows,
                                                        int32_t *const *counts, int n, int tok0) {
    const int m = blockIdx.y;
    const SampleRow p = rows[m];
    const float *x = raw + size_t(m) * n;
    float *y = L + size_t(m) * n;
    const bool on = p.presence != 0.f || p.frequency != 0.f || p.repetition != 1.f;
    // generated tokens of the row's run up to itself (not in the slot's counts yet)
    __shared__ int32_t run[MAX_ROWS];
    __shared__ int n_run;
    if (threadIdx.x == 0) {
        n_run = 0;
        if (on)
            for (int j = p.first; j <= m; ++j) run[n_run++] = rows[j].token - tok0;
    }
    __syncthreads();
    const int32_t *c = on ? counts[p.slot] : nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        if (on) {
            int e = 0;
            for (int j = 0; j < n_run; ++j) e += run[j] == i;
            v = penalized(v, c[i] + e, c[n + i] + e, p);
        }
        y[i] = v;
    }
}

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// candidate a before b: higher logit, then lower token id
__device__ __forceinline__ bool before(float la, int ia, float lb, int ib) {
    return la > lb || (la == lb && ia < ib);
}

constexpr int SEL_THREADS = 1024;

__global__ __launch_bounds__(SEL_THREADS) void select_kernel(const float *L_all, const float *raw_all,
                                                              const SampleRow *rows, int n, int tok0, SampleOut *out) {
    const int m = blockIdx.x, t = threadIdx.x;
    const SampleRow p = rows[m];
    const float *L = L_all + size_t(m) * n;
    const float *raw = raw_all + size_t(m) * n;
    constexpr int NW = SEL_THREADS / WAVE;
    __shared__ float red_f[NW];
    __shared__ double red_d[NW];
    __shared__ int red_i[NW];
    __shared__ float s_lmax;
    const int w = t / WAVE, lane = t % WAVE;

    // max
    float mx = -INFINITY;
    for (int i = t; i < n; i += SEL_THREADS) mx = fmaxf(mx, L[i]);
    mx = wave_max(mx);
    if (lane == 0) red_f[w] = mx;
    __syncthreads();
    if (t == 0) {
        float v = red_f[0];
        for (int k = 1; k < NW; ++k) v = fmaxf(v, red_f[k]);
        s_lmax = v;
    }
    __syncthreads();
    const float lmax = s_lmax;

    // normalizer and Gumbel-max draw
    float sum = 0.f;
    double best = -INFINITY;
    int best_i = 0x7fffffff;
    if (p.inv_t > 0.f) {
        for (int i = t; i < n; i += SEL_THREADS) {
            const float l = L[i];
            sum += __expf((l - lmax) * p.inv_t);
            const uint64_t h = mix64(p.key ^ (uint64_t(tok0 + i) * 0x9e3779b97f4a7c15ull));
            const double u = (double(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
            const double sc = double(l) * double(p.inv_t) - log(-log(u));
            if (sc > best) {
                best = sc;
                best_i = i;
            }
        }
    }
    sum = wave_sum(sum);
    for (int o = WAVE / 2; o > 0; o >>= 1) {
        const double b2 = __shfl_xor(best, o, WAVE);
        const int i2 = __shfl_xor(best_i, o, WAVE);
        if (b2 > best || (b2 == best && i2 < best_i)) {
            best = b2;
            best_i = i2;
        }
    }
    __syncthreads();
    if (lane == 0) {
        red_f[w] = sum;
        red_d[w] = best;
        red_i[w] = best_i;
    }
    __syncthreads();
    if (t == 0) {
        float s = 0.f;
        double b = -INFINITY;
        int bi = 0x7fffffff;
        for (int k = 0; k < NW; ++k) {
            s += red_f[k];
            if (red_d[k] > b || (red_d[k] == b && red_i[k] < bi)) {
                b = red_d[k];
                bi = red_i[k];
            }
        }
        SampleOut &o = out[m];
        o.lmax = lmax;
        o.sumexp = s;
        o.g_score = float(b);
        o.g_idx = bi == 0x7fffffff ? -1 : tok0 + bi;
        o.g_raw = bi == 0x7fffffff ? 0.f : raw[bi];
    }

    // top SAMPLE_CAND by L: radix select of the threshold key over 4 digits of 8 bits
    constexpr int C = SAMPLE_CAND;
    __shared__ int hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_rem;
    if (t == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_rem = min(C, n);
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256) hist[t] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int i = t; i < n; i += SEL_THREADS) {
            const uint32_t k = order_key(L[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (t == 0) {
            int cum = 0, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= s_rem) break;
                cum += hist[d];
            }
            s_rem -= cum;
            s_prefix = prefix | (uint32_t(d) << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    // gather: keys above the threshold, then as many equal to it as fit
    __shared__ float cl[C], cr[C];
    __shared__ int ci[C];
    __shared__ int n_gt, n_eq;
    const uint32_t T = s_prefix;
    const int take = min(C, n), rem = s_rem;
    if (t == 0) {
        n_gt = 0;
        n_eq = 0;
    }
    if (t < C) {
        cl[t] = -INFINITY;
        ci[t] = 0x7fffffff;
        cr[t] = 0.f;
    }
    __syncthreads();
    for (int i = t; i < n; i += SEL_THREADS) {
        const float l = L[i];
        const uint32_t k = order_key(l);
        int pos = -1;
        if (k > T)
            pos = atomicAdd(&n_gt, 1);
        else if (k == T) {
            const int e = atomicAdd(&n_eq, 1);
            if (e < rem) pos = take - rem + e;
        }
        if (pos >= 0 && pos < C) {
            cl[pos] = l;
            cr[pos] = raw[i];
            ci[pos] = tok0 + i;
        }
    }
    __syncthreads();
    // bitonic sort of the C candidates into `before` order
    for (int k = 2; k <= C; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (t < C) {
                const int x = t ^ j;
                if (x > t) {
                    const bool asc = (t & k) == 0;
                    const bool sw = asc ? before(cl[x], ci[x], cl[t], ci[t]) : before(cl[t], ci[t], cl[x], ci[x]);
                    if (sw) {
                        const float a = cl[t], b = cr[t];
                        const int c = ci[t];
                        cl[t] = cl[x];
                        cr[t] = cr[x];
                        ci[t] = ci[x];
                        cl[x] = a;
                        cr[x] = b;
                        ci[x] = c;
                    }
                }
            }
            __syncthreads();
        }
    SampleOut &o = out[m];
    if (t < C) {
        o.cand_l[t] = cl[t];
        o.cand_raw[t] = cr[t];
        o.cand_idx[t] = ci[t];
    }
    if (t == 0) o.n = take;
}

__global__ void penalty_add_kernel(int32_t *counts, const int32_t *tokens, const uint8_t *gen, int n_tokens, int n,
                                   int tok0) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_tokens; i += gridDim.x * blockDim.x) {
        const int k = tokens[i] - tok0;
        if (k < 0 || k >= n) continue;
        atomicAdd(&counts[k], 1);
        if (gen[i]) atomicAdd(&counts[n + k], 1);
    }
}

}  // namespace

void sample_penalize(const float *raw, float *L, const SampleRow *rows, int32_t *const *counts, int n, int tok0, int M,
                     hipStream_t s) {
    penalize_kernel<<<dim3(16, M), 256, 0, s>>>(raw, L, rows, counts, n, tok0);
}

void sample_select(const float *L, const float *raw, const SampleRow *rows, int n, int tok0, int M, SampleOut *out,
                   hipStream_t s) {
    select_kernel<<<M, SEL_THREADS, 0, s>>>(L, raw, rows, n, tok0, out);
}

void penalty_add(int32_t *counts, const int32_t *tokens, const uint8_t *gen, int n_tokens, int n, int tok0,
                 hipStream_t s) {
    if (n_tokens <= 0) return;
    penalty_add_kernel<<<std::min(1024, (n_tokens + 255) / 256), 256, 0, s>>>(counts, tokens, gen, n_tokens, n, tok0);
}

}  // namespace qw::gpu
