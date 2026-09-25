#include "kernels/ple.hpp"

#include "kernels/model_common.cuh"

namespace qw::gpu {

// ---- prefill
__global__ __launch_bounds__(256) void ple_stats_T_kernel(const __half *kv, const float *X, const float *nk,
                                                          const float *nq, float *send) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    const __half *key = kv + size_t(t) * (XW + SH), *val = key + XW;
    const float *x = X + size_t(t) * XW;
    float *o = send + t * 16;
    for (int s = 0; s < HC; ++s) {
        float sk = 0.f, sx = 0.f, d = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            const float k = __half2float(key[i]), xv = x[i];
            sk += k * k;
            sx += xv * xv;
            d += k * nk[i] * xv * nq[i];
        }
        sk = block_sum<8>(sk, red);
        sx = block_sum<8>(sx, red);
        d = block_sum<8>(d, red);
        if (threadIdx.x == 0) {
            o[s] = sk;
            o[4 + s] = sx;
            o[8 + s] = d;
        }
    }
    float sv = 0.f;
    for (int j = threadIdx.x; j < SH; j += blockDim.x) {
        const float v = __half2float(val[j]);
        sv += v * v;
    }
    sv = block_sum<8>(sv, red);
    if (threadIdx.x == 0) {
        o[12] = sv;
        o[13] = o[14] = o[15] = 0.f;
    }
}

void ple_stats_T(const uint16_t *kv, const float *X, const float *nk, const float *nq, int T, float *send,
                 hipStream_t s) {
    ple_stats_T_kernel<<<T, 256, 0, s>>>(reinterpret_cast<const __half *>(kv), X, nk, nq, send);
}

__device__ __forceinline__ float ple_gate(const float *red, int s) {
    const float rk = rsqrtf(red[s] / H + EPS), rq = rsqrtf(red[4 + s] / H + EPS);
    const float gs = red[8 + s] * rk * rq * rsqrtf(float(H));
    const float mag = sqrtf(fmaxf(fabsf(gs), 1e-6f));
    return sigmoid(gs > 0.f ? mag : (gs < 0.f ? -mag : 0.f));
}

__global__ void ple_nc_T_kernel(const __half *kv, const float *red, const float *nc, int T, float *ncbuf) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW), s = c / SH, j = c % SH;
    const float *r = red + t * 16;
    const float gate = ple_gate(r, s);
    const float rc = rsqrtf(gate * gate * r[12] / H + EPS);
    ncbuf[i] = gate * __half2float(kv[size_t(t) * (XW + SH) + XW + j]) * rc * nc[c];
}

__global__ void ple_conv_T_kernel(const __half *kv, const float *red, const float *conv_w, const float *ring,
                                  const float *ncbuf, int64_t start, int T, float *X) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * XW) return;
    const int t = int(i / XW), c = int(i % XW), s = c / SH, j = c % SH;
    const int64_t pos = start + t;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < PLE_CONV; ++k) {
        const int64_t q = pos - (PLE_CONV - 1 - k) * PLE_DILATION;
        float v;
        if (q >= start)
            v = ncbuf[size_t(q - start) * XW + c];
        else if (q >= 0)
            v = ring[int(q % PLE_RING) * XW + c];
        else
            v = 0.f;
        acc += conv_w[c * PLE_CONV + k] * v;
    }
    const float gv = ple_gate(red + t * 16, s) * __half2float(kv[size_t(t) * (XW + SH) + XW + j]);
    X[i] += gv + silu(acc);
}

__global__ void ple_ring_T_kernel(const float *ncbuf, float *ring, int64_t start, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= XW) return;
    const int hist = (PLE_CONV - 1) * PLE_DILATION;
    for (int64_t q = std::max<int64_t>(start, start + T - hist); q < start + T; ++q)
        ring[int(q % PLE_RING) * XW + c] = ncbuf[size_t(q - start) * XW + c];
}

// The whole ring as it stands at position c (start < c <= start + T).
__global__ void ple_ring_capture_kernel(const float *ncbuf, const float *ring, int64_t start, int64_t c, float *dst) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= XW) return;
    for (int64_t q = c - PLE_RING; q < c; ++q) {
        const int cell = int((q + PLE_RING) % PLE_RING);
        float v = 0.f;
        if (q >= start)
            v = ncbuf[size_t(q - start) * XW + ch];
        else if (q >= 0)
            v = ring[cell * XW + ch];
        dst[cell * XW + ch] = v;
    }
}

void ple_apply_T(const uint16_t *kv, const float *red, const float *nc, const float *conv_w, float *ring, int64_t start,
                 int T, float *ncbuf, float *X, hipStream_t s, const Captures &cap) {
    const auto *k = reinterpret_cast<const __half *>(kv);
    const unsigned g = unsigned((size_t(T) * XW + 255) / 256);
    ple_nc_T_kernel<<<g, 256, 0, s>>>(k, red, nc, T, ncbuf);
    ple_conv_T_kernel<<<g, 256, 0, s>>>(k, red, conv_w, ring, ncbuf, start, T, X);
    for (int i = 0; i < cap.n; ++i)
        if (cap.pos[i] > start && cap.pos[i] <= start + T)
            ple_ring_capture_kernel<<<XW / 256, 256, 0, s>>>(ncbuf, ring, start, cap.pos[i], cap.dst[i]);
    ple_ring_T_kernel<<<XW / 256, 256, 0, s>>>(ncbuf, ring, start, T);
}

// ---- batched decode rows
__global__ __launch_bounds__(256) void ple_stats_B_kernel(const float *kv_all, const float *X_all, const float *nk,
                                                          const float *nq, float *send_all) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    const float *key = kv_all + size_t(r) * (XW + SH), *val = key + XW;
    const float *x = X_all + size_t(r) * XW;
    float *o = send_all + size_t(r) * 16;
    for (int s = 0; s < HC; ++s) {
        float sk = 0.f, sx = 0.f, d = 0.f;
        for (int j = threadIdx.x; j < SH; j += blockDim.x) {
            const int i = s * SH + j;
            const float k = key[i], xv = x[i];
            sk += k * k;
            sx += xv * xv;
            d += k * nk[i] * xv * nq[i];
        }
        sk = block_sum<8>(sk, red);
        sx = block_sum<8>(sx, red);
        d = block_sum<8>(d, red);
        if (threadIdx.x == 0) {
            o[s] = sk;
            o[4 + s] = sx;
            o[8 + s] = d;
        }
    }
    float sv = 0.f;
    for (int j = threadIdx.x; j < SH; j += blockDim.x) sv += val[j] * val[j];
    sv = block_sum<8>(sv, red);
    if (threadIdx.x == 0) {
        o[12] = sv;
        o[13] = o[14] = o[15] = 0.f;
    }
}

void ple_stats_B(const float *kv, const float *X, const float *nk, const float *nq, int M, float *send, hipStream_t s) {
    ple_stats_B_kernel<<<M, 256, 0, s>>>(kv, X, nk, nq, send);
}

__device__ __forceinline__ float ple_gate_B(const float *red, int s) {
    const float rk = rsqrtf(red[s] / H + EPS), rq = rsqrtf(red[4 + s] / H + EPS);
    const float gs = red[8 + s] * rk * rq * rsqrtf(float(H));
    const float mag = sqrtf(fmaxf(fabsf(gs), 1e-6f));
    return sigmoid(gs > 0.f ? mag : (gs < 0.f ? -mag : 0.f));
}

__global__ void ple_nc_B_kernel(const float *kv, const float *red, const float *nc, float *ncbuf) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= XW) return;
    const float *rr = red + size_t(r) * 16;
    const float gate = ple_gate_B(rr, i / SH);
    const float rc = rsqrtf(gate * gate * rr[12] / H + EPS);
    ncbuf[size_t(r) * XW + i] = gate * kv[size_t(r) * (XW + SH) + XW + i % SH] * rc * nc[i];
}

__global__ void ple_conv_B_kernel(const float *kv, const float *red, const float *conv_w, const SlotPtrs *tab,
                                  Rows rows, const float *ncbuf, float *X) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= XW) return;
    const int64_t pos = rows.pos[r], first_pos = rows.pos[rows.run_first[r]];
    const float *ring = tab[rows.slot[r]].ple;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < PLE_CONV; ++k) {
        const int64_t q = pos - (PLE_CONV - 1 - k) * PLE_DILATION;
        float v;
        if (q >= first_pos)
            v = ncbuf[size_t(r - (pos - q)) * XW + i];
        else if (q >= 0)
            v = ring[int(q % PLE_RING) * XW + i];
        else
            v = 0.f;
        acc += conv_w[i * PLE_CONV + k] * v;
    }
    const float gv = ple_gate_B(red + size_t(r) * 16, i / SH) * kv[size_t(r) * (XW + SH) + XW + i % SH];
    X[size_t(r) * XW + i] += gv + silu(acc);
}

__global__ void ple_ring_B_kernel(const float *ncbuf, const SlotPtrs *tab, Rows rows) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= XW) return;
    tab[rows.slot[r]].ple[int(rows.pos[r] % PLE_RING) * XW + i] = ncbuf[size_t(r) * XW + i];
}

void ple_apply_B(const float *kv, const float *red, const float *nc, const float *conv_w, const SlotPtrs *tab,
                 Rows rows, float *ncbuf, float *X, hipStream_t s) {
    const dim3 grid(XW / 256, rows.M);
    ple_nc_B_kernel<<<grid, 256, 0, s>>>(kv, red, nc, ncbuf);
    ple_conv_B_kernel<<<grid, 256, 0, s>>>(kv, red, conv_w, tab, rows, ncbuf, X);
    ple_ring_B_kernel<<<grid, 256, 0, s>>>(ncbuf, tab, rows);
}

}  // namespace qw::gpu
