// Shared device helpers for gfx1030 (wave32, no matrix units, v_dot2_f32_f16).
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace qw::gpu {

constexpr int WAVE = 32;

// Device-side assertions for debugging (cmake -DQW_DEVICE_CHECKS=ON): a failed
// check traps, and the runtime's fault dump names the kernel.
#ifdef QW_DEVICE_CHECKS
#define QW_DCHECK(cond)                \
    do {                               \
        if (!(cond)) __builtin_trap(); \
    } while (0)
#else
#define QW_DCHECK(cond) \
    do {                \
    } while (0)
#endif

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WAVE);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int o = WAVE / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WAVE));
    return v;
}

// Two fp16 products accumulated into fp32 in one v_dot2_f32_f16.
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float acc) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    h2 x = __builtin_bit_cast(h2, a);
    h2 y = __builtin_bit_cast(h2, b);
    return __builtin_amdgcn_fdot2(x, y, acc, false);
}

// Eight fp16 pairs (16 bytes each side).
__device__ __forceinline__ float dot8(const uint4 &a, const uint4 &b, float acc) {
    acc = dot2(a.x, b.x, acc);
    acc = dot2(a.y, b.y, acc);
    acc = dot2(a.z, b.z, acc);
    return dot2(a.w, b.w, acc);
}

// Two int4 weights in the low nibbles of each 16-bit half (bits 0-3 and
// 16-19) -> two fp16 values (n - 8), exactly. 0x6400 is 1024.0 in fp16, so
// 0x6400 | n reads as 1024 + n.
__device__ __forceinline__ uint32_t nib2_to_h2(uint32_t w) {
    uint32_t v = (w & 0x000F000Fu) | 0x64006400u;
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    h2 h = __builtin_bit_cast(h2, v);
    h -= h2{(_Float16)1032.0f, (_Float16)1032.0f};
    return __builtin_bit_cast(uint32_t, h);
}

// Sum over a block of N waves. All threads get the result. `red` is shared
// scratch of N floats; safe to call back to back.
template <int N>
__device__ __forceinline__ float block_sum(float v, float *red) {
    v = wave_sum(v);
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i) t += red[i];
    return t;
}

// 32 k values (one uint4 of int4 weights) against 32 fp16 activations.
__device__ __forceinline__ float dot_int4x32(const uint4 &w, const uint4 *x, float acc) {
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint4 xv = x[i];
        acc = dot2(nib2_to_h2(ws[i]), xv.x, acc);
        acc = dot2(nib2_to_h2(ws[i] >> 4), xv.y, acc);
        acc = dot2(nib2_to_h2(ws[i] >> 8), xv.z, acc);
        acc = dot2(nib2_to_h2(ws[i] >> 12), xv.w, acc);
    }
    return acc;
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + __expf(-x));
}
__device__ __forceinline__ float silu(float x) {
    return x * sigmoid(x);
}

}  // namespace qw::gpu
