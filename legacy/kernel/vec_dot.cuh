#ifndef QW_KERNEL_VEC_DOT_CUH
#define QW_KERNEL_VEC_DOT_CUH

#ifndef __AMDGCN__
#error "vec_dot.cuh is a device header: it must be compiled for an AMDGCN target (HIP), never from host code"
#endif

#include <cstdint>

// __builtin_amdgcn_sdot4: dot product of 4 signed bytes packed one per int32 lane,
// accumulated into acc. On gfx1030 the emitted mnemonic is v_dot4c_i32_i8.
// clamp must stay false: the non-clamping form is the fast path on gfx1030.
// This builtin spelling is gfx10-specific: on gfx11+ the ISA renamed it to
// v_dot4_i32_iu8 (host builtin __builtin_amdgcn_sudot4), which do NOT exist on gfx1030.
static __device__ __forceinline__ int qw_vdot4_q8(int a, int b, int acc) {
  return __builtin_amdgcn_sdot4(a, b, acc, false);
}

// Load 4 int8 elements as one int. Byte order matches the sdot4 lane order
// (little-endian int8s 0..3 map to lanes 0..3). Caller MUST guarantee 4-byte
// alignment of p; misaligned access is UB.
static __device__ __forceinline__ int qw_vload4_i8(const int8_t *p) {
  return *(const int *)p;
}

// Dot product of n4 packed 4-byte groups, returned as one int32.
static __device__ __forceinline__ int qw_vdot4_q8_row(const int8_t *a, const int8_t *b, int n4) {
  int acc = 0;
  for (int i = 0; i < n4; i++)
    acc = qw_vdot4_q8(qw_vload4_i8(a + 4 * i), qw_vload4_i8(b + 4 * i), acc);
  return acc;
}

#endif
