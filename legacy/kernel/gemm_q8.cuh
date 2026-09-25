#ifndef QW_KERNEL_GEMM_Q8_CUH
#define QW_KERNEL_GEMM_Q8_CUH

#ifndef __AMDGCN__
#error "gemm_q8.cuh is a device header: it must be compiled for an AMDGCN target (HIP), never from host code"
#endif

#include <cstdint>

#include "vec_dot.cuh"

/* ====================================================================
 * W8A8 prefill GEMM for the MoE expert projections, gfx1030 (RDNA2).
 *
 *   C[M][N] = sum_k A[M][K] * B[N][K]
 *
 * A is [M][K] row-major int8 (per-row fp32 scale a_scale[m]),
 * B is [N][K] row-major int8 (per-col fp32 scale b_scale[n]),
 * int32 accumulation, dequantized in the epilogue:
 *
 *   C[m][n] = (float)acc * a_scale[m] * b_scale[n]
 *
 * Real shapes: K=2560, N=640  and  K=640, N=2560; M in {32, 80, 256}
 * (80 is the per-expert token load for a 4096-token prompt). Both Ks
 * are divisible by 4, so every sdot4 group is full: no partial-lane
 * masking, no remainder path is ever taken.
 *
 * gfx1030 has NO WMMA and NO MFMA, no matrix registers, no
 * tensor-core tile instruction. The int8 path is the
 * __builtin_amdgcn_sdot4 builtin (vec_dot.cuh), verified on this ROCm
 * to lower to v_dot4c_i32_i8. warpSize is 32 but is NEVER hardcoded
 * below: this kernel does no warp collectives; its only warp-level
 * property is the natural coalescing of the linear
 * (blockIdx*blockDim+threadIdx) mapping documented per kernel.
 *
 * Two kernels live in this header, selected by the macro
 * GEMM_Q8_USE_BLOCKED (0 = naive, 1 = blocked, default 1) via the
 * host helper qw_gemm_q8_launch(); the test harness flips the macro
 * per pass and diffs the outputs.
 *
 *   qw_gemm_q8_naive_kernel    — 1 output element per thread, scalar
 *                                int K-loop. Correctness reference.
 *   qw_gemm_q8_blocked_kernel  — 4x4 (M x N) register tile per thread,
 *                                16 int32 accumulators in VGPRs,
 *                                sdot4 over K in steps of 4 bytes.
 *
 * Both kernels take an optional int32 accumulator out-buffer
 * (acc_out == NULL -> only the float C is written; acc_out != NULL ->
 * the raw int32 sum is also written, for the harness's bit-identical
 * check against the CPU int32 references in include/qw/quant.h).
 *
 * Tile choice and loads per output element.
 *
 * Blocked tile: TM=4 M-rows x TN=4 N-cols per thread, 16 int32
 * accumulators. The block is 16x16 threads (256, a divisor of the
 * 1024-thread-per-CU limit), covering 64 N x 64 M of C per block.
 *
 *   blockDim.x -> N axis (fastest, threadIdx.x fastest-varying)
 *   blockDim.y -> M axis
 *
 * A-loads (A[m][k], k contiguous): for one (k4, i) step a warp's 32
 * lanes read 8 distinct A rows (32 lanes / 4 n-positions per lane) at
 * the same 4-byte offset, 4 consecutive lanes per row -> 8 fully
 * coalesced 128B segments per warp step.
 *
 * B-loads (B[n][k], k contiguous): the 32 lanes read 32 distinct B
 * rows (4 n-positions x 8 m-rows) at the same 4-byte offset; rows are
 * K bytes apart, so each lane's 4-byte load is 4B-aligned but hits a
 * distinct 64B cache line. No two lanes in a warp share a B row (each
 * thread owns a disjoint tile), so B is a strided-per-line pattern,
 * NOT a coalesced segment.
 *
 * Expected raw loads per output element (K/4 sdot4 groups):
 *   A: (K/4) groups * (TM loads / (TM*TN outputs)) = K/16
 *   B: (K/4) groups * (TN loads / (TM*TN outputs)) = K/16
 *   total = K/8 4-byte loads per output element.
 *
 * Reuse: each A row byte feeds TN=4 outputs; each B row byte TM=4.
 * With L1/L2 credit the effective unique traffic is lower; the A rows
 * are shared by 4 consecutive lanes of a warp (same row, same offset),
 * which is what keeps the A side coalesced. Measured hit rates and
 * the resulting occupancy can only be confirmed on the V620 box.
 *
 * Grid (blocked): gridDim = (ceil(N/64), ceil(M/64)).
 *   M=32,  N=640:  (10, 1)  =  10 blocks
 *   M=80,  N=640:  (10, 2)  =  20 blocks
 *   M=256, N=640:  (10, 4)  =  40 blocks
 *   M=32,  N=2560: (40, 1)  =  40 blocks
 *   M=80,  N=2560: (40, 2)  =  80 blocks
 *   M=256, N=2560: (40, 4)  = 160 blocks
 *
 * The blocked kernel is a plain SIMT register-blocked kernel: NO
 * shared memory, NO __syncthreads. A/B tiles stream straight from
 * global (L2-cached); the only on-chip state is the 16 VGPR
 * accumulators per thread. (An LDS-staging variant with barriers is a
 * larger rewrite whose payoff only a real occupancy/traffic
 * measurement on the V620 box can justify.)
 *
 * Register pressure: 16 accumulator VGPRs + pointers/temps; the
 * static estimate is far under the 256-VGPR limit, but the compiler's
 * final allocation and hence wavefronts-per-CU occupancy are only
 * known from a real run's SASS/occupancy report.
 * ==================================================================== */

#define GEMM_Q8_TM 4 /* M-tile per thread (C rows / A rows) */
#define GEMM_Q8_TN 4 /* N-tile per thread (C cols / B rows) */
#define GEMM_Q8_BX 16 /* block threads on the N axis (fastest) */
#define GEMM_Q8_BY 16 /* block threads on the M axis */

#ifndef GEMM_Q8_USE_BLOCKED
#define GEMM_Q8_USE_BLOCKED 1
#endif

/* ---------------------------------------------------------------- naive */

/* One output element per thread. Grid covers (M, N) with a 32x8
 * block; threads past M/N early-return. K loop is the plain scalar
 * form acc += (int)a * (int)b — no sdot4 on purpose: this is the
 * reference the blocked kernel must match bit-for-bit, and it doubles
 * as the control for "does a trivial loop vectorize at all".
 *
 * Loads per output element: K A-bytes + K B-bytes, all unique to this
 * thread (no intra-warp reuse: each lane owns one (m, n)).
 */
__global__ void qw_gemm_q8_naive_kernel(const int8_t *__restrict__ A,
                                        const int8_t *__restrict__ B,
                                        const float *__restrict__ a_scale,
                                        const float *__restrict__ b_scale,
                                        float *__restrict__ C,
                                        int32_t *__restrict__ acc_out, int M,
                                        int N, int K) {
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N)
    return;
  const int8_t *arow = A + (size_t)m * K;
  const int8_t *brow = B + (size_t)n * K;
  int acc = 0;
  for (int k = 0; k < K; k++)
    acc += (int)arow[k] * (int)brow[k];
  if (acc_out)
    acc_out[(size_t)m * N + n] = acc;
  C[(size_t)m * N + n] = (float)acc * a_scale[m] * b_scale[n];
}

/* --------------------------------------------------------------- blocked */

/* 4x4 register tile per thread; 16 int32 accumulators; sdot4 over K
 * in steps of 4 packed bytes. Memory analysis in the header block.
 *
 * The K loop is K4 = K/4 full groups: 4 A-loads (one int per M-row) +
 * 4 B-loads (one int per N-row) + 16 sdot4 per group. The A int for a
 * given (i, k4) is loaded once and fed to all 4 B lanes of the tile —
 * the classic inner-A reuse of a register-blocked GEMM.
 *
 * The integer fold is exact and matches the CPU int32 references in
 * include/qw/quant.h: sdot4(k4) = sum of 4 consecutive products, and
 * acc = ((sdot4(0) + sdot4(1)) + sdot4(2)) + ... left-to-right, the
 * same value the CPU k-loop produces (integer addition, no overflow
 * for K <= 133144, checked by qw_q8_overflow_ok).
 */
__global__ void qw_gemm_q8_blocked_kernel(const int8_t *__restrict__ A,
                                          const int8_t *__restrict__ B,
                                          const float *__restrict__ a_scale,
                                          const float *__restrict__ b_scale,
                                          float *__restrict__ C,
                                          int32_t *__restrict__ acc_out, int M,
                                          int N, int K) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m0 = (blockIdx.y * blockDim.y + ty) * GEMM_Q8_TM;
  const int n0 = (blockIdx.x * blockDim.x + tx) * GEMM_Q8_TN;

  int acc[GEMM_Q8_TM][GEMM_Q8_TN];
#pragma unroll
  for (int i = 0; i < GEMM_Q8_TM; i++)
#pragma unroll
    for (int j = 0; j < GEMM_Q8_TN; j++) acc[i][j] = 0;

  /* Main loop: full 4-byte groups. K % 4 == 0 for the real shapes;
   * the scalar tail below covers any remainder. Inner loop is the
   * exact shape v_dot4c_i32_i8 lowers from: straight-line induction
   * over k4, no branch, only the accumulates as side effects. */
  const int K4 = K / 4;
  for (int k4 = 0; k4 < K4; k4++) {
    const size_t koff = (size_t)k4 * 4;
    int av[GEMM_Q8_TM];
#pragma unroll
    for (int i = 0; i < GEMM_Q8_TM; i++) {
      int m = m0 + i;
      av[i] = (m < M) ? qw_vload4_i8(A + (size_t)m * K + koff) : 0;
    }
#pragma unroll
    for (int j = 0; j < GEMM_Q8_TN; j++) {
      int n = n0 + j;
      if (n >= N)
        continue;
      int bv = qw_vload4_i8(B + (size_t)n * K + koff);
#pragma unroll
      for (int i = 0; i < GEMM_Q8_TM; i++)
        acc[i][j] = qw_vdot4_q8(av[i], bv, acc[i][j]);
    }
  }

  /* Scalar tail for K % 4 != 0 (dead for the real shapes). */
  if (K4 * 4 < K) {
    for (int k = K4 * 4; k < K; k++) {
#pragma unroll
      for (int i = 0; i < GEMM_Q8_TM; i++) {
        int m = m0 + i;
        if (m >= M)
          continue;
        int av = (int)A[(size_t)m * K + k];
#pragma unroll
        for (int j = 0; j < GEMM_Q8_TN; j++) {
          int n = n0 + j;
          if (n >= N)
            continue;
          acc[i][j] += av * (int)B[(size_t)n * K + k];
        }
      }
    }
  }

  /* Epilogue: raw int32 out (for the bit-identical check), then fused
   * dequant — int32 -> float once, at store time; a_scale per-row (m),
   * b_scale per-col (n). */
#pragma unroll
  for (int i = 0; i < GEMM_Q8_TM; i++) {
    int m = m0 + i;
    if (m >= M)
      continue;
    const float as = a_scale[m];
    float *crow = C + (size_t)m * N;
    int32_t *arow_out = acc_out ? acc_out + (size_t)m * N : NULL;
#pragma unroll
    for (int j = 0; j < GEMM_Q8_TN; j++) {
      int n = n0 + j;
      if (n >= N)
        continue;
      if (arow_out)
        arow_out[n] = acc[i][j];
      crow[n] = (float)acc[i][j] * as * b_scale[n];
    }
  }
}

/* --------------------------------------------------------- host launch */

/* Dispatch on GEMM_Q8_USE_BLOCKED (compile-time macro). Grid shapes
 * differ per kernel, so the helper computes the right one. Both
 * kernels are always compiled into the binary; the macro only selects
 * which one this call site launches. Returns hipGetLastError() of
 * the launch (call hipDeviceSynchronize / hipGetLastError after). */
inline hipError_t qw_gemm_q8_launch(const int8_t *A, const int8_t *B,
                                    const float *a_scale,
                                    const float *b_scale, float *C,
                                    int32_t *acc_out, int M, int N, int K,
                                    hipStream_t stream) {
#if GEMM_Q8_USE_BLOCKED
  dim3 block(GEMM_Q8_BX, GEMM_Q8_BY);
  dim3 grid((N + GEMM_Q8_BX * GEMM_Q8_TN - 1) / (GEMM_Q8_BX * GEMM_Q8_TN),
            (M + GEMM_Q8_BY * GEMM_Q8_TM - 1) / (GEMM_Q8_BY * GEMM_Q8_TM));
  qw_gemm_q8_blocked_kernel<<<grid, block, 0, stream>>>(A, B, a_scale, b_scale,
                                                        C, acc_out, M, N, K);
#else
  dim3 block(32, 8);
  dim3 grid((N + 31) / 32, (M + 7) / 8);
  qw_gemm_q8_naive_kernel<<<grid, block, 0, stream>>>(A, B, a_scale, b_scale,
                                                      C, acc_out, M, N, K);
#endif
  return hipGetLastError();
}

#endif /* QW_KERNEL_GEMM_Q8_CUH */
