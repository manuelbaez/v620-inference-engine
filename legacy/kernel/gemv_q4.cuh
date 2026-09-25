#ifndef QW_KERNEL_GEMV_Q4_CUH
#define QW_KERNEL_GEMV_Q4_CUH

#ifndef __AMDGCN__
#error "gemv_q4.cuh is a device header: it must be compiled for an AMDGCN target (HIP), never from host code"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

/* ============================================================================
 * Decode-path skinny GEMV: one token, weight-stationary.
 *
 *     y[n] = sum_{k=0}^{K-1} A[k] * W[n][k],  W stored row-major as GGUF q4_0
 *
 * Weight layout (pinned by tests/test_dequant.c, test 5):
 *   - one q4_0 block = 32 elements in 32 bytes: { fp16 d (uint16, little-
 *     endian); uint8 qh[16] }
 *   - element i lives in byte i/2: even i = LOW nibble, odd i = HIGH nibble
 *   - dequantized value = d * (nib - 8)
 *
 * Target: gfx1030 (RDNA2). No rocWMMA, no MFMA, no fp16-acc/fp8/fp4 hardware.
 * fp16 and int dot instructions DO exist (v_dot2_f32_f16, v_dot4c_i32_i8,
 * v_dot8_u32_u4), but this kernel deliberately does NOT use them: with a
 * random activation A, dequantizing in registers and accumulating in fp32
 * gives the same memory-bound performance (the kernel streams W once at ~12
 * B/elt effective and is L2/BW-limited, not ALU-limited), while keeping the
 * dequant bit-identical to the CPU oracle in src/core/dequant.c and avoiding
 * an int4->fp16 round-trip per product. The int-dot path (pack d*(nib-8) into
 * int8, sdot4) is the later optimization once A itself is int8-quantized.
 * ========================================================================== */

/* Q4_0 on-disk block: 32 elts, 32 bytes. The payload (the 16 nibble bytes)
 * starts at offset 2 and is 16 bytes, i.e. 4x int32 aligned when the block
 * is 32-byte aligned. hipMalloc returns 256-byte-aligned memory, so every
 * block payload below is 16-byte aligned. */
/* The on-disk block is 32 bytes, but the meaningful payload is 18 bytes
 * (2-byte scale + 16 payload bytes); the trailing 14 bytes are GGUF padding.
 * The struct is packed to exactly the 18-byte payload (a uint16 + 16 bytes is
 * 16-byte aligned with no internal padding). Pointers to a block therefore
 * address the scale; the nibble payload is at +2 (16-byte aligned when the
 * block is 32-byte aligned). */
struct QwQ4_0Block {
  uint16_t d;       /* fp16 scale, little-endian */
  uint8_t  qh[16];  /* packed nibbles: byte i holds elts 2i (lo), 2i+1 (hi) */
};
static_assert(sizeof(QwQ4_0Block) == 18, "q4_0 payload is 18 bytes (2 scale + 16 nibbles)");

/* Byte stride of one q4_0 block on disk / in W. */
static constexpr int QW_Q4_0_BLK_STRIDE = 32;

/* Dequantize one q4_0 block into 32 fp16 values, entirely in registers.
 * NO shared memory, NO extra global round-trip: one 16-byte aligned read of
 * the nibble payload, one 2-byte read of the scale, and everything else is
 * integer ALU in registers. Nibble convention matches qw_dequant_q4_0_row
 * exactly: byte b, low nibble -> element 2b, high nibble -> element 2b+1,
 * value = d * (nib - 8). */
static __device__ __forceinline__ void qw_dequant_q4_0_block(const QwQ4_0Block *blk,
                                                             uint16_t *w16) {
  const float d = __half2float(*reinterpret_cast<const __half *>(&blk->d));
  const int4 payload = *reinterpret_cast<const int4 *>(blk->qh);
  const uint32_t u[4] = {
      static_cast<uint32_t>(payload.x), static_cast<uint32_t>(payload.y),
      static_cast<uint32_t>(payload.z), static_cast<uint32_t>(payload.w)};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const uint32_t v = u[j];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint8_t q = static_cast<uint8_t>((v >> (8 * k)) & 0xFFu);
      const int e = 8 * j + 2 * k; /* element index within the block */
      w16[e]     = __float2half(d * (float)((q & 0x0Fu) - 8));
      w16[e + 1] = __float2half(d * (float)((q >> 4) - 8));
    }
  }
}

/* Fused dequant + dot: accumulate one q4_0 block into a float accumulator.
 * Dequantizes into fp16 registers (no memory traffic, no shared memory) and
 * folds each dequantized weight into acc in fp32. 32 products/block. */
static __device__ __forceinline__ float qw_dot_q4_0_block(const QwQ4_0Block *blk,
                                                          const __half *a16,
                                                          float acc) {
  uint16_t w16[32];
  qw_dequant_q4_0_block(blk, w16);
  // w16 is an array of uint16_t: the element stride is 2 bytes, so w[e] is a
  // 2-byte-aligned half read (the register-backed dequantized weights).
  const __half *w = reinterpret_cast<const __half *>(w16);
#pragma unroll
  for (int e = 0; e < 32; e += 2) {
    acc = fmaf(__half2float(a16[e]), __half2float(w[e]), acc);
    acc += __half2float(a16[e + 1]) * __half2float(w[e + 1]);
  }
  return acc;
}

/* ============================================================================
 * Kernel. Grid-stride over output rows; one output element per warp.
 *
 * Why one output per warp (small N, the decode case): each warp strides K/32
 * q4_0 blocks of 32 bytes, so its 32 lanes touch 32 consecutive blocks.
 * Consecutive lanes read CONSECUTIVE 16-byte block payloads (and the
 * consecutive 2-byte scales), i.e. the warp's accesses are 16 bytes apart
 * monotonically, so the 32-lane warp issues fully coalesced 512-byte
 * transactions (32 lanes x 16 B) per iteration. With one output per thread
 * the K/32 blocks of one output would be spread across 32 warps instead,
 * each lane group striding 1024 bytes through W — scattered, uncoalesced
 * accesses and poor DRAM burst utilization.
 *
 * Switch to one output per thread when N is LARGE (prefill / big GEMMs,
 * N >> K/32): then the whole W matrix streams through memory once and
 * occupancy/parallelism, not per-row coalescing, dominates, and a thread can
 * amortize the per-row activation broadcast over many k iterations.
 * ========================================================================== */
__global__ void gemv_q4_0_kernel(const uint8_t *__restrict__ W,
                                            const __half *__restrict__ A,
                                            float *__restrict__ y, int N, int Kblk) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= N)
    return;

  /* Activation: fp16 in device memory (converted host-side from the fp32
   * activation, which has no representable error for in-range activations).
   * The whole vector is shared by the warp; each lane reads its 32-element
   * chunk once per block iteration. */
  const __half *a_row = A + warp * Kblk * 32;

  /* Coalescing: lane l of this warp reads block (it * 32 + l) of row `warp`.
   * Block b occupies bytes [32b, 32b+32): scale at bytes 2..4, payload at
   * 32b+2 .. 32b+18. So across lanes the payload reads are 16-byte-aligned
   * and consecutive -> 32 x 16 B = 512 B coalesced per iteration. W is a raw
   * byte pointer; each block is QW_Q4_0_BLK_STRIDE (32) bytes apart. */
  const QwQ4_0Block *Wb = reinterpret_cast<const QwQ4_0Block *>(W);
  float acc = 0.0f;
  for (int it = 0; it < (Kblk + 31) / 32; it++) {
    const int b = it * 32 + lane;
    if (b < Kblk) {
      const __half *a16 = a_row + b * 32;
      acc = qw_dot_q4_0_block(Wb + (warp * Kblk + b) * QW_Q4_0_BLK_STRIDE, a16, acc);
    }
  }

  /* Warp reduce: butterfly over the device warpSize (never hardcode 32). */
  const int WSIZE = static_cast<int>(warpSize);
#pragma unroll
  for (int off = WSIZE / 2; off > 0; off /= 2)
    acc += __shfl_xor_sync(0xFFFFFFFFull, acc, off);

  if (lane == 0)
    y[warp] = acc;
}

#endif
