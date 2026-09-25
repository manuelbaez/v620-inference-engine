/* test_gemv_q4.hip — decode-path q4_0 skinny GEMV test (one token, weight-stationary).
 *
 * y[n] = sum_k A[k] * W[n][k], W in GGUF q4_0, A a single fp32 vector.
 *
 * Deterministic pseudo-random data (splitmix64, seeded per shape) on the 4
 * real model shapes, compared against a readable CPU reference that
 * dequantizes to float then dots in double (the same convention tests/
 * test_dequant.c pins: even i = low nibble, odd i = high nibble, value =
 * d * (nib - 8)).
 *
 * Tolerance: max relative error <= 4e-4 (below, with headroom, the 1e-3
 * worst-case bound). Reason: every term is formed in fp32 (fp32 product,
 * fp32 running sum; the fp16 dequantized values widen exactly, so the only
 * error is fp32 rounding of each product and of the sum), the double
 * reference rounds each term identically, and the sum of K random-sign
 * terms gives rel_err ~ sqrt(K) * 2^-24. Worst shape K = 12288 -> ~3e-4.
 * The bound also covers the A fp32->fp16 conversion (A values |a|<4, exact
 * for |a|<=2^11) and the 5-level warp-reduce (negligible vs the K-term sum).
 *
 * Safe on a laptop: if hipMalloc fails (no GPU / no device) prints
 * "SKIP: no GPU" and returns 0. Every HIP return code is checked.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gemv_q4.cuh"

/* Deterministic 64-bit PRNG (splitmix64), independent of host stdlib. */
static uint64_t rng_state;
static uint64_t splitmix64(void) {
  uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static void f16_bits_to_f32(uint16_t h, float *out) {
  __half_raw hr;
  hr.x = h;
  __half hh = *reinterpret_cast<const __half *>(&hr);
  *out = __half2float(hh);
}

static uint16_t f32_to_f16_bits(float f) {
  __half hh = __float2half_rn(f);
  __half_raw hr = *reinterpret_cast<const __half_raw *>(&hh);
  return hr.x;
}

struct Shape {
  const char *name;
  int K;  /* = Kblk * 32 */
  int N;
};

static double ref_dequant_dot(const uint8_t *W, int n, int K, const float *A) {
  /* W[n][k]: row n starts at W + n*K/32*32 bytes. */
  const size_t row_bytes = (size_t)(K / 32) * 32;
  const uint8_t *row = W + (size_t)n * row_bytes;
  double s = 0.0;
  for (int b = 0; b < K / 32; b++) {
    const uint8_t *blk = row + (size_t)b * 32;
    uint16_t dbits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
    float d;
    f16_bits_to_f32(dbits, &d);
    for (int i = 0; i < 32; i++) {
      uint8_t q = blk[2 + i / 2];
      int nib = (i % 2 == 0) ? (q & 0x0F) : (q >> 4);
      s += (double)d * (double)(nib - 8) * (double)A[b * 32 + i];
    }
  }
  return s;
}

int main(void) {
  const Shape shapes[] = {
      {"K=2560,N=640", 2560, 640},
      {"K=2560,N=2560", 2560, 2560},
      {"K=2560,N=12288", 2560, 12288},
      {"K=640,N=2560", 640, 2560},
  };
  const int NSHAPES = 4;
  const double TOL = 4e-4;

  hipError_t e;
  bool any_fail = false;

  for (int s = 0; s < NSHAPES; s++) {
    const Shape sh = shapes[s];
    const int K = sh.K, N = sh.N;
    const int Kblk = K / 32;
    const size_t w_bytes = (size_t)N * Kblk * 32;

    /* --- deterministic data --- */
    rng_state = 0x5eedULL + (uint64_t)s * 0x9e3779b9ULL;
    /* A in (-2,2): exactly representable in fp16 (|a|<=2^11), so the fp32
     * -> fp16 conversion of the activation on device is lossless. */
    float *hA = (float *)malloc((size_t)K * sizeof(float));
    uint8_t *hW = (uint8_t *)malloc(w_bytes);
    if (!hA || !hW) {
      printf("FAIL: host malloc\n");
      return 1;
    }
    for (int i = 0; i < K; i++)
      hA[i] = (float)((double)(splitmix64() >> 11) / 4503599627370496.0) * 4.0 - 2.0;
    for (size_t i = 0; i < N; i++) {
      uint8_t *row = hW + i * Kblk * 32;
      for (int b = 0; b < Kblk; b++) {
        uint8_t *blk = row + (size_t)b * 32;
        float d = (float)((double)(splitmix64() >> 11) / 4503599627370496.0) * 4.0 - 2.0;
        uint16_t dbits = f32_to_f16_bits(d);
        blk[0] = (uint8_t)(dbits & 0xFF);
        blk[1] = (uint8_t)(dbits >> 8);
        for (int j = 0; j < 16; j++)
          blk[2 + j] = (uint8_t)(splitmix64() & 0xFF);
      }
    }

    /* --- device buffers --- */
    uint8_t *dW = NULL;
    __half *dA = NULL;
    float *dY = NULL;
    e = hipMalloc((void **)&dW, w_bytes);
    if (e != hipSuccess) {
      printf("SKIP: no GPU\n");
      return 0;
    }
    e = hipMalloc((void **)&dA, (size_t)N * Kblk * 32 * sizeof(__half));
    if (e != hipSuccess) {
      printf("SKIP: no GPU\n");
      return 0;
    }
    e = hipMalloc((void **)&dY, (size_t)N * sizeof(float));
    if (e != hipSuccess) {
      printf("SKIP: no GPU\n");
      return 0;
    }
    /* Convert A fp32 -> fp16 (host-side; __float2half_rn is __HOST_DEVICE__). */
    __half *hA16 = (__half *)malloc((size_t)N * Kblk * 32 * sizeof(__half));
    if (!hA16) {
      printf("FAIL: host malloc A16\n");
      return 1;
    }
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++)
        hA16[n * Kblk * 32 + k] = __float2half_rn(hA[k]);

    e = hipMemcpy(dW, hW, w_bytes, hipMemcpyHostToDevice);
    if (e != hipSuccess) {
      printf("FAIL: hipMemcpy W H2D: %s\n", hipGetErrorString(e));
      return 1;
    }
    e = hipMemcpy(dA, hA16, (size_t)N * Kblk * 32 * sizeof(__half), hipMemcpyHostToDevice);
    if (e != hipSuccess) {
      printf("FAIL: hipMemcpy A H2D: %s\n", hipGetErrorString(e));
      return 1;
    }

    /* --- launch: one warp per output row, grid-stride over rows --- */
    const int threads = 256; /* 8 warps/block */
    int blocks = (N + threads / 32 - 1) / (threads / 32);
    if (blocks < 1)
      blocks = 1;
    gemv_q4_0_kernel<<<blocks, threads>>>(dW, dA, dY, N, Kblk);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("FAIL: hipDeviceSynchronize: %s\n", hipGetErrorString(e));
      return 1;
    }

    float *hY = (float *)malloc((size_t)N * sizeof(float));
    if (!hY) {
      printf("FAIL: host malloc Y\n");
      return 1;
    }
    e = hipMemcpy(hY, dY, (size_t)N * sizeof(float), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
      printf("FAIL: hipMemcpy Y D2H: %s\n", hipGetErrorString(e));
      return 1;
    }

    /* --- compare against double reference --- */
    double max_rel = 0.0;
    int fail_n = -1;
    for (int n = 0; n < N; n++) {
      double ref = ref_dequant_dot(hW, n, K, hA);
      double got = (double)hY[n];
      double rel = fabs(got - ref) / fmax(fabs(ref), 1.0e-3);
      if (rel > max_rel)
        max_rel = rel;
      if (rel > TOL && fail_n < 0)
        fail_n = n;
    }

    bool ok = max_rel <= TOL;
    if (ok)
      printf("PASS %-14s K=%-5d N=%-6d max_rel=%.3e\n", sh.name, K, N, max_rel);
    else {
      printf("FAIL %-14s K=%-5d N=%-6d max_rel=%.3e (tol %.1e) first@n=%d\n",
             sh.name, K, N, max_rel, TOL, fail_n);
      any_fail = true;
    }

    /* cleanup per shape (check every HIP return code) */
    if (hipFree(dW) != hipSuccess || hipFree(dA) != hipSuccess ||
        hipFree(dY) != hipSuccess) {
      printf("WARN: hipFree: %s\n", hipGetErrorString(hipGetLastError()));
    }
    free(hA);
    free(hW);
    free(hA16);
    free(hY);
  }

  if (any_fail) {
    printf("RESULT: FAIL\n");
    return 1;
  }
  printf("RESULT: PASS (tol %.0e, fp32-accum, double reference)\n", TOL);
  return 0;
}
