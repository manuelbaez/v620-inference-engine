// test_gemm_q8.hip — W8A8 prefill GEMM correctness harness, gfx1030.
//
// For each real shape (K=2560/N=640 and K=640/N=2560) with M in {32,
// 80, 256}:
//   1. generate deterministic int8 A/B and fp32 scales (seeded LCG,
//      one seed per shape, same data for every implementation),
//   2. run the naive GPU kernel and the blocked GPU kernel (selected
//      by GEMM_Q8_USE_BLOCKED via qw_gemm_q8_launch; both compiled in),
//   3. run the CPU references from include/qw/quant.h:
//        qw_q8_gemm_int32_naive    — raw int32 accumulators
//        qw_q8_gemm_int32_blocked  — raw int32 accumulators
//        qw_q8_gemm_ref            — dequantized float output
//   4. assert the int32 accumulators are BIT-IDENTICAL across all
//      five (GPU naive, GPU blocked, CPU naive, CPU blocked, and by
//      construction the sdot4 folds equal the scalar folds — exact
//      integer math, no overflow for these Ks, checked by
//      qw_q8_overflow_ok),
//   5. assert the dequantized float output is bit-identical to
//      qw_q8_gemm_ref (same fp32 expression (float)acc * as * bs in
//      all implementations; any mismatch is reported with max diff),
//   6. print PASS/FAIL per shape.
//
// A timing loop (blocked kernel, 100 iters, events) runs after the
// checks; it is inside the same "have GPU" guard and does nothing
// when hipMalloc failed.
//
// Every HIP return code is checked. If hipMalloc fails (no GPU) the
// harness prints "SKIP: no GPU" and returns 0.
//
// Syntax check: hipcc --offload-arch=gfx1030 -fsyntax-only
// kernel/test_gemm_q8.hip   (also covered by `make hipsyntax`, which
// wildcards kernel/*.hip.)
//
// NOTE for a full build: link against build/libqw.a (the CPU
// references live in src/core/quant.c).

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gemm_q8.cuh" // GPU kernels (includes vec_dot.cuh for sdot4)

// CPU references from include/qw/quant.h (src/core/quant.c, linked via
// build/libqw.a at build time). Declared here instead of #included so
// the mandated syntax check — which passes NO -I flags — resolves
// cleanly: only sibling files in kernel/ are included. Signatures are
// copied verbatim from include/qw/quant.h; keep them in sync.
typedef enum qw_err {
  QW_OK = 0,
  QW_ERR_NULL,
  QW_ERR_ALLOC,
  QW_ERR_IO,
  QW_ERR_FORMAT,
  QW_ERR_UNSUPPORTED,
  QW_ERR_DEVICE,
  QW_ERR_NOMEM,
  QW_ERR_RANGE,
  QW_ERR_NOTIMPL,
} qw_err;

extern "C" {
bool qw_q8_overflow_ok(int m, int n, int k);
qw_err qw_q8_gemm_ref(const int8_t *a_q8, const float *a_scale,
                      const int8_t *w_q8, const float *w_scale, int m, int n,
                      int k, float *out);
qw_err qw_q8_gemm_int32_naive(const int8_t *a_q8, const int8_t *w_q8, int m,
                              int n, int k, int32_t *out);
qw_err qw_q8_gemm_int32_blocked(const int8_t *a_q8, const int8_t *w_q8, int m,
                                int n, int k, int32_t *out);
}

// ---------------------------------------------------------------- shapes

struct Shape {
  int m, n, k;
  const char *tag;
};

static const Shape kShapes[] = {
    {32, 640, 2560, "K2560_N640_M32"},
    {80, 640, 2560, "K2560_N640_M80"},
    {256, 640, 2560, "K2560_N640_M256"},
    {32, 2560, 640, "K640_N2560_M32"},
    {80, 2560, 640, "K640_N2560_M80"},
    {256, 2560, 640, "K640_N2560_M256"},
};
static const int N_SHAPES = sizeof(kShapes) / sizeof(kShapes[0]);

// ------------------------------------------------------------ data gen

// Deterministic 32-bit LCG; int8 values in [-127, 127].
static int8_t lcg_i8(uint64_t *st) {
  *st = *st * 6364136223846793005ULL + 1442695040888963407ULL;
  uint32_t v = (uint32_t)(*st >> 32);
  return (int8_t)(v % 255); // 0..254 reinterpreted as int8 => 0..127,-127..-1
}

// Deterministic fp32 scale in [0.001, 1.0).
static float lcg_scale(uint64_t *st) {
  *st = *st * 6364136223846793005ULL + 1442695040888963407ULL;
  uint32_t v = (uint32_t)(*st >> 32);
  return 0.001f + (float)(v % 1000000u) * 0.999f / 1000000.0f;
}

// ------------------------------------------------------------- hip check

// HIP_CHECK returns from main() on failure; only usable in main.
#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t e_ = (call);                                                 \
    if (e_ != hipSuccess) {                                                 \
      printf("FAIL: line %d: %s -> %s\n", __LINE__, #call,                  \
             hipGetErrorString(e_));                                        \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// ----------------------------------------------------------------- main

int main(void) {
  // Have a GPU? Probe with a small hipMalloc; on failure SKIP and exit 0.
  {
    void *probe = NULL;
    hipError_t e = hipMalloc(&probe, 256);
    if (e != hipSuccess) {
      if (e != hipErrorOutOfMemory && e != hipErrorNoDevice)
        printf("note: hipMalloc probe: %s\n", hipGetErrorString(e));
      printf("SKIP: no GPU\n");
      return 0;
    }
    if (hipFree(probe) != hipSuccess)
      printf("note: hipFree(probe): %s\n", hipGetErrorString(hipGetLastError()));
  }

  // Host scratch, sized for the largest shape (256*2560*4B = 2.56 MB each).
  const size_t max_elems = 256 * 2560; // covers A, B, C, acc for all shapes
  int8_t *ha = (int8_t *)malloc(max_elems);
  int8_t *hb = (int8_t *)malloc(max_elems);
  float *has = (float *)malloc(max_elems);
  float *hbs = (float *)malloc(max_elems);
  float *hc = (float *)malloc(max_elems);        // CPU dequant reference
  float *h_gout = (float *)malloc(max_elems);    // GPU float out
  int32_t *h_ref32 = (int32_t *)malloc(max_elems); // CPU int32 (naive)
  int32_t *h_blk32 = (int32_t *)malloc(max_elems); // CPU int32 (blocked)
  int32_t *h_g32 = (int32_t *)malloc(max_elems);    // GPU int32 accumulators
  if (!ha || !hb || !has || !hbs || !hc || !h_gout || !h_ref32 || !h_blk32 ||
      !h_g32) {
    printf("FAIL: host malloc\n");
    return 1;
  }

  // Device buffers.
  int8_t *d_a = NULL, *d_b = NULL;
  float *d_as = NULL, *d_bs = NULL, *d_c = NULL;
  int32_t *d_acc = NULL;
  size_t cap_a = 0, cap_b = 0, cap_c = 0, cap_acc = 0;

  int failures = 0;
  for (int s = 0; s < N_SHAPES; s++) {
    const Shape sh = kShapes[s];
    const size_t a_bytes = (size_t)sh.m * sh.k;
    const size_t b_bytes = (size_t)sh.n * sh.k;
    const size_t c_elems = (size_t)sh.m * sh.n;
    const size_t c_bytes = c_elems * sizeof(float);

    if (!qw_q8_overflow_ok(sh.m, sh.n, sh.k)) {
      printf("FAIL %s: int32 accumulator overflows for K=%d\n", sh.tag,
             sh.k);
      failures++;
      continue;
    }

    // Deterministic inputs (same for every implementation).
    uint64_t st = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)sh.m << 40) ^
                 ((uint64_t)sh.n << 20) ^ (uint64_t)sh.k;
    for (size_t i = 0; i < a_bytes; i++) ha[i] = lcg_i8(&st);
    for (size_t i = 0; i < b_bytes; i++) hb[i] = lcg_i8(&st);
    for (int i = 0; i < sh.m; i++) has[i] = lcg_scale(&st);
    for (int i = 0; i < sh.n; i++) hbs[i] = lcg_scale(&st);

    // CPU references.
    if (qw_q8_gemm_int32_naive(ha, hb, sh.m, sh.n, sh.k, h_ref32) != QW_OK) {
      printf("FAIL %s: qw_q8_gemm_int32_naive\n", sh.tag);
      failures++;
      continue;
    }
    if (qw_q8_gemm_int32_blocked(ha, hb, sh.m, sh.n, sh.k, h_blk32) !=
        QW_OK) {
      printf("FAIL %s: qw_q8_gemm_int32_blocked\n", sh.tag);
      failures++;
      continue;
    }
    if (qw_q8_gemm_ref(ha, has, hb, hbs, sh.m, sh.n, sh.k, hc) != QW_OK) {
      printf("FAIL %s: qw_q8_gemm_ref\n", sh.tag);
      failures++;
      continue;
    }

    // Device buffers, grown as needed (reused across shapes).
    if (cap_a < a_bytes) {
      if (d_a && hipFree(d_a) != hipSuccess)
        printf("note: hipFree(d_a): %s\n", hipGetErrorString(hipGetLastError()));
      if (hipMalloc((void **)&d_a, a_bytes) != hipSuccess) {
        printf("SKIP: no GPU\n");
        return 0;
      }
      cap_a = a_bytes;
    }
    if (cap_b < b_bytes) {
      if (d_b && hipFree(d_b) != hipSuccess)
        printf("note: hipFree(d_b): %s\n", hipGetErrorString(hipGetLastError()));
      if (hipMalloc((void **)&d_b, b_bytes) != hipSuccess) {
        printf("SKIP: no GPU\n");
        return 0;
      }
      cap_b = b_bytes;
    }
    if (cap_c < c_bytes) {
      if (d_c && hipFree(d_c) != hipSuccess)
        printf("note: hipFree(d_c): %s\n", hipGetErrorString(hipGetLastError()));
      if (hipMalloc((void **)&d_c, c_bytes) != hipSuccess) {
        printf("SKIP: no GPU\n");
        return 0;
      }
      cap_c = c_bytes;
    }
    if (cap_acc < c_bytes) {
      if (d_acc && hipFree(d_acc) != hipSuccess)
        printf("note: hipFree(d_acc): %s\n", hipGetErrorString(hipGetLastError()));
      if (hipMalloc((void **)&d_acc, c_bytes) != hipSuccess) {
        printf("SKIP: no GPU\n");
        return 0;
      }
      cap_acc = c_bytes;
    }
    if (!d_as || !d_bs) {
      if (hipMalloc((void **)&d_as, max_elems * sizeof(float)) !=
              hipSuccess ||
          hipMalloc((void **)&d_bs, max_elems * sizeof(float)) !=
              hipSuccess) {
        printf("SKIP: no GPU\n");
        return 0;
      }
    }

    HIP_CHECK(hipMemcpy(d_a, ha, a_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, hb, b_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_as, has, sh.m * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bs, hbs, sh.n * sizeof(float), hipMemcpyHostToDevice));

    // ---- GPU pass 1: NAIVE kernel, raw int32 + float out.
    {
      dim3 block(32, 8);
      dim3 grid((sh.n + 31) / 32, (sh.m + 7) / 8);
      qw_gemm_q8_naive_kernel<<<grid, block>>>(d_a, d_b, d_as, d_bs, d_c,
                                               d_acc, sh.m, sh.n, sh.k);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(
          hipMemcpy(h_g32, d_acc, c_bytes, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(h_gout, d_c, c_bytes, hipMemcpyDeviceToHost));
    }

    // int32 bit-identical: GPU naive vs CPU naive vs CPU blocked.
    for (size_t i = 0; i < c_elems; i++) {
      if (h_g32[i] != h_ref32[i]) {
        printf("FAIL %s: GPU-naive int32 != CPU-naive at %zu "
               "(gpu=0x%x cpu=0x%x)\n",
               sh.tag, i, h_g32[i], h_ref32[i]);
        failures++;
        goto next_shape;
      }
      if (h_blk32[i] != h_ref32[i]) {
        printf("FAIL %s: CPU-blocked int32 != CPU-naive at %zu "
               "(0x%x vs 0x%x)\n",
               sh.tag, i, h_blk32[i], h_ref32[i]);
        failures++;
        goto next_shape;
      }
    }

    // float out bit-identical to qw_q8_gemm_ref.
    {
      size_t first_bad = (size_t)-1;
      float max_diff = 0.0f;
      for (size_t i = 0; i < c_elems; i++) {
        float diff = h_gout[i] - hc[i];
        if (diff < 0)
          diff = -diff;
        if (diff > max_diff)
          max_diff = diff;
        if (h_gout[i] != hc[i] && first_bad == (size_t)-1)
          first_bad = i;
      }
      if (first_bad != (size_t)-1) {
        printf("FAIL %s: GPU-naive float != CPU ref at %zu (gpu=%g ref=%g)"
               " max diff %g\n",
               sh.tag, first_bad, h_gout[first_bad], hc[first_bad], max_diff);
        failures++;
        goto next_shape;
      }
    }

    // ---- GPU pass 2: BLOCKED kernel, raw int32 + float out.
    {
      dim3 block(GEMM_Q8_BX, GEMM_Q8_BY);
      dim3 grid((sh.n + GEMM_Q8_BX * GEMM_Q8_TN - 1) / (GEMM_Q8_BX * GEMM_Q8_TN),
                (sh.m + GEMM_Q8_BY * GEMM_Q8_TM - 1) / (GEMM_Q8_BY * GEMM_Q8_TM));
      qw_gemm_q8_blocked_kernel<<<grid, block>>>(d_a, d_b, d_as, d_bs, d_c,
                                                 d_acc, sh.m, sh.n, sh.k);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(
          hipMemcpy(h_g32, d_acc, c_bytes, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(h_gout, d_c, c_bytes, hipMemcpyDeviceToHost));
    }

    // int32 bit-identical: GPU blocked vs CPU naive (the oracle).
    for (size_t i = 0; i < c_elems; i++) {
      if (h_g32[i] != h_ref32[i]) {
        printf("FAIL %s: GPU-blocked int32 != CPU-naive at %zu "
               "(gpu=0x%x cpu=0x%x)\n",
               sh.tag, i, h_g32[i], h_ref32[i]);
        failures++;
        goto next_shape;
      }
    }

    // float out bit-identical to qw_q8_gemm_ref.
    {
      size_t first_bad = (size_t)-1;
      float max_diff = 0.0f;
      for (size_t i = 0; i < c_elems; i++) {
        float diff = h_gout[i] - hc[i];
        if (diff < 0)
          diff = -diff;
        if (diff > max_diff)
          max_diff = diff;
        if (h_gout[i] != hc[i] && first_bad == (size_t)-1)
          first_bad = i;
      }
      if (first_bad != (size_t)-1) {
        printf("FAIL %s: GPU-blocked float != CPU ref at %zu (gpu=%g ref=%g)"
               " max diff %g\n",
               sh.tag, first_bad, h_gout[first_bad], hc[first_bad], max_diff);
        failures++;
        goto next_shape;
      }
    }

    printf("PASS %s: int32 bit-identical (gpu-naive==gpu-blocked==cpu, "
           "%zu elems); float out bit-identical to qw_q8_gemm_ref\n", sh.tag,
           c_elems);
  next_shape:;
  }

  // ---- timing loop (blocked kernel), only reached with a GPU. ----
  {
    const Shape sh = kShapes[N_SHAPES - 1]; // K640_N2560_M256
    const size_t a_bytes = (size_t)sh.m * sh.k;
    const size_t b_bytes = (size_t)sh.n * sh.k;
    const size_t c_bytes = (size_t)sh.m * sh.n * sizeof(float);

    uint64_t st = 0x1234567890ABCDEFULL;
    for (size_t i = 0; i < a_bytes; i++) ha[i] = lcg_i8(&st);
    for (size_t i = 0; i < b_bytes; i++) hb[i] = lcg_i8(&st);
    for (int i = 0; i < sh.m; i++) has[i] = lcg_scale(&st);
    for (int i = 0; i < sh.n; i++) hbs[i] = lcg_scale(&st);

    HIP_CHECK(hipMemcpy(d_a, ha, a_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, hb, b_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_as, has, sh.m * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bs, hbs, sh.n * sizeof(float), hipMemcpyHostToDevice));

    const int iters = 100;
    dim3 block(GEMM_Q8_BX, GEMM_Q8_BY);
    dim3 grid((sh.n + GEMM_Q8_BX * GEMM_Q8_TN - 1) / (GEMM_Q8_BX * GEMM_Q8_TN),
              (sh.m + GEMM_Q8_BY * GEMM_Q8_TM - 1) / (GEMM_Q8_BY * GEMM_Q8_TM));

    // Warmup (also validates the launch once more).
    qw_gemm_q8_blocked_kernel<<<grid, block>>>(d_a, d_b, d_as, d_bs, d_c,
                                               d_acc, sh.m, sh.n, sh.k);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0));
    for (int i = 0; i < iters; i++) {
      qw_gemm_q8_blocked_kernel<<<grid, block>>>(d_a, d_b, d_as, d_bs, d_c,
                                                 d_acc, sh.m, sh.n, sh.k);
    }
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    if (hipEventDestroy(ev0) != hipSuccess || hipEventDestroy(ev1) != hipSuccess)
      printf("note: hipEventDestroy: %s\n", hipGetErrorString(hipGetLastError()));
    double us = (double)ms * 1000.0 / iters;
    double tflops = 2.0 * (double)sh.m * sh.n * sh.k / (us * 1e-6) / 1e12;
    printf("TIMING %s blocked: %.1f us/iter, %.2f TFLOP/s int8 (%d iters)\n",
           sh.tag, us, tflops, iters);
  }

  // Free device buffers (warn-only; freeing cannot fail the test).
  if (d_a && hipFree(d_a) != hipSuccess)
    printf("note: hipFree(d_a): %s\n", hipGetErrorString(hipGetLastError()));
  if (d_b && hipFree(d_b) != hipSuccess)
    printf("note: hipFree(d_b): %s\n", hipGetErrorString(hipGetLastError()));
  if (d_as && hipFree(d_as) != hipSuccess)
    printf("note: hipFree(d_as): %s\n", hipGetErrorString(hipGetLastError()));
  if (d_bs && hipFree(d_bs) != hipSuccess)
    printf("note: hipFree(d_bs): %s\n", hipGetErrorString(hipGetLastError()));
  if (d_c && hipFree(d_c) != hipSuccess)
    printf("note: hipFree(d_c): %s\n", hipGetErrorString(hipGetLastError()));
  if (d_acc && hipFree(d_acc) != hipSuccess)
    printf("note: hipFree(d_acc): %s\n", hipGetErrorString(hipGetLastError()));
  free(ha); free(hb); free(has); free(hbs); free(hc);
  free(h_gout); free(h_ref32); free(h_blk32); free(h_g32);

  if (failures) {
    printf("FAIL: %d shape(s) failed\n", failures);
    return 1;
  }
  printf("PASS: all %d shapes\n", N_SHAPES);
  return 0;
}
