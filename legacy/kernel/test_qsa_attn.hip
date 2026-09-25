// test_qsa_attn.hip — CPU-reference check for the QSA single-token
// attention kernel (kernel/qsa_attn.cuh) on gfx1030.
//
// Builds fp16 K/V in the layout include/qw/kvcache.h uses for one layer of
// one token (K then V, n_kv heads, head_dim 256), plus a 24-head fp16 query,
// runs both the split-K and non-split-K paths, and compares against a naive
// double-precision attention reference.
//
// fp16-appropriate tolerance: query/K/V are stored in fp16 (~11-bit
// mantissa, rel eps ~4.9e-4) and the kernel accumulates in fp32. The score
// is a 256-term dot of fp16 products, so per-output relative error vs the
// double reference is expected on the order of 1e-3. We use:
//     rel_err = |gpu - ref| / (|ref| + 1e-3)   <= 2e-3
// The 1e-3 in the denominator guards the all-equal-scores case where the
// softmax output is ~1/L (tiny) and relative error is dominated by fp16
// input noise, not the softmax math.
//
// Build: hipcc --offload-arch=gfx1030 kernel/test_qsa_attn.hip -o test_qsa_attn
// (or: make hipsyntax for a syntax-only pass). Do NOT run without gfx1030.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "qsa_attn.cuh"

namespace {

constexpr int kNqh = 24;
constexpr int kNkv = 2;
constexpr int kD = 256;
constexpr int kHeadGroup = kNqh / kNkv;  // 12
constexpr double kTol = 2e-3;

// Deterministic fp16 value in [-1, 1] from a 32-bit word.
inline float fp16_rand(uint32_t x) {
  // Fibonacci-style scramble to decorrelate indices.
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  return (float)(x % 2001u) / 1000.0f - 1.0f;  // [-1, 1] step 1e-3
}

inline __half hfp16(float f) { return __float2half(f); }

// CPU reference: naive attention in double precision.
// q: [n_qh][D] (fp16), kbuf: [n_tok][n_kv][D] (fp16), vbuf: same.
// Returns out_ref[n_qh][D] as double.
void ref_attention(const __half *q, const __half *k, const __half *v,
                   int n_tok, double *out_ref) {
  const double inv_sqrt_d = 1.0 / std::sqrt((double)kD);
  for (int qh = 0; qh < kNqh; qh++) {
    int kv = qh / kHeadGroup;
    double mx = -1e300, lsum = 0.0;
    // first pass: scores + max
    std::vector<double> s(n_tok);
    for (int t = 0; t < n_tok; t++) {
      double dot = 0.0;
      for (int d = 0; d < kD; d++) {
        double a = (double)__half2float(q[qh * kD + d]);
        double b = (double)__half2float(k[(size_t)t * kNkv * kD + kv * kD + d]);
        dot += a * b;
      }
      s[t] = dot * inv_sqrt_d;
      if (s[t] > mx) mx = s[t];
    }
    for (int t = 0; t < n_tok; t++) {
      double e = std::exp(s[t] - mx);
      lsum += e;
    }
    double inv_l = 1.0 / lsum;
    for (int d = 0; d < kD; d++) {
      double acc = 0.0;
      for (int t = 0; t < n_tok; t++) {
        double e = std::exp(s[t] - mx) * inv_l;
        double b = (double)__half2float(v[(size_t)t * kNkv * kD + kv * kD + d]);
        acc += e * b;
      }
      out_ref[qh * kD + d] = acc;
    }
  }
}

struct Case {
  int n_tok;
  bool uniform;  // all K rows identical -> all-equal scores
  const char *name;
};

}  // namespace

int main() {
  // Lengths: 1 (edge), 7 (not multiple of chunk, small), 32, 1000 (not
  // multiple of 64: 1000 = 15*64 + 40 -> exercises ragged last split).
  Case cases[] = {
    {1,    false, "len=1"},
    {7,    false, "len=7"},
    {32,   false, "len=32"},
    {1000, false, "len=1000"},
    {65,   false, "len=65 (non-multiple of chunk)"},
    {128,  true,  "len=128 uniform-scores"},
  };

  const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

  // Allocate host buffers at max length.
  const int max_len = 1000;
  std::vector<__half> hq((size_t)kNqh * kD);
  std::vector<__half> hk((size_t)max_len * kNkv * kD);
  std::vector<__half> hv((size_t)max_len * kNkv * kD);
  std::vector<double> ref((size_t)kNqh * kD);
  std::vector<float> hout((size_t)kNqh * kD);

  // Fill query deterministically once.
  for (int i = 0; i < kNqh * kD; i++) hq[i] = hfp16(fp16_rand(0xA11CEu + (uint32_t)i));

  int n_fail = 0;

  __half *dq = nullptr, *dk = nullptr, *dv = nullptr;
  float *dout = nullptr;
  size_t q_bytes = (size_t)kNqh * kD * sizeof(__half);
  size_t kv_bytes = (size_t)max_len * kNkv * kD * sizeof(__half);
  if (hipMalloc((void **)&dq, q_bytes) != hipSuccess ||
      hipMalloc((void **)&dk, kv_bytes) != hipSuccess ||
      hipMalloc((void **)&dv, kv_bytes) != hipSuccess ||
      hipMalloc((void **)&dout, (size_t)kNqh * kD * sizeof(float)) != hipSuccess) {
    printf("SKIP: no GPU\n");
    return 0;
  }

  for (int ci = 0; ci < n_cases; ci++) {
    const Case &c = cases[ci];
    const int n_tok = c.n_tok;

    // Fill K/V. Uniform: every token's K row for each kv head identical
    // (and V identical) so all scores per head are equal.
    uint32_t seed = 0xBEEFu + (uint32_t)ci * 7919u;
    if (c.uniform) {
      // one row per kv head, replicated across tokens
      for (int t = 0; t < n_tok; t++)
        for (int kv = 0; kv < kNkv; kv++)
          for (int d = 0; d < kD; d++) {
            float f = fp16_rand(seed + (kv * kD + d));
            hk[(size_t)t * kNkv * kD + kv * kD + d] = hfp16(f);
            hv[(size_t)t * kNkv * kD + kv * kD + d] = hfp16(fp16_rand(seed + 0x100000u + kv * kD + d));
          }
    } else {
      for (int t = 0; t < n_tok; t++)
        for (int kv = 0; kv < kNkv; kv++)
          for (int d = 0; d < kD; d++) {
            hk[(size_t)t * kNkv * kD + kv * kD + d] = hfp16(fp16_rand(seed + ((size_t)t * kNkv + kv) * kD + d));
            hv[(size_t)t * kNkv * kD + kv * kD + d] = hfp16(fp16_rand(seed + 0x100000u + ((size_t)t * kNkv + kv) * kD + d));
          }
    }

    ref_attention(hq.data(), hk.data(), hv.data(), n_tok, ref.data());

    bool case_ok = true;
    for (int split = 0; split <= 1; split++) {
      // upload
      size_t kv_case = (size_t)n_tok * kNkv * kD * sizeof(__half);
      hipError_t e = hipMemcpy(dq, hq.data(), q_bytes, hipMemcpyHostToDevice);
      if (e != hipSuccess) { printf("FAIL [%s split=%d] hipMemcpy q: %s\n", c.name, split, hipGetErrorString(e)); case_ok = false; break; }
      e = hipMemcpy(dk, hk.data(), kv_case, hipMemcpyHostToDevice);
      if (e != hipSuccess) { printf("FAIL [%s split=%d] hipMemcpy k: %s\n", c.name, split, hipGetErrorString(e)); case_ok = false; break; }
      e = hipMemcpy(dv, hv.data(), kv_case, hipMemcpyHostToDevice);
      if (e != hipSuccess) { printf("FAIL [%s split=%d] hipMemcpy v: %s\n", c.name, split, hipGetErrorString(e)); case_ok = false; break; }

      e = qsa::qsa_attn_launch(dq, dk, dv, dout, n_tok, kNqh, kNkv, kD, split);
      if (e != hipSuccess) { printf("FAIL [%s split=%d] launch: %s\n", c.name, split, hipGetErrorString(e)); case_ok = false; break; }
      e = hipDeviceSynchronize();
      if (e != hipSuccess) { printf("FAIL [%s split=%d] sync: %s\n", c.name, split, hipGetErrorString(e)); case_ok = false; break; }

      e = hipMemcpy(hout.data(), dout, (size_t)kNqh * kD * sizeof(float), hipMemcpyDeviceToHost);
      if (e != hipSuccess) { printf("FAIL [%s split=%d] hipMemcpy out: %s\n", c.name, split, hipGetErrorString(e)); case_ok = false; break; }

      double max_rel = 0.0;
      int bad = -1;
      for (int i = 0; i < kNqh * kD; i++) {
        double g = (double)hout[i];
        double r = ref[i];
        double rel = std::fabs(g - r) / (std::fabs(r) + 1e-3);
        if (rel > max_rel) { max_rel = rel; bad = i; }
      }
      if (max_rel > kTol) {
        case_ok = false;
        printf("FAIL [%s split=%d] max_rel=%.3e > %.1e at idx=%d (gpu=%.6f ref=%.6f)\n",
               c.name, split, max_rel, kTol, bad, (double)hout[bad], ref[bad]);
      } else {
        printf("PASS [%s split=%d] max_rel=%.3e (tol %.1e)\n", c.name, split, max_rel, kTol);
      }
    }
    if (!case_ok) n_fail++;
  }

  (void)hipFree(dq); (void)hipFree(dk); (void)hipFree(dv); (void)hipFree(dout);

  if (n_fail) {
    printf("RESULT: FAIL (%d case(s))\n", n_fail);
    return 1;
  }
  printf("RESULT: PASS (all cases)\n");
  return 0;
}
