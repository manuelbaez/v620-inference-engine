// Skinny fp16 GEMV for decode batches (M <= 32 rows): one wave per weight row,
// every row's activations dotted against it (weights read once).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace qw::gpu {

// y[m*ys + n] (+)= W[n] . x[m*xs + (n / rpg) * K ...]; xs, ys in elements;
// rpg = rows per group (0: no grouping). W fp16 [N][K], x fp16, y fp32.
void gemv_rows(const uint16_t *W, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M, int rpg,
               bool accumulate, hipStream_t s);
// The same with int8 weights W8 [N][K] and a scale per group of I8_GROUP
// along K: W[n][k] = W8[n][k] * scale[n][k / I8_GROUP]. Half the bytes of fp16
// (decode GEMVs are bandwidth-bound).
#ifndef QW_I8_GROUP
#define QW_I8_GROUP 32  // 16 or 32 (a multiple of the kernel's 16-weight chunks)
#endif
constexpr int I8_GROUP = QW_I8_GROUP;
// e4m3: the bytes are E4M3 instead (MXFP8 matrices kept 8-bit, EngineOptions::dense_fp8), read as value x 2^-8
// (exact in fp16): W[n][k] = E4M3(W8[n][k]) * 2^-8 * scale[n][k / I8_GROUP].
void gemv_rows_i8(const int8_t *W8, const float *scale, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M,
                  int rpg, bool accumulate, hipStream_t s, bool e4m3 = false);
// The fp16 matrix of such E4M3 weights (prefill's GEMMs): out[n][k] = fp16(E4M3 * 2^-8 * scale), clamped to +-65504
// like the loader's bf16 -> fp16 (so the same values as the fp16 copy it replaces). N * K a multiple of 16.
void e4m3_to_f16(const int8_t *W8, const float *scale, int64_t N, int64_t K, uint16_t *out, hipStream_t s);

}  // namespace qw::gpu
