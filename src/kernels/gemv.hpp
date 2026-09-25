// Skinny fp16 GEMV for decode batches (M <= 16 rows): one wave per weight row,
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
constexpr int I8_GROUP = 32;
void gemv_rows_i8(const int8_t *W8, const float *scale, const uint16_t *x, int xs, float *y, int ys, int N, int K, int M,
                  int rpg, bool accumulate, hipStream_t s);

}  // namespace qw::gpu
