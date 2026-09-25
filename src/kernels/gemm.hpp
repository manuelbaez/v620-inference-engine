// Dense GEMMs (rocBLAS) for prefill, and fp16 -> fp32 conversion of their output.
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstddef>
#include <cstdint>

namespace qw::gpu {

// Y[t][n] = sum_k W[n][k] * X[t][k]. W fp16 [N][K]; X fp16 rows of ldx;
// Y fp16 (f16_out) or fp32 rows of ldy. fp16 output is ~5x faster on gfx1030.
void gemm(rocblas_handle h, const uint16_t *W, int N, int K, const uint16_t *X, int ldx, int T, void *Y, int ldy,
          bool f16_out);

void f16_to_f32_T(const uint16_t *x, float *y, size_t n, hipStream_t s);

}  // namespace qw::gpu
