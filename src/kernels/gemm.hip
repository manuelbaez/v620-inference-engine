#include "kernels/gemm.hpp"

#include <string>

#include "core/common.hpp"
#include "kernels/model_common.cuh"

namespace qw::gpu {

void gemm(rocblas_handle h, const uint16_t *W, int N, int K, const uint16_t *X, int ldx, int T, void *Y, int ldy,
          bool f16_out) {
    const float alpha = 1.f, beta = 0.f;
    const auto ct = f16_out ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
    rocblas_status st = rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, T, K, &alpha, W,
                                        rocblas_datatype_f16_r, K, X, rocblas_datatype_f16_r, ldx, &beta, Y, ct, ldy, Y,
                                        ct, ldy, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (st != rocblas_status_success) fail("rocblas_gemm_ex failed: " + std::to_string(int(st)));
}

__global__ void f16_to_f32_kernel(const __half *x, float *y, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

void f16_to_f32_T(const uint16_t *x, float *y, size_t n, hipStream_t s) {
    f16_to_f32_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(reinterpret_cast<const __half *>(x), y, n);
}

}  // namespace qw::gpu
