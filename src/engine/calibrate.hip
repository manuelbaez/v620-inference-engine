// GPTQ calibration: Hessians X^T X of the inputs of the decode matrices that
// have int8 copies, accumulated on each rank during prefills
// (tools/qw_calibrate, then tools/gptq_int8.py).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdio>
#include <filesystem>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

namespace qw {

void Engine::calib_begin(int lo, int hi) {
    QW_CHECK(std::getenv("QW_NO_SPLIT"), "calibration needs QW_NO_SPLIT=1 (one stream per rank)");
    calib_lo_ = lo;
    calib_hi_ = hi;
}

// H += X^T X for X fp16 [rows][K] with row stride ldx (H is symmetric, so the
// column-major product X X^T of rocBLAS's view is the same matrix).
void Engine::calib_acc(Rank &rk, const std::string &name, const uint16_t *X, int ldx, int rows, int K,
                       rocblas_handle blas) {
    auto &c = rk.calib[name];
    if (!c.H) {
        c.H = zeros<float>(size_t(K) * K);
        c.K = K;
    }
    QW_CHECK(c.K == K, "calibration: " + name + " changed width");
    const float one = 1.f;
    const rocblas_status st = rocblas_gemm_ex(blas, rocblas_operation_none, rocblas_operation_transpose, K, K, rows,
                                              &one, X, rocblas_datatype_f16_r, ldx, X, rocblas_datatype_f16_r, ldx,
                                              &one, c.H, rocblas_datatype_f32_r, K, c.H, rocblas_datatype_f32_r, K,
                                              rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    QW_CHECK(st == rocblas_status_success, "calibration GEMM: rocBLAS " + std::to_string(int(st)));
    c.rows += rows;
}

void Engine::calib_dump(const std::string &dir) {
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        CK(hipStreamSynchronize(rk.s));
        const std::string sub = dir + "/r" + std::to_string(rk.r) + "/";
        std::filesystem::create_directories(sub);
        for (auto &[name, c] : rk.calib) {
            std::vector<float> h(size_t(c.K) * c.K);
            CK(hipMemcpy(h.data(), c.H, h.size() * 4, hipMemcpyDeviceToHost));
            std::FILE *f = std::fopen((sub + name + ".h").c_str(), "wb");
            QW_CHECK(f, "cannot write " + sub + name + ".h");
            const int64_t K = c.K;
            std::fwrite(&K, 8, 1, f);
            std::fwrite(&c.rows, 8, 1, f);
            std::fwrite(h.data(), 4, h.size(), f);
            std::fclose(f);
            CK(hipFree(c.H));
        }
        log("calibration: rank %d wrote %zu Hessians", rk.r, rk.calib.size());
        rk.calib.clear();
    }
    calib_lo_ = calib_hi_ = -1;
}

}  // namespace qw
