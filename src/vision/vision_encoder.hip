#include "vision/vision_encoder.hpp"

#include <algorithm>
#include <cmath>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "kernels/gemm.hpp"
#include "kernels/vision.hpp"

namespace qw {

namespace {
constexpr size_t SCORE_BYTES = 128u << 20;  // attention scores per chunk of query rows

void free_all(std::initializer_list<void *> ps) {
    for (void *p : ps)
        if (p) (void)hipFree(p);
}
}  // namespace

VisionEncoder::VisionEncoder(const VisionHostWeights &hw, rocblas_handle blas, hipStream_t stream)
    : cfg_(hw.cfg), w_(upload_vision_weights(hw)), blas_(blas), s_(stream) {}

VisionEncoder::~VisionEncoder() {
    free_all({x_, xn_, qkv_, att_, mlp_, scores_, idx_, pos_, wts_});
}

void VisionEncoder::reserve(int L) {
    if (L <= cap_) return;
    free_all({x_, xn_, qkv_, att_, mlp_, scores_, idx_, pos_, wts_});
    const int D = cfg_.dim;
    cap_ = L;
    qc_ = int(std::max<size_t>(16, std::min<size_t>(L, SCORE_BYTES / (size_t(cfg_.heads) * L * 2))));
    x_ = zeros<float>(size_t(L) * D);
    xn_ = zeros<uint16_t>(size_t(L) * D);
    qkv_ = zeros<uint16_t>(size_t(L) * 3 * D);
    att_ = zeros<uint16_t>(size_t(L) * D);
    mlp_ = zeros<uint16_t>(size_t(L) * std::max(cfg_.mlp, cfg_.patch_in));
    scores_ = zeros<uint16_t>(size_t(cfg_.heads) * qc_ * L);
    idx_ = zeros<int32_t>(size_t(L) * 4);
    pos_ = zeros<int32_t>(size_t(L) * 2);
    wts_ = zeros<float>(size_t(L) * 4);
}

// softmax(Q K^T / sqrt(d)) V per head, in chunks of query rows: two strided-
// batched GEMMs over the heads straight out of the qkv rows ([L][3][heads][hd]).
void VisionEncoder::attention(int L) {
    const int D = cfg_.dim, H = cfg_.heads, hd = D / H, ld = 3 * D;
    const float scale = 1.f / std::sqrt(float(hd)), one = 1.f, zero = 0.f;
    const auto f16 = rocblas_datatype_f16_r, f32 = rocblas_datatype_f32_r;
    for (int q0 = 0; q0 < L; q0 += qc_) {
        const int Q = std::min(qc_, L - q0);
        // scores[h][q][k] = q_h . k_h * scale
        rocblas_status st = rocblas_gemm_strided_batched_ex(
            blas_, rocblas_operation_transpose, rocblas_operation_none, L, Q, hd, &scale, qkv_ + D, f16, ld, hd,
            qkv_ + size_t(q0) * ld, f16, ld, hd, &zero, scores_, f16, L, int64_t(Q) * L, scores_, f16, L,
            int64_t(Q) * L, H, f32, rocblas_gemm_algo_standard, 0, 0);
        QW_CHECK(st == rocblas_status_success, "vision attention scores: rocBLAS " + std::to_string(int(st)));
        gpu::vit_softmax(scores_, H * Q, L, L, s_);
        // att[q][h*hd + d] = sum_k p[h][q][k] v_h[k][d]
        st = rocblas_gemm_strided_batched_ex(blas_, rocblas_operation_none, rocblas_operation_none, hd, Q, L, &one,
                                             qkv_ + 2 * D, f16, ld, hd, scores_, f16, L, int64_t(Q) * L, &zero,
                                             att_ + size_t(q0) * D, f16, D, hd, att_ + size_t(q0) * D, f16, D, hd, H,
                                             f32, rocblas_gemm_algo_standard, 0, 0);
        QW_CHECK(st == rocblas_status_success, "vision attention values: rocBLAS " + std::to_string(int(st)));
    }
}

void VisionEncoder::encode(const float *patches, int h, int w, float *out) {
    const int L = h * w, D = cfg_.dim, m = cfg_.merge, side = cfg_.grid_side;
    QW_CHECK(L > 0 && h % m == 0 && w % m == 0, "vision: grid must be a multiple of the merge size");
    reserve(L);
    // host side: fp16 patches; each patch's (row, col) in merge-window order; the
    // bilinear taps (align_corners) of the learned position table at it
    h_patches_.resize(size_t(L) * cfg_.patch_in);
    for (size_t i = 0; i < h_patches_.size(); ++i) h_patches_[i] = f32_to_f16(patches[i]);
    h_idx_.resize(size_t(L) * 4);
    h_wts_.resize(size_t(L) * 4);
    h_pos_.resize(size_t(L) * 2);
    const int bw = w / m;
    for (int i = 0; i < L; ++i) {
        const int in_col = i % m, in_row = (i / m) % m, bcol = (i / (m * m)) % bw, brow = i / (m * m * bw);
        const int row = brow * m + in_row, col = bcol * m + in_col;
        h_pos_[size_t(i) * 2] = row;
        h_pos_[size_t(i) * 2 + 1] = col;
        auto taps = [&](int idx, int n, int &t0, int &t1, float &w0, float &w1) {
            const float src = float(idx) * float(side - 1) / float(std::max(n - 1, 1));
            const float fl = std::floor(src);
            t0 = std::min(int(fl), side - 1);
            t1 = std::min(int(fl) + 1, side - 1);
            w1 = src - fl;
            w0 = 1.f - w1;
        };
        int y0, y1, x0, x1;
        float wy0, wy1, wx0, wx1;
        taps(row, h, y0, y1, wy0, wy1);
        taps(col, w, x0, x1, wx0, wx1);
        const int base = i * 4;
        h_idx_[size_t(base)] = y0 * side + x0;
        h_idx_[size_t(base) + 1] = y0 * side + x1;
        h_idx_[size_t(base) + 2] = y1 * side + x0;
        h_idx_[size_t(base) + 3] = y1 * side + x1;
        h_wts_[size_t(base)] = wy0 * wx0;
        h_wts_[size_t(base) + 1] = wy0 * wx1;
        h_wts_[size_t(base) + 2] = wy1 * wx0;
        h_wts_[size_t(base) + 3] = wy1 * wx1;
    }
    CK(hipMemcpyAsync(mlp_, h_patches_.data(), h_patches_.size() * 2, hipMemcpyHostToDevice, s_));
    CK(hipMemcpyAsync(idx_, h_idx_.data(), h_idx_.size() * 4, hipMemcpyHostToDevice, s_));
    CK(hipMemcpyAsync(wts_, h_wts_.data(), h_wts_.size() * 4, hipMemcpyHostToDevice, s_));
    CK(hipMemcpyAsync(pos_, h_pos_.data(), h_pos_.size() * 4, hipMemcpyHostToDevice, s_));

    // patch embedding + positions
    gpu::gemm(blas_, w_.patch_w, D, cfg_.patch_in, mlp_, cfg_.patch_in, L, att_, D, true);
    gpu::vit_embed(x_, att_, w_.patch_b, w_.pos_table, idx_, wts_, L, D, s_);
    for (const auto &b : w_.blocks) {
        gpu::vit_layernorm(x_, b.n1w, b.n1b, L, D, xn_, s_);
        gpu::gemm(blas_, b.qkv, 3 * D, D, xn_, D, L, qkv_, 3 * D, true);
        gpu::vit_bias(qkv_, b.qkv_b, L, 3 * D, s_);
        gpu::vit_rope2d(qkv_, pos_, L, cfg_.heads, s_);
        attention(L);
        gpu::gemm(blas_, b.proj, D, D, att_, D, L, xn_, D, true);
        gpu::vit_residual_add(x_, xn_, D, b.proj_b, L, D, s_);
        gpu::vit_layernorm(x_, b.n2w, b.n2b, L, D, xn_, s_);
        gpu::gemm(blas_, b.fc1, cfg_.mlp, D, xn_, D, L, mlp_, cfg_.mlp, true);
        gpu::vit_bias_gelu(mlp_, b.fc1_b, L, cfg_.mlp, true, s_);
        gpu::gemm(blas_, b.fc2, D, cfg_.mlp, mlp_, cfg_.mlp, L, xn_, D, true);
        gpu::vit_residual_add(x_, xn_, D, b.fc2_b, L, D, s_);
    }
    // merger: layernorm per patch, then each 2x2 window (4 consecutive patches) as one row
    const int T = L / (m * m), DM = D * m * m;
    gpu::vit_layernorm(x_, w_.m_nw, w_.m_nb, L, D, xn_, s_);
    gpu::gemm(blas_, w_.m_fc1, DM, DM, xn_, DM, T, mlp_, DM, true);
    gpu::vit_bias_gelu(mlp_, w_.m_fc1_b, T, DM, false, s_);
    gpu::gemm(blas_, w_.m_fc2, cfg_.out_dim, DM, mlp_, DM, T, qkv_, cfg_.out_dim, true);
    gpu::vit_bias_out(qkv_, w_.m_fc2_b, T, cfg_.out_dim, x_, s_);
    CK(hipMemcpyAsync(out, x_, size_t(T) * cfg_.out_dim * 4, hipMemcpyDeviceToHost, s_));
    CK(hipStreamSynchronize(s_));
}

}  // namespace qw
