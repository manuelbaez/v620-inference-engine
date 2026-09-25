// The vision tower on one card: patches of one image (or of one temporal
// slice of a video) -> vision tokens [patches / 4][2560] for the language
// model. Qwen3-VL ViT: patch embedding + interpolated learned positions, 27
// pre-norm blocks with 2D rotary attention over the whole slice, then the 2x2
// patch merger. Residual stream fp32, GEMMs fp16 (rocBLAS).
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdint>
#include <vector>

#include "vision/vision_weights.hpp"

namespace qw {

class VisionEncoder {
public:
    // Uploads the weights to the current device; runs on `stream` with `blas`.
    VisionEncoder(const VisionHostWeights &hw, rocblas_handle blas, hipStream_t stream);
    ~VisionEncoder();
    VisionEncoder(const VisionEncoder &) = delete;
    VisionEncoder &operator=(const VisionEncoder &) = delete;

    // patches fp32 [h*w][1536] in merge-window order (one temporal slice);
    // out fp32 [h*w/4][2560] (host). Synchronous.
    void encode(const float *patches, int h, int w, float *out);
    size_t weight_bytes() const { return w_.bytes; }

private:
    void reserve(int L);  // workspace for slices of L patches
    void attention(int L);

    VisionConfig cfg_;
    VisionDeviceWeights w_;
    rocblas_handle blas_;
    hipStream_t s_;
    int cap_ = 0, qc_ = 0;  // workspace capacity (patches), query rows per attention chunk
    float *x_ = nullptr;
    uint16_t *xn_ = nullptr, *qkv_ = nullptr, *att_ = nullptr, *mlp_ = nullptr, *scores_ = nullptr;
    int32_t *idx_ = nullptr, *pos_ = nullptr;
    float *wts_ = nullptr;
    std::vector<uint16_t> h_patches_;
    std::vector<int32_t> h_idx_, h_pos_;
    std::vector<float> h_wts_;
};

}  // namespace qw
