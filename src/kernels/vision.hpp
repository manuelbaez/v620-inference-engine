// Kernels of the vision encoder (a Qwen3-VL ViT: src/vision/vision_encoder).
// The residual stream is fp32 [N][D]; GEMM inputs and outputs are fp16.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace qw::gpu {

// y fp16 [N][D] = layernorm(x fp32 [N][D]) * w + b (eps 1e-6).
void vit_layernorm(const float *x, const float *w, const float *b, int N, int D, uint16_t *y, hipStream_t s);
// x fp32 [N][D] += y fp16 [N][D] (row stride ldy) + bias
void vit_residual_add(float *x, const uint16_t *y, int ldy, const float *bias, int N, int D, hipStream_t s);
// y fp16 [N][D] = act(y + bias) in place; gelu_tanh or exact gelu.
void vit_bias_gelu(uint16_t *y, const float *bias, int N, int D, bool tanh_approx, hipStream_t s);
// y fp16 [N][D] += bias (in place)
void vit_bias(uint16_t *y, const float *bias, int N, int D, hipStream_t s);
// x fp32 [N][D] = y fp16 + bias + sum_k w[n][k] * table[idx[n][k]] (k < 4; the learned
// position table fp32 [*][D] interpolated to each patch)
void vit_embed(float *x, const uint16_t *y, const float *bias, const float *table, const int32_t *idx,
               const float *wts, int N, int D, hipStream_t s);
// Rotates q and k of qkv fp16 [N][3][heads][72] in place by each patch's (h, w) position:
// NeoX halves of 36, angles h*inv[i] for i < 18 and w*inv[i - 18] above.
void vit_rope2d(uint16_t *qkv, const int32_t *pos_hw, int N, int heads, hipStream_t s);
// Row softmax in place over fp16 [rows][n], rows of stride ld, fp32 math.
void vit_softmax(uint16_t *s_rows, int rows, int n, int ld, hipStream_t s);
// out fp32 [N][D] = y fp16 [N][D] + bias
void vit_bias_out(const uint16_t *y, const float *bias, int N, int D, float *out, hipStream_t s);

}  // namespace qw::gpu
