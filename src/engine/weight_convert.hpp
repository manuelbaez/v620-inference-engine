// Host-side conversion of checkpoint tensors (bf16, compressed-tensors int4)
// into the engine's device layouts, sliced for one rank.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "core/safetensors.hpp"

namespace qw {

// Checkpoint prefix of decoder layer `layer`.
std::string layer_prefix(int layer);

// bf16 -> fp16 with saturation (counted in saturated_weights, reported once at load).
extern std::atomic<int64_t> saturated_weights;
uint16_t bf16_to_f16(uint16_t b);

// Appends rows [r0, r0+n) x cols [c0, c0+nc) of a bf16 matrix as fp16.
void add_rows(std::vector<uint16_t> &dst, const TensorView &t, int64_t r0, int64_t n, int64_t c0 = 0, int64_t nc = -1);
std::vector<float> f32_of(const TensorView &t);
// 1 + w for the rank's columns of a [HC*H] Gemma norm weight, as [HC][SH].
std::vector<float> shard_w1(const TensorView &t, int r);
std::vector<float> plus_one(std::vector<float> v);

// Symmetric int8 of an fp16 [N][K] matrix in groups of `group` along K:
// q[n][k] = round(w / s[n][k / group]), s = max |w| over the group / 127.
void quantize_rows_i8(const std::vector<uint16_t> &fp16, int64_t N, int64_t K, std::vector<int8_t> &q,
                      std::vector<float> &scale, int64_t group);

// Repacks compressed-tensors int4 words (element 8c+i at bits 4i) into the
// kernels' layout (even elements in the low half-word, odd in the high one).
void pack_int4_words(const int32_t *ct, uint32_t *out, size_t n_words);
// Quantizes n_groups groups of 128 bf16 values (row-major, groups along K) to
// symmetric int4 (scale = amax / 7): compressed-tensors words and fp16 scales.
void quantize_int4_g128(const uint16_t *bf16, size_t n_groups, int32_t *ct_words, uint16_t *scales);

}  // namespace qw
