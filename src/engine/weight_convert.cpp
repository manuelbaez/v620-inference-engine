#include "engine/weight_convert.hpp"

#include <algorithm>
#include <cmath>

#include "core/common.hpp"
#include "core/shard.hpp"

namespace qw {

using namespace cfg;

std::string layer_prefix(int layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

std::atomic<int64_t> saturated_weights{0};

uint16_t bf16_to_f16(uint16_t b) {
    float f = bf16_to_f32(b);
    if (std::fabs(f) > 65504.f) {
        ++saturated_weights;
        f = f > 0 ? 65504.f : -65504.f;
    }
    return f32_to_f16(f);
}

void add_rows(std::vector<uint16_t> &dst, const TensorView &t, int64_t r0, int64_t n, int64_t c0, int64_t nc) {
    QW_CHECK(t.dtype == DType::BF16 && t.shape.size() >= 2, t.name + ": expected bf16 matrix");
    const int64_t cols = t.numel() / t.dim(0);
    if (nc < 0) nc = cols - c0;
    QW_CHECK(r0 + n <= t.dim(0) && c0 + nc <= cols, t.name + ": slice out of range");
    size_t o = dst.size();
    dst.resize(o + size_t(n * nc));
    for (int64_t r = 0; r < n; ++r) {
        const uint16_t *src = t.u16() + (r0 + r) * cols + c0;
        for (int64_t c = 0; c < nc; ++c) dst[o + size_t(r * nc + c)] = bf16_to_f16(src[c]);
    }
}

std::vector<float> f32_of(const TensorView &t) {
    QW_CHECK(t.dtype == DType::BF16, t.name + ": expected bf16");
    std::vector<float> v(size_t(t.numel()));
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_to_f32(t.u16()[i]);
    return v;
}

std::vector<float> shard_w1(const TensorView &t, int r) {
    auto w = f32_of(t);
    std::vector<float> o(size_t(HC) * SH);
    for (int s = 0; s < HC; ++s)
        for (int j = 0; j < SH; ++j) o[size_t(s * SH + j)] = 1.f + w[size_t(s * H + r * SH + j)];
    return o;
}

std::vector<float> plus_one(std::vector<float> v) {
    for (auto &x : v) x += 1.f;
    return v;
}

void pack_int4_words(const int32_t *ct, uint32_t *out, size_t n_words) {
    for (size_t i = 0; i < n_words; ++i) {
        uint32_t w = uint32_t(ct[i]), r = 0;
        for (int k = 0; k < 8; ++k) {
            uint32_t nib = (w >> (4 * k)) & 0xf;
            int bit = (k & 1) ? 16 + 4 * (k >> 1) : 4 * (k >> 1);
            r |= nib << bit;
        }
        out[i] = r;
    }
}

void quantize_int4_g128(const uint16_t *src, size_t n_groups, int32_t *ct, uint16_t *scales) {
    for (size_t g = 0; g < n_groups; ++g) {
        const uint16_t *x = src + g * QGROUP;
        float amax = 0.f;
        for (int k = 0; k < QGROUP; ++k) amax = std::max(amax, std::fabs(bf16_to_f32(x[k])));
        const uint16_t sc16 = f32_to_f16(amax / 7.f);
        const float sc = f16_to_f32(sc16), inv = sc > 0.f ? 1.f / sc : 0.f;
        for (int k = 0; k < QGROUP; k += 8) {
            uint32_t word = 0;
            for (int j = 0; j < 8; ++j) {
                const long q = std::lrint(bf16_to_f32(x[k + j]) * inv);
                word |= uint32_t(std::clamp<long>(q, -8, 7) + 8) << (4 * j);
            }
            ct[(g * QGROUP + size_t(k)) / 8] = int32_t(word);
        }
        scales[g] = sc16;
    }
}

}  // namespace qw
