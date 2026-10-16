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

void add_rows_bf16(std::vector<uint16_t> &dst, const TensorView &t, int64_t r0, int64_t n, int64_t c0, int64_t nc) {
    QW_CHECK(t.dtype == DType::BF16 && t.shape.size() >= 2, t.name + ": expected bf16 matrix");
    const int64_t cols = t.numel() / t.dim(0);
    if (nc < 0) nc = cols - c0;
    QW_CHECK(r0 + n <= t.dim(0) && c0 + nc <= cols, t.name + ": slice out of range");
    for (int64_t r = 0; r < n; ++r) {
        const uint16_t *src = t.u16() + (r0 + r) * cols + c0;
        dst.insert(dst.end(), src, src + nc);
    }
}

namespace {
// x as an E4M3 byte, if it is exactly one (no rounding): |x| <= 448, 3 mantissa bits, subnormals down to 2^-9.
bool e4m3_exact(float x, uint8_t &b) {
    const uint8_t s = std::signbit(x) ? 0x80 : 0;
    const float a = std::fabs(x);
    if (a == 0.f) {
        b = s;
        return true;
    }
    if (!(a <= 448.f)) return false;
    const int e = std::ilogb(a);
    if (e < -6) {  // subnormal: m * 2^-9
        const float m = std::ldexp(a, 9);
        if (m != std::floor(m)) return false;
        b = uint8_t(s | uint8_t(m));
        return true;
    }
    const float f = std::ldexp(a, -e) * 8.f - 8.f;
    if (f != std::floor(f)) return false;
    b = uint8_t(s | uint8_t((e + 7) << 3) | uint8_t(f));
    return true;
}
}  // namespace

bool encode_mxfp8(const std::vector<uint16_t> &bf16, std::vector<int8_t> &q, std::vector<float> &scale) {
    QW_CHECK(bf16.size() % 32 == 0, "encode_mxfp8: whole groups of 32");
    q.resize(bf16.size());
    scale.resize(bf16.size() / 32);
    for (size_t g = 0; g < scale.size(); ++g) {
        const uint16_t *v = bf16.data() + g * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(bf16_to_f32(v[j])));
        // the finest power of two that keeps the group within 448: values exact at the encoder's (coarser or equal)
        // power stay exact at it
        int k = amax > 0.f ? std::ilogb(amax) - 8 : 0;
        if (amax > 0.f && std::ldexp(amax, -k) > 448.f) ++k;
        for (int j = 0; j < 32; ++j) {
            uint8_t b;
            if (!e4m3_exact(std::ldexp(bf16_to_f32(v[j]), -k), b)) return false;
            q[g * 32 + size_t(j)] = int8_t(b);
        }
        scale[g] = std::ldexp(1.f, k + 8);  // the kernels read E4M3 as value x 2^-8
    }
    return true;
}

F4PackedHost pack_f4_scales(const std::vector<uint16_t> &words, size_t rows, size_t K32) {
    QW_CHECK(words.size() == rows * K32 && rows < (1u << 24) && 2 * K32 <= 256, "pack_f4_scales: shape");
    F4PackedHost p;
    p.base.resize(rows);
    p.off.resize(rows * K32);
    std::vector<std::pair<uint32_t, uint8_t>> esc;  // (key, code)
    std::vector<uint8_t> codes(2 * K32), sorted(2 * K32);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = 0; j < K32; ++j) {
            codes[2 * j] = uint8_t(words[r * K32 + j] & 0xff);
            codes[2 * j + 1] = uint8_t(words[r * K32 + j] >> 8);
        }
        sorted = codes;
        std::sort(sorted.begin(), sorted.end());
        size_t best = 0, b = 0;
        for (size_t i = 0, e = 0; i < sorted.size(); ++i) {  // window [sorted[i], sorted[i] + 14]
            while (e < sorted.size() && sorted[e] <= sorted[i] + 14) ++e;
            if (e - i > best) best = e - i, b = sorted[i];
        }
        b = std::min<size_t>(b, 241);  // base + 14 <= 255 (a window starting higher holds the same codes)
        p.base[r] = uint8_t(b);
        for (size_t j = 0; j < K32; ++j) {
            uint8_t nib[2];
            for (int h = 0; h < 2; ++h) {
                const int c = codes[2 * j + size_t(h)];
                if (c >= int(b) && c <= int(b) + 14) {
                    nib[h] = uint8_t(c - int(b));
                } else {
                    nib[h] = 15;
                    esc.emplace_back(((uint32_t(r) << 8) | uint32_t(2 * j + size_t(h))) + 1, uint8_t(c));
                }
            }
            p.off[r * K32 + j] = uint8_t(nib[0] | (nib[1] << 4));
        }
    }
    int bits = 10;  // >= 1024 slots, under half full
    while ((size_t(1) << bits) < 2 * esc.size()) ++bits;
    p.shift = uint32_t(32 - bits);
    const uint32_t mask = (1u << bits) - 1;
    p.key.assign(size_t(1) << bits, 0);
    p.val.assign(size_t(1) << bits, 0);
    for (const auto &[k, c] : esc) {
        uint32_t s = (k * 0x9E3779B1u) >> p.shift;
        while (p.key[s]) s = (s + 1) & mask;
        p.key[s] = k;
        p.val[s] = c;
    }
    p.escapes = esc.size();
    return p;
}

uint8_t f4_packed_code(const F4PackedHost &p, size_t K32, uint32_t row, uint32_t idx) {
    const uint32_t o = p.off[row * K32 + idx / 2], nib = (idx & 1) ? o >> 4 : o & 15;
    if (nib != 15) return uint8_t(p.base[row] + nib);
    const uint32_t key = ((row << 8) | idx) + 1, mask = (1u << (32 - p.shift)) - 1;
    uint32_t s = (key * 0x9E3779B1u) >> p.shift;
    while (p.key[s] != key) s = (s + 1) & mask;
    return p.val[s];
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

void quantize_rows_i8(const std::vector<uint16_t> &fp16, int64_t N, int64_t K, std::vector<int8_t> &q,
                      std::vector<float> &scale, int64_t group) {
    QW_CHECK(int64_t(fp16.size()) == N * K && K % group == 0, "quantize_rows_i8: size");
    const int64_t G = K / group;
    q.resize(fp16.size());
    scale.resize(size_t(N * G));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < G; ++g) {
            const uint16_t *w = fp16.data() + n * K + g * group;
            float amax = 0.f;
            for (int64_t k = 0; k < group; ++k) amax = std::max(amax, std::fabs(f16_to_f32(w[k])));
            const float s = amax > 0.f ? amax / 127.f : 1.f;
            scale[size_t(n * G + g)] = s;
            for (int64_t k = 0; k < group; ++k)
                q[size_t(n * K + g * group + k)] = int8_t(std::lrint(std::clamp(f16_to_f32(w[k]) / s, -127.f, 127.f)));
        }
}

}  // namespace qw
