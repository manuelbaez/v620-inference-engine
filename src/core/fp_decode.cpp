#include "core/fp_decode.hpp"

#include <cstring>

#include "core/common.hpp"

namespace qw {

uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

void csf_expand(const uint8_t *fixed, const uint32_t *exceptions, size_t n_exceptions, size_t rows, size_t cols,
                uint8_t *out) {
    QW_CHECK(rows % 16 == 0 && cols % 2 == 0, "csf: rows a multiple of 16, cols even");
    const size_t slab = 16 + 8 * cols;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *s = fixed + (r / 16) * slab, base = s[r % 16];
        const uint8_t *packed = s + 16 + (r % 16) * (cols / 2);
        for (size_t c = 0; c < cols; c += 2) {
            out[r * cols + c] = uint8_t(base + (packed[c / 2] & 15));
            out[r * cols + c + 1] = uint8_t(base + (packed[c / 2] >> 4));
        }
    }
    for (size_t i = 0; i < n_exceptions; ++i) {
        const size_t pos = exceptions[i] & 0xFFFFFFu;
        QW_CHECK(pos < rows * cols, "csf: exception out of range");
        out[pos] = uint8_t(exceptions[i] >> 24);
    }
}

void mxfp8_to_bf16(const uint8_t *w, const uint8_t *scale, size_t rows, size_t cols, uint16_t *out) {
    QW_CHECK(cols % 32 == 0, "MXFP8: groups of 32");
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            const float v = std::ldexp(f8e4m3_to_f32(w[r * cols + c]), int(scale[r * (cols / 32) + c / 32]) - 127);
            const uint16_t b = f32_to_bf16(v);
            QW_CHECK(bf16_to_f32(b) == v, "MXFP8 value not exact in bf16");
            out[r * cols + c] = b;
        }
}

void nvfp4_to_f32(const uint8_t *w, const uint8_t *codes, float global, size_t rows, size_t cols, float *out) {
    QW_CHECK(cols % 16 == 0, "NVFP4: groups of 16");
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; c += 2) {
            const uint8_t b = w[r * (cols / 2) + c / 2];
            const float s = f8e4m3_to_f32(codes[r * (cols / 16) + c / 16]) * global;
            out[r * cols + c] = E2M1_VALUES[b & 15] * s;
            out[r * cols + c + 1] = E2M1_VALUES[b >> 4] * s;
        }
}

}  // namespace qw
