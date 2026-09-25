// Small shared utilities: errors, logging, number conversions.
#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace qw {

[[noreturn]] inline void fail(const std::string &msg) {
    throw std::runtime_error(msg);
}

#define QW_CHECK(cond, msg)                                                                           \
    do {                                                                                              \
        if (!(cond)) ::qw::fail(std::string(__FILE__ ":") + std::to_string(__LINE__) + ": " + (msg)); \
    } while (0)

inline void log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
inline void log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

// bf16 is the top half of an fp32.
inline float bf16_to_f32(uint16_t v) {
    uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) {
            u = sign;
        } else {  // subnormal: renormalize
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                --exp;
            }
            man &= 0x3ff;
            u = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (man << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Round-to-nearest-even fp32 -> fp16.
inline uint16_t f32_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = int32_t((u >> 23) & 0xff) - 127 + 15;
    uint32_t man = u & 0x7fffff;
    if (((u >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00 | (man ? 0x200 : 0));
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        man |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(exp) << 10) | (man >> 13);
    uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
    return uint16_t(half);
}

}  // namespace qw
