// QW_DENSE_FP8 (docs/DESIGN.md "The FP4 checkpoint"): MXFP8 matrices kept as E4M3 bytes.
//   test_fp8_dense         the encoder on the CPU: random MXFP8 matrices come back exactly, other bf16 ones are refused
//   test_fp8_dense --gpu   also the kernels: e4m3_to_f16 gives the loader's fp16 values bit for bit, and gemv_rows_i8
//                          in e4m3 mode agrees with the fp16 GEMV to rounding (1, 4 and 16 rows)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/weight_convert.hpp"
#include "kernels/gemv.hpp"

using namespace qw;

namespace {
int fails = 0;
void check(const std::string &name, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
    fails += !ok;
}

float e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
    return (b & 0x80) ? -v : v;
}

uint16_t f32_to_bf16_exact(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t(u >> 16);  // the values here have at most 8 significant bits: exact
}

// An MXFP8 matrix as bf16: E4M3 values (never 0x7F / 0xFF, NaN) times 2^k per group of 32, k in [-30, 4].
std::vector<uint16_t> mxfp8_matrix(int64_t N, int64_t K, std::mt19937 &rng) {
    std::vector<uint16_t> m(size_t(N * K));
    std::uniform_int_distribution<int> byte(0, 255), kd(-30, 4);
    for (size_t g = 0; g < m.size() / 32; ++g) {
        const int k = kd(rng);
        for (int j = 0; j < 32; ++j) {
            int b;
            do b = byte(rng);
            while ((b & 0x7F) == 0x7F);
            m[g * 32 + size_t(j)] = f32_to_bf16_exact(std::ldexp(e4m3_value(uint8_t(b)), k));
        }
    }
    return m;
}
}  // namespace

int main(int argc, char **argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    std::mt19937 rng(11);
    const int64_t N = 96, K = 2560;
    const auto m = mxfp8_matrix(N, K, rng);
    std::vector<int8_t> q;
    std::vector<float> sc;
    check("an MXFP8 matrix encodes", encode_mxfp8(m, q, sc));
    bool same = true;
    for (size_t i = 0; i < m.size() && same; ++i)
        same = e4m3_value(uint8_t(q[i])) * sc[i / 32] * std::ldexp(1.f, -8) == bf16_to_f32(m[i]);
    check("... and decodes to the same values", same);

    std::vector<uint16_t> plain(size_t(N * K));
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (auto &v : plain) v = uint16_t([&] { float f = nd(rng); uint32_t u; std::memcpy(&u, &f, 4); return u >> 16; }());
    check("a bf16 matrix of ordinary weights is refused", !encode_mxfp8(plain, q, sc));
    auto one_off = m;
    one_off[1234] ^= 1;  // one mantissa bit more than E4M3 has
    check("one value off the grid is enough to refuse", !encode_mxfp8(one_off, q, sc));

    if (gpu) {
        // values in fp16 range (as the GDN / QSA weights are): powers 2^-20..2^0
        std::uniform_int_distribution<int> kd(-20, 0);
        auto w = mxfp8_matrix(N, K, rng);
        for (size_t g = 0; g < w.size() / 32; ++g) {
            const float f = std::ldexp(1.f, kd(rng));
            for (int j = 0; j < 32; ++j)  // (bit 0 clear: never 0x7F; bit 7 clear: positive, sign tested above)
                w[g * 32 + size_t(j)] = f32_to_bf16_exact(e4m3_value(uint8_t(rng() & 0x7E)) * f);
        }
        QW_CHECK(encode_mxfp8(w, q, sc), "test matrix is not MXFP8");
        std::vector<uint16_t> f16(w.size());
        for (size_t i = 0; i < w.size(); ++i) f16[i] = bf16_to_f16(w[i]);
        int8_t *dq = upload(q);
        float *ds = upload(sc);
        uint16_t *dw = upload(f16), *out = zeros<uint16_t>(f16.size());
        gpu::e4m3_to_f16(dq, ds, N, K, out, nullptr);
        std::vector<uint16_t> got(f16.size());
        CK(hipMemcpy(got.data(), out, got.size() * 2, hipMemcpyDeviceToHost));
        check("e4m3_to_f16 gives the loader's fp16 values bit for bit", got == f16);
        for (int M : {1, 4, 16}) {
            std::vector<uint16_t> x(size_t(M * K));
            for (auto &v : x) v = f32_to_f16(nd(rng) * 50.f);
            uint16_t *dx = upload(x);
            float *y8 = zeros<float>(size_t(M * N)), *y16 = zeros<float>(size_t(M * N));
            gpu::gemv_rows_i8(dq, ds, dx, int(K), y8, int(N), int(N), int(K), M, 0, false, nullptr, true);
            gpu::gemv_rows(dw, dx, int(K), y16, int(N), int(N), int(K), M, 0, false, nullptr);
            std::vector<float> a(size_t(M * N)), b(a.size());
            CK(hipMemcpy(a.data(), y8, a.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b.data(), y16, b.size() * 4, hipMemcpyDeviceToHost));
            double worst = 0, norm = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                worst = std::max(worst, double(std::fabs(a[i] - b[i])));
                norm = std::max(norm, double(std::fabs(b[i])));
            }
            check("gemv e4m3 against fp16, " + std::to_string(M) + " rows: worst " + std::to_string(worst / norm) +
                      " of the largest output",
                  worst <= 1e-5 * norm);
        }
    }
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
