// The decode MoE path against the prefill one on one GPU: random int4
// experts, random router logits for M = 1..16 rows; moe_route_B routes them,
// then moe_experts_P (decode) and moe_experts_T (prefill tiles) compute every
// (token, expert) pair from the same routing; their fp16 outputs must agree.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/shard.hpp"
#include "kernels/moe.hpp"

using namespace qw;
using namespace qw::cfg;

#define CK(x)                                                                           \
    do {                                                                                \
        hipError_t err_ = (x);                                                          \
        if (err_ != hipSuccess) fail(std::string(#x) + ": " + hipGetErrorString(err_)); \
    } while (0)

template <typename T>
T *dev(const std::vector<T> &v) {
    T *d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T>
std::vector<T> host(const T *d, size_t n) {
    std::vector<T> v(n);
    CK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

int main() {
    try {
        CK(hipSetDevice(0));
        std::mt19937 rng(1);
        auto words = [&](size_t n) {
            std::vector<uint32_t> v(n);
            for (auto &x : v) x = rng();
            return v;
        };
        int fails = 0;
        // both expert formats: a scale per 128 inputs, and a scale with a zero point per 32 (core/config.hpp)
        for (const int G : {QGROUP, QGROUP_Z}) {
        auto scales = [&](size_t n) {
            std::vector<uint16_t> v(n);
            for (auto &x : v) {
                x = f32_to_f16(std::uniform_real_distribution<float>(0.001f, 0.01f)(rng));
                if (G == QGROUP_Z) x = qz_word(x & QZ_SCALE_MASK, int(rng() % 16));
            }
            return v;
        };
        const size_t gu_words = size_t(EXP_L) * FFN * H / 8, d_words = size_t(EXP_L) * H * FFN / 8;
        const auto hgw = words(gu_words), huw = words(gu_words), hdw = words(d_words);
        const auto hgs = scales(size_t(EXP_L) * FFN * H / G), hus = scales(size_t(EXP_L) * FFN * H / G),
                   hds = scales(size_t(EXP_L) * H * FFN / G);
        uint32_t *gw = dev(hgw), *uw = dev(huw), *dw = dev(hdw);
        uint16_t *gs = dev(hgs), *us = dev(hus), *ds = dev(hds);
        // element k of row n of expert e in the kernels' packed layout (even elements in the low half-word)
        auto weight = [G](const std::vector<uint32_t> &w, const std::vector<uint16_t> &sc, int e, int rows, int K, int n,
                         int k) {
            const size_t row = size_t(e) * rows + size_t(n);
            const uint32_t word = w[row * size_t(K / 8) + size_t(k / 8)];
            const int i = k % 8, bit = (i & 1) ? 16 + 4 * (i >> 1) : 4 * (i >> 1);
            const uint16_t sw = sc[row * size_t(K / G) + size_t(k / G)];
            const int q = int((word >> bit) & 0xf) - (G == QGROUP ? 8 : qz_nibble(sw));
            return float(q) * f16_to_f32(G == QGROUP ? sw : uint16_t(sw & QZ_SCALE_MASK));
        };
        constexpr int MR = 16;
        int32_t *counts, *offsets, *tok_slot, *pair_tok, *tiles;
        float *tok_w;
        uint16_t *hP, *hT, *yP, *yT;
        CK(hipMalloc(&counts, EXP_L * 4));
        CK(hipMalloc(&offsets, (EXP_L + 1) * 4));
        CK(hipMalloc(&tok_slot, MR * TOP_K * 4));
        CK(hipMalloc(&pair_tok, MR * TOP_K * 4));
        CK(hipMalloc(&tiles, (2 * (MR * TOP_K / 32 + EXP_L) + 1) * 4));
        CK(hipMalloc(&tok_w, MR * TOP_K * 4));
        CK(hipMalloc(&hP, size_t(MR) * TOP_K * FFN * 2));
        CK(hipMalloc(&hT, size_t(MR) * TOP_K * FFN * 2));
        CK(hipMalloc(&yP, size_t(MR) * TOP_K * H * 2));
        CK(hipMalloc(&yT, size_t(MR) * TOP_K * H * 2));
        std::printf("experts with a scale per %d inputs\n", G);
        for (int trial = 0; trial < 64; ++trial) {
            const int M = 1 + trial % MR;
            std::vector<float> logits(size_t(M) * N_EXPERTS);
            for (auto &x : logits) x = std::normal_distribution<float>(0.f, 1.f)(rng);
            // half the trials: pile the rows onto a few local experts (shared experts, empty ones)
            if (trial % 2)
                for (int m = 0; m < M; ++m)
                    for (int e = 0; e < 12; ++e) logits[size_t(m) * N_EXPERTS + size_t(e * 7 % EXP_L)] += 6.f;
            std::vector<uint16_t> bin(size_t(M) * H);
            for (auto &x : bin) x = f32_to_f16(std::uniform_real_distribution<float>(-1.f, 1.f)(rng));
            float *d_logits = dev(logits);
            uint16_t *d_bin = dev(bin);
            gpu::moe_route_B(d_logits, M, 0, counts, offsets, tok_w, tok_slot, pair_tok, nullptr);
            CK(hipMemset(yP, 0, size_t(MR) * TOP_K * H * 2));
            CK(hipMemset(yT, 0, size_t(MR) * TOP_K * H * 2));
            gpu::moe_experts_P(gw, gs, uw, us, dw, ds, offsets, pair_tok, d_bin, M, hP, yP, nullptr, G);
            gpu::moe_experts_T(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, d_bin, M, tiles, hT, yT, nullptr,
                               nullptr, nullptr, G);
            CK(hipDeviceSynchronize());
            const auto off = host(offsets, EXP_L + 1);
            const auto ts = host(tok_slot, size_t(M) * TOP_K);
            const auto pt = host(pair_tok, size_t(M) * TOP_K);
            const int n_pairs = off[EXP_L];
            // routing sanity: slots are a permutation of [0, n_pairs), pair_tok matches
            std::vector<int> seen(size_t(n_pairs), 0);
            int bad_route = 0;
            for (int i = 0; i < M * TOP_K; ++i)
                if (ts[size_t(i)] >= 0) {
                    if (ts[size_t(i)] >= n_pairs || seen[size_t(ts[size_t(i)])]++ ||
                        pt[size_t(ts[size_t(i)])] != i / TOP_K)
                        ++bad_route;
                }
            for (int p = 0; p < n_pairs; ++p) bad_route += seen[size_t(p)] != 1;
            const auto a = host(yP, size_t(n_pairs) * H), b = host(yT, size_t(n_pairs) * H);
            double worst = 0, scale = 1e-6;  // max |P - T| / max |T|
            for (size_t i = 0; i < a.size(); ++i) scale = std::max(scale, double(std::fabs(f16_to_f32(b[i]))));
            for (size_t i = 0; i < a.size(); ++i)
                worst = std::max(worst, std::fabs(f16_to_f32(a[i]) - f16_to_f32(b[i])) / scale);
            // fp32 reference for the first pair of the first local expert
            double ref_err_P = 0, ref_err_T = 0;
            if (n_pairs > 0) {
                int e = 0;
                while (off[size_t(e) + 1] == 0) ++e;
                const int t = pt[0];
                std::vector<float> hh(FFN);
                for (int n = 0; n < FFN; ++n) {
                    double g = 0, u = 0;
                    for (int k = 0; k < H; ++k) {
                        const double x = f16_to_f32(bin[size_t(t) * H + size_t(k)]);
                        g += weight(hgw, hgs, e, FFN, H, n, k) * x;
                        u += weight(huw, hus, e, FFN, H, n, k) * x;
                    }
                    hh[size_t(n)] = f16_to_f32(f32_to_f16(float(g / (1.0 + std::exp(-g)) * u)));
                }
                double scale = 0;
                std::vector<double> y(H);
                for (int n = 0; n < H; ++n) {
                    double acc = 0;
                    for (int k = 0; k < FFN; ++k) acc += weight(hdw, hds, e, H, FFN, n, k) * hh[size_t(k)];
                    y[size_t(n)] = acc;
                    scale = std::max(scale, std::fabs(acc));
                }
                for (int n = 0; n < H; ++n) {
                    ref_err_P = std::max(ref_err_P, std::fabs(f16_to_f32(a[size_t(n)]) - y[size_t(n)]) / scale);
                    ref_err_T = std::max(ref_err_T, std::fabs(f16_to_f32(b[size_t(n)]) - y[size_t(n)]) / scale);
                }
            }
            const bool ok = bad_route == 0 && worst < 0.01 && ref_err_P < 0.02 && ref_err_T < 0.02;
            if (trial < 6)
                std::printf("   pair 0 vs fp32 reference: P %.4f, T %.4f (max abs err / max |y|)\n", ref_err_P,
                            ref_err_T);
            fails += !ok;
            if (!ok || trial < 4)
                std::printf("M=%2d pairs %3d: routing errors %d, max diff P vs T (of max |y|) %.4f %s\n", M, n_pairs,
                            bad_route, worst, ok ? "" : "<-- FAIL");
            CK(hipFree(d_logits));
            CK(hipFree(d_bin));
        }
        for (void *d : {(void *)gw, (void *)uw, (void *)dw, (void *)gs, (void *)us, (void *)ds, (void *)counts,
                        (void *)offsets, (void *)tok_slot, (void *)pair_tok, (void *)tiles, (void *)tok_w, (void *)hP,
                        (void *)hT, (void *)yP, (void *)yT})
            CK(hipFree(d));
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
