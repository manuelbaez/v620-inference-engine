// The routed experts' prefill kernels (moe_experts_T: tiles, gate/up, down) alone on one GPU and one stream, by
// expert format, with the same routing: AWQ int4 (a scale per 128), NVFP4 with random E2M1 values, and NVFP4 whose
// values all have a nonzero exponent (no subnormal intermediates: E2M1 0.5 is fp16 2^-15 in the kernels' decode).
// In prefill the kernel trace showed the NVFP4 down kernel 39% slower than AWQ's, its calls spread much wider; this
// tells a cost of the format from the overlap with the other micro-batch's kernels. Run under rocprofv3 --kernel-trace
// --stats for gate/up and down separately.
// --contend: a second stream copies 512 MB back and forth meanwhile (memory traffic like the other micro-batch's).
//   moe_format_bench [--tokens 2048] [--reps 50] [--contend]
#include <hip/hip_runtime.h>

#include <chrono>
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

int main(int argc, char **argv) {
    try {
        int T = 2048, reps = 50;
        bool contend = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) T = std::atoi(argv[++i]);
            if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
            if (a == "--contend") contend = true;
        }
        CK(hipSetDevice(0));
        std::mt19937 rng(3);
        // the same routing for every format: random router logits for T tokens, the first rank's 128 experts
        std::vector<float> logits(size_t(T) * N_EXPERTS);
        for (auto &x : logits) x = std::normal_distribution<float>(0.f, 1.f)(rng);
        std::vector<uint16_t> bin(size_t(T) * H);
        for (auto &x : bin) x = f32_to_f16(std::normal_distribution<float>(0.f, 0.5f)(rng));
        float *d_logits = dev(logits), *tok_w;
        uint16_t *d_bin = dev(bin), *h, *yp;
        int32_t *tok_e, *counts, *offsets, *fill, *tok_slot, *pair_tok, *tiles;
        const size_t pairs = size_t(T) * TOP_K;
        CK(hipMalloc(&tok_e, pairs * 4));
        CK(hipMalloc(&tok_w, pairs * 4));
        CK(hipMalloc(&counts, EXP_L * 4));
        CK(hipMalloc(&offsets, (EXP_L + 1) * 4));
        CK(hipMalloc(&fill, EXP_L * 4));
        CK(hipMalloc(&tok_slot, pairs * 4));
        CK(hipMalloc(&pair_tok, pairs * 4));
        CK(hipMalloc(&tiles, (2 * (pairs / 32 + EXP_L) + 1) * 4));
        CK(hipMalloc(&h, pairs * FFN * 2));
        CK(hipMalloc(&yp, pairs * H * 2));
        CK(hipMemset(counts, 0, EXP_L * 4));  // (as prefill.hip does)
        gpu::moe_route_T(d_logits, T, 0, tok_e, tok_w, counts, nullptr);
        CK(hipMemset(fill, 0, EXP_L * 4));
        gpu::moe_scatter_T(tok_e, counts, T, offsets, fill, tok_slot, pair_tok, nullptr);
        CK(hipDeviceSynchronize());

        const size_t gu_words = size_t(EXP_L) * FFN * H / 8, d_words = size_t(EXP_L) * H * FFN / 8;
        auto words = [&](size_t n, bool normal_only) {
            std::vector<uint32_t> v(n);
            for (auto &x : v) {
                x = rng();
                if (normal_only)  // every nibble's exponent bits (1-2) nonzero: magnitudes 1..6
                    for (int k = 0; k < 8; ++k)
                        if (((x >> (4 * k + 1)) & 3u) == 0) x |= 2u << (4 * k);
            }
            return v;
        };
        struct Fmt {
            const char *name;
            int group;
            bool normal_only;
        };
        for (const Fmt f : {Fmt{"AWQ int4, scale per 128", QGROUP, false}, Fmt{"NVFP4, random values", QGROUP_F4, false},
                            Fmt{"NVFP4, no subnormal intermediates", QGROUP_F4, true}}) {
            const int S = scale_span(f.group);
            auto scales = [&](size_t n) {  // fp16 scales (AWQ) or two E4M3 codes per word, codes like the checkpoint's
                std::vector<uint16_t> v(n);
                std::uniform_int_distribution<int> code(40, 70);
                for (auto &x : v)
                    x = f.group == QGROUP ? f32_to_f16(std::uniform_real_distribution<float>(0.001f, 0.01f)(rng))
                                          : uint16_t(code(rng) | (code(rng) << 8));
                return v;
            };
            const auto hgw = words(gu_words, f.normal_only), huw = words(gu_words, f.normal_only),
                       hdw = words(d_words, f.normal_only);
            uint32_t *gw = dev(hgw), *uw = dev(huw), *dw = dev(hdw);
            uint16_t *gs = dev(scales(size_t(EXP_L) * FFN * H / S)), *us = dev(scales(size_t(EXP_L) * FFN * H / S)),
                     *ds = dev(scales(size_t(EXP_L) * H * FFN / S));
            float *glob = dev(std::vector<float>(3 * EXP_L, 1e-4f));
            auto run = [&] {
                gpu::moe_experts_T(gw, gs, uw, us, dw, ds, counts, offsets, pair_tok, d_bin, T, tiles, h, yp, nullptr,
                                   nullptr, nullptr, f.group, f.group == QGROUP ? nullptr : glob);
            };
            for (int i = 0; i < 5; ++i) run();
            CK(hipDeviceSynchronize());
            hipStream_t s2 = nullptr;
            void *ca = nullptr, *cb = nullptr;
            const size_t cbytes = size_t(512) << 20;
            if (contend) {  // enough copies queued on another stream to outlast the timed calls
                CK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
                CK(hipMalloc(&ca, cbytes));
                CK(hipMalloc(&cb, cbytes));
                for (int i = 0; i < 4 * reps; ++i) CK(hipMemcpyAsync(i % 2 ? ca : cb, i % 2 ? cb : ca, cbytes, hipMemcpyDeviceToDevice, s2));
            }
            hipEvent_t a, b;
            CK(hipEventCreate(&a));
            CK(hipEventCreate(&b));
            CK(hipEventRecord(a, nullptr));
            for (int i = 0; i < reps; ++i) run();
            CK(hipEventRecord(b, nullptr));
            CK(hipEventSynchronize(b));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, a, b));
            std::printf("%-36s %d tokens: %.3f ms per call (tiles + gate/up + down)%s\n", f.name, T, ms / reps,
                        contend ? ", with copies on another stream" : "");
            if (contend) {
                CK(hipStreamSynchronize(s2));
                CK(hipStreamDestroy(s2));
                CK(hipFree(ca));
                CK(hipFree(cb));
            }
            for (void *p : {(void *)gw, (void *)uw, (void *)dw, (void *)gs, (void *)us, (void *)ds, (void *)glob})
                CK(hipFree(p));
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
