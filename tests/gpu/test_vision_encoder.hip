// Vision tower against the HF reference (tools/vision_ref.py, converted by
// tools/vision_ref_raw.py to raw files): per case, the output tokens of every
// temporal slice must match the fp32 reference to a relative L2 error under
// 2e-2, and the same slice must encode the same on every card (to 1e-3:
// rocBLAS split-K kernels may sum in another order). Also
// reports encode times.
//   test_vision_encoder DIR [cases...]   (DIR holds <case>.grid.i64, .pixels.f32, .embeds.f32)
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/safetensors.hpp"
#include "engine/device_mem.hpp"
#include "vision/vision_encoder.hpp"

using namespace qw;

template <typename T>
static std::vector<T> read_raw(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    QW_CHECK(f.good(), "cannot read " + path);
    std::vector<T> v(size_t(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

int main(int argc, char **argv) {
    try {
        QW_CHECK(argc >= 3, "usage: test_vision_encoder DIR case...");
        const std::string dir = argv[1];
        SafeTensors st;
        st.add_index("/mnt/llms/qwen3.8-flash-next-awq");
        const auto hw = load_vision_weights(st);
        QW_CHECK(hw != nullptr, "no vision tower in the checkpoint");
        int devices = 0;
        CK(hipGetDeviceCount(&devices));
        std::vector<std::unique_ptr<VisionEncoder>> enc;
        std::vector<hipStream_t> streams(static_cast<size_t>(devices));
        std::vector<rocblas_handle> blas(static_cast<size_t>(devices));
        for (int d = 0; d < devices; ++d) {
            CK(hipSetDevice(d));
            CK(hipStreamCreateWithFlags(&streams[size_t(d)], hipStreamNonBlocking));
            rocblas_create_handle(&blas[size_t(d)]);
            rocblas_set_stream(blas[size_t(d)], streams[size_t(d)]);
            enc.push_back(std::make_unique<VisionEncoder>(*hw, blas[size_t(d)], streams[size_t(d)]));
        }
        std::printf("vision tower: %.2f GB per card, %d cards\n", double(enc[0]->weight_bytes()) / 1e9, devices);
        int fails = 0;
        for (int c = 2; c < argc; ++c) {
            const std::string name = argv[c];
            const auto grid = read_raw<int64_t>(dir + "/" + name + ".grid.i64");
            const auto pixels = read_raw<float>(dir + "/" + name + ".pixels.f32");
            const auto ref = read_raw<float>(dir + "/" + name + ".embeds.f32");
            std::vector<float> out(ref.size());
            size_t p_off = 0, o_off = 0;
            double worst = 0, ms = 0;
            int slices = 0;
            for (size_t g = 0; g + 2 < grid.size(); g += 3)
                for (int t = 0; t < grid[g]; ++t) {
                    const int h = int(grid[g + 1]), w = int(grid[g + 2]);
                    CK(hipSetDevice(0));
                    const auto t0 = std::chrono::steady_clock::now();
                    enc[0]->encode(pixels.data() + p_off * 1536, h, w, out.data() + o_off * 2560);
                    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    ++slices;
                    const size_t n = size_t(h) * w / 4 * 2560;
                    double d = 0, z = 0;
                    for (size_t i = 0; i < n; ++i) {
                        const double a = out[o_off * 2560 + i], b = ref[o_off * 2560 + i];
                        d += (a - b) * (a - b);
                        z += b * b;
                    }
                    worst = std::max(worst, std::sqrt(d / std::max(z, 1e-30)));
                    if (slices == 1)  // same slice on every other card
                        for (int dv = 1; dv < devices; ++dv) {
                            std::vector<float> o2(size_t(h) * w / 4 * 2560);
                            CK(hipSetDevice(dv));
                            enc[size_t(dv)]->encode(pixels.data() + p_off * 1536, h, w, o2.data());
                            double dd = 0, zz = 0;
                            for (size_t i = 0; i < o2.size(); ++i) {
                                const double a = o2[i], b = out[o_off * 2560 + i];
                                dd += (a - b) * (a - b);
                                zz += b * b;
                            }
                            // rocBLAS may pick split-K kernels whose summation order varies
                            const double rel = std::sqrt(dd / std::max(zz, 1e-30));
                            if (rel > 1e-3) {
                                std::printf("  card %d differs from card 0: rel. %.2e\n", dv, rel);
                                ++fails;
                            }
                        }
                    p_off += size_t(h) * w;
                    o_off += size_t(h) * w / 4;
                }
            const bool ok = worst < 2e-2 && o_off * 2560 == ref.size();
            fails += !ok;
            std::printf("%-14s %d slices, %zu tokens: rel. error %.2e, %.1f ms %s\n", name.c_str(), slices, o_off,
                        worst, ms, ok ? "" : "FAIL");
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
