// QW_F4_PACKED (kernels/moe.hpp F4Packed): NVFP4 expert scale words packed as a base per row, 4-bit offsets and an
// escape table must give every code back. CPU only; the kernels are checked end to end (the engine's logits with
// QW_F4_PACKED=1 and =0 are identical).
#include <cstdio>
#include <random>
#include <vector>

#include "engine/weight_convert.hpp"

using namespace qw;

int main() {
    std::mt19937 rng(5);
    int fails = 0;
    for (const auto [rows, K32] : {std::pair<size_t, size_t>{640 * 4, 80}, {2560 * 4, 20}, {64, 128}}) {
        // codes like the checkpoint's: a row's cluster 6-12 codes wide, ~1% far away, some rows all far apart
        std::vector<uint16_t> words(rows * K32);
        std::uniform_int_distribution<int> centre(20, 120), spread(6, 12), any(0, 254), pct(0, 999);
        for (size_t r = 0; r < rows; ++r) {
            const int c = centre(rng), w = spread(rng);
            const bool wild = pct(rng) < 5;
            std::uniform_int_distribution<int> near(c, c + w);
            for (size_t j = 0; j < 2 * K32; ++j) {
                const int code = wild || pct(rng) < 10 ? any(rng) : near(rng);
                uint16_t &wd = words[r * K32 + j / 2];
                wd = uint16_t(j % 2 ? (wd & 0xff) | (code << 8) : (wd & 0xff00) | code);
            }
        }
        const F4PackedHost p = pack_f4_scales(words, rows, K32);
        size_t bad = 0;
        for (size_t r = 0; r < rows; ++r)
            for (size_t j = 0; j < 2 * K32; ++j) {
                const int want = j % 2 ? words[r * K32 + j / 2] >> 8 : words[r * K32 + j / 2] & 0xff;
                bad += f4_packed_code(p, K32, uint32_t(r), uint32_t(j)) != want;
            }
        std::printf("%s %zu rows x %zu scales: %zu escapes (%.2f%%), table %zu slots, %zu codes wrong\n",
                    bad ? "FAIL" : "ok  ", rows, 2 * K32, p.escapes, 100.0 * double(p.escapes) / double(rows * 2 * K32),
                    p.key.size(), bad);
        fails += bad != 0;
    }
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
