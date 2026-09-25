// Snapshots captured inside a prefill chunk: restoring one and prefilling the
// rest of the prompt must give the next-token distribution of prefilling the
// whole prompt. A capture at a chunk's end must equal snapshot_save there,
// bit for bit; a capture mid-chunk (in either micro-batch) must match a
// prefill split at that position to within the rounding noise of the
// different GEMM shapes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> text(int n, int seed) {  // deterministic token soup of common ids
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

static double logprob(const std::vector<float> &l, int32_t tok) {
    const float mx = *std::max_element(l.begin(), l.end());
    double z = 0;
    for (float x : l) z += std::exp(double(x - mx));
    return double(l[size_t(tok)] - mx) - std::log(z);
}

static int32_t argmax(const std::vector<float> &l) {
    return int32_t(std::max_element(l.begin(), l.end()) - l.begin());
}

int main() {
    try {
        EngineOptions opt;
        opt.slot_tokens = {8192};
        opt.warmup = false;
        Engine e(opt);
        const auto P = text(5000, 1), tail = text(64, 2);
        auto slice = [&](int64_t a, int64_t b) {
            return std::vector<int32_t>(P.begin() + ptrdiff_t(a), P.begin() + ptrdiff_t(b));
        };
        auto with_tail = [&](std::vector<int32_t> v) {
            v.insert(v.end(), tail.begin(), tail.end());
            return v;
        };
        int fails = 0;

        // 1. capture at the chunk's end vs snapshot_save: identical continuations
        e.slot_reset(0);
        e.prefill(0, P, nullptr, nullptr, {{int64_t(P.size()), 0}});
        e.snapshot_save(1, 0);
        e.snapshot_restore(0, 0, int64_t(P.size()), P);
        const auto l_cap = e.prefill(0, tail);
        e.snapshot_restore(1, 0, int64_t(P.size()), P);
        const auto l_save = e.prefill(0, tail);
        const bool same = l_cap == l_save;
        fails += !same;
        std::printf("capture at the chunk end vs snapshot_save: %s\n", same ? "identical" : "DIFFERENT");

        // 2. mid-chunk captures, one per micro-batch (halves [0, 2500) and [2500, 5000))
        const int64_t cuts[] = {1000, 3701};
        e.slot_reset(0);
        e.prefill(0, P, nullptr, nullptr, {{cuts[0], 2}, {cuts[1], 3}});
        for (int i = 0; i < 2; ++i) {
            const int64_t c = cuts[i];
            e.snapshot_restore(2 + i, 0, c, slice(0, c));
            const auto got = e.prefill(0, with_tail(slice(c, int64_t(P.size()))));
            e.slot_reset(0);  // reference: the prompt prefilled with a split at c
            e.prefill(0, slice(0, c));
            const auto ref = e.prefill(0, with_tail(slice(c, int64_t(P.size()))));
            e.slot_reset(0);  // noise floor: the whole prompt in one piece
            const auto whole = e.prefill(0, with_tail(P));
            const int32_t top = argmax(ref);
            const double d = std::fabs(logprob(ref, top) - logprob(got, top));
            const double noise = std::fabs(logprob(ref, top) - logprob(whole, top));
            const bool ok = argmax(got) == top && d < std::max(0.05, 3 * noise);
            fails += !ok;
            std::printf("capture at %lld: top %s, |dlogprob| vs split prefill %.4f (whole-prompt noise %.4f) %s\n",
                        (long long)c, argmax(got) == top ? "same" : "DIFFERENT", d, noise, ok ? "" : "FAIL");
            if (i == 0) {  // re-run the capturing prefill for the next cut (the slot was reused)
                e.slot_reset(0);
                e.prefill(0, P, nullptr, nullptr, {{cuts[1], 3}});
            }
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
