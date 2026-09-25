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
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

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

// Relative L2 differences of two snapshots' recurrent states at c tokens,
// per part; for the rings, only the cells the next token reads.
static std::vector<double> state_diff(Engine &e, int a, int b, int64_t c) {
    const size_t n = Engine::recurrent_rank_bytes() / 4;
    Engine::RankBufs ba{}, bb{};
    for (int r = 0; r < RANKS; ++r) {
        ba[size_t(r)] = pinned<uint8_t>(n * 4);
        bb[size_t(r)] = pinned<uint8_t>(n * 4);
    }
    e.export_recurrent(a, ba);
    e.export_recurrent(b, bb);
    e.host_copies_wait();
    // is float i of part k read by the token at position c?
    auto used = [&](int k, size_t i) {
        if (k == 1) {  // GDN ring [N_GDN][GDN_RING][GDN_QKV_L]: positions c-3 .. c-1
            const int64_t cell = int64_t(i / GDN_QKV_L % GDN_RING);
            for (int64_t q = c - (GDN_CONV - 1); q < c; ++q)
                if (q >= 0 && q % GDN_RING == cell) return true;
            return false;
        }
        if (k == 2) {  // PLE ring [PLE_RING][XW]: positions c - 3 * dilation .. c - 1
            const int64_t cell = int64_t(i / XW);
            for (int64_t q = c - (PLE_CONV - 1) * PLE_DILATION; q < c; ++q)
                if (q >= 0 && q % PLE_RING == cell) return true;
            return false;
        }
        return true;
    };
    const size_t parts[] = {SLOT_S, SLOT_RING, SLOT_PLE, size_t(XW)};
    std::vector<double> out;
    size_t off = 0;
    for (int k = 0; k < 4; ++k) {
        double d = 0, z = 0;
        for (int r = 0; r < RANKS; ++r) {
            const float *x = reinterpret_cast<const float *>(ba[size_t(r)]) + off;
            const float *y = reinterpret_cast<const float *>(bb[size_t(r)]) + off;
            for (size_t i = 0; i < parts[k]; ++i)
                if (used(k, i)) {
                    d += double(x[i] - y[i]) * (x[i] - y[i]);
                    z += double(y[i]) * y[i];
                }
        }
        out.push_back(std::sqrt(d / std::max(z, 1e-30)));
        off += parts[k];
    }
    for (int r = 0; r < RANKS; ++r) {
        (void)hipHostFree(ba[size_t(r)]);
        (void)hipHostFree(bb[size_t(r)]);
    }
    return out;
}

static int32_t argmax(const std::vector<float> &l) {
    return int32_t(std::max_element(l.begin(), l.end()) - l.begin());
}

int main() {
    try {
        EngineOptions opt;
        opt.slot_tokens = {8192, 8192};
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
            // direct comparison with the state after a prefill split at c; noise floor:
            // the same split, with the first part in two pieces (other GEMM shapes)
            e.slot_reset(1);
            e.prefill(1, slice(0, c));
            e.snapshot_save(4, 1);
            e.slot_reset(1);
            e.prefill(1, slice(0, c / 2));
            e.prefill(1, slice(c / 2, c));
            e.snapshot_save(5, 1);
            const auto d_cap = state_diff(e, 2 + i, 4, c), d_noise = state_diff(e, 5, 4, c);
            const char *names[] = {"GDN state", "GDN ring", "PLE ring", "MTP input"};
            for (int k = 0; k < 4; ++k) {
                const bool ok = d_cap[size_t(k)] < std::max(1e-4, 3 * d_noise[size_t(k)]);
                fails += !ok;
                std::printf("  capture at %lld, %-9s rel. diff vs split prefill %.2e (noise %.2e) %s\n", (long long)c,
                            names[k], d_cap[size_t(k)], d_noise[size_t(k)], ok ? "" : "FAIL");
            }
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
