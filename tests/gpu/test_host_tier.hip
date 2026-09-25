// Host-RAM tier of the prefix cache: with 2 slots, a third conversation
// evicts the first to host memory; coming back to the first (plus a new
// message) must reuse all of its tokens and give the same next-token
// distribution as a fresh prefill of the whole prompt.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

using namespace qw;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

int main() {
    try {
        EngineOptions opt;
        opt.slot_tokens = {8192, 8192};
        opt.warmup = false;
        Engine e(opt);
        Session s(e);
        const auto A = text(4000, 1), B = text(3000, 2), C = text(3000, 3);
        auto A2 = A;
        const auto tail = text(40, 4);
        A2.insert(A2.end(), tail.begin(), tail.end());
        SamplingParams greedy;
        greedy.temperature = 0.f;

        auto run = [&](const std::vector<int32_t> &p, const char *name) {
            const int slot = s.acquire(p, 16);
            QW_CHECK(slot >= 0, "no slot");
            const double t0 = now();
            const int64_t reused = s.set_prompt(slot, p);
            const double dt = now() - t0;
            const std::vector<float> l = e.logits();
            s.release(slot);
            std::printf("%-3s %5zu tokens in slot %d: reused %5lld, %.3f s\n", name, p.size(), slot, (long long)reused,
                        dt);
            return std::make_pair(reused, l);
        };
        const auto a1 = run(A, "A");
        run(B, "B");
        run(C, "C");  // evicts A's slot: A goes to host RAM
        const auto a_again = run(A, "A");
        int fails = 0;
        const bool exact = a_again.first == int64_t(A.size()) && a_again.second == a1.second;
        fails += !exact;
        std::printf("A again: %s\n", exact ? "fully reused, identical logits" : "MISMATCH");
        run(B, "B");
        run(C, "C");  // A's slot evicted again
        const auto a2 = run(A2, "A2");
        fails += a2.first != int64_t(A.size());
        // reference: fresh prefill of A2 in a slot
        e.slot_reset(0);
        const std::vector<float> ref = e.prefill(0, A2);
        const int32_t top = int32_t(std::max_element(ref.begin(), ref.end()) - ref.begin());
        const double d = std::fabs(logprob(ref, top) - logprob(a2.second, top));
        const bool same_top = top == int32_t(std::max_element(a2.second.begin(), a2.second.end()) - a2.second.begin());
        // baseline: the same split without the tier (prefill A, then the tail in place)
        e.slot_reset(0);
        e.prefill(0, A);
        const std::vector<float> split = e.prefill(0, tail);
        const double d_split = std::fabs(logprob(ref, top) - logprob(split, top));
        const double d_tier = std::fabs(logprob(split, top) - logprob(a2.second, top));
        fails += !(same_top && d_tier < 1e-3);
        std::printf("A2 vs fresh prefill: top token %s, |dlogprob| %.4f (in-place split: %.4f; tier vs split: %.5f)\n",
                    same_top ? "same" : "DIFFERENT", d, d_split, d_tier);
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
