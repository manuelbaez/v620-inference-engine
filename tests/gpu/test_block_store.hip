// Block store: prefixes shared across conversations. Conversation 1 is a
// system prompt, a message boundary and a message; conversation 2, in another
// slot, shares the system prompt only. It must restore the system prompt from
// the snapshot captured at the boundary (inside conversation 1's prefill
// chunk): bit-identical to restoring that capture in VRAM directly, and close
// to a prefill of the whole prompt. Then a RAM budget of about two
// conversations forces evictions while one conversation keeps coming back.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

using namespace qw;

constexpr int32_t BOUNDARY = 7;  // stands in for <|im_start|>; the token soup never contains it

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

static std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t> &b) {
    a.push_back(BOUNDARY);
    a.insert(a.end(), b.begin(), b.end());
    return a;
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
        opt.slot_tokens = {8192, 8192};
        opt.warmup = false;
        Engine e(opt);
        setenv("QW_HOST_CACHE_GB", "8", 1);
        unsetenv("QW_DISK_CACHE_DIR");
        int fails = 0;
        const auto sys = text(3000, 1);
        const auto c1 = cat(sys, text(2000, 2)), c2 = cat(sys, text(1500, 3));

        // reference for conversation 2: its whole prompt, prefilled with a split at the boundary
        e.slot_reset(1);
        e.prefill(1, sys);
        const std::vector<float> ref = e.prefill(1, std::vector<int32_t>(c2.begin() + ptrdiff_t(sys.size()), c2.end()));
        e.slot_reset(1);
        const std::vector<float> whole = e.prefill(1, c2);
        e.slot_reset(0);
        e.slot_reset(1);

        {
            Session s(e);
            s.set_boundary_token(BOUNDARY);
            auto run = [&](const std::vector<int32_t> &p, const char *name) {
                const int slot = s.acquire(p, 16);
                QW_CHECK(slot >= 0, "no slot");
                const double t0 = now();
                const int64_t reused = s.set_prompt(slot, p);
                const double dt = now() - t0;
                std::vector<float> l = e.logits();
                const Session::Timing tm = s.slot_timing(slot);
                std::printf("%-3s %5zu tokens in slot %d: reused %5lld, %.3f s (restore %.3f, prefill %.3f, saves %.3f, pin wait %.3f)\n",
                            name, p.size(), slot, (long long)reused, dt, tm.restore_s, tm.prefill_s, tm.save_s, tm.pin_wait_s);
                // the timings add up: the prefill call includes its saves and they their pin wait, and nothing
                // exceeds the wall time of the call
                QW_CHECK(tm.save_s <= tm.prefill_s + 1e-3 && tm.pin_wait_s <= tm.save_s + 1e-3, "slot timing: parts exceed their whole");
                QW_CHECK(tm.restore_s + tm.prefill_s <= dt + 1e-3, "slot timing: more than the call took");
                QW_CHECK(reused == int64_t(p.size()) || tm.prefill_s > 0, "slot timing: a prefill with no prefill time");
                return std::make_pair(reused, std::move(l));
            };
            const int slot1 = s.acquire(c1, 16);  // keep conversation 1's slot busy
            s.set_prompt(slot1, c1);
            const auto [reused, got] = run(c2, "c2");
            const int32_t top = argmax(ref);
            const double d = std::fabs(logprob(ref, top) - logprob(got, top));
            const double noise = std::fabs(logprob(ref, top) - logprob(whole, top));
            // the same capture restored in VRAM, without the store
            e.slot_reset(0);
            e.prefill(0, c1, nullptr, nullptr, {{int64_t(sys.size()), 0}});
            e.snapshot_restore(0, 0, int64_t(sys.size()), sys);
            const std::vector<float> direct =
                e.prefill(0, std::vector<int32_t>(c2.begin() + ptrdiff_t(sys.size()), c2.end()));
            const bool exact = got == direct;
            const bool ok = reused == int64_t(sys.size()) && exact && argmax(got) == top && d < 0.25;
            fails += !ok;
            std::printf("c2: reused %lld of the %zu shared tokens; %s a VRAM restore of the capture; |dlogprob| vs "
                        "split prefill %.4f (whole-prompt noise %.4f) %s\n",
                        (long long)reused, sys.size(), exact ? "identical to" : "DIFFERENT from", d, noise,
                        ok ? "" : "FAIL");
            const auto st = s.cache_stats();
            std::printf("store: %zu blocks, %zu snapshots, %.2f GB RAM, %llu hits\n", st.blocks, st.snapshots,
                        double(st.ram_bytes) / 1e9, (unsigned long long)st.hits);
        }

        {  // ~380 MB per conversation: room for about two
            setenv("QW_HOST_CACHE_GB", "0.9", 1);
            Session s(e);
            s.set_boundary_token(BOUNDARY);
            std::vector<std::vector<int32_t>> convs;
            for (int i = 0; i < 4; ++i) convs.push_back(cat(text(1200 + 100 * i, 10 + i), text(300, 20 + i)));
            bool seen0 = false;
            for (int i : {0, 1, 0, 2, 0, 3, 0}) {
                    const auto &p = convs[size_t(i)];
                    const int slot = s.acquire(p, 16);
                    const int64_t reused = s.set_prompt(slot, p);
                    const std::vector<float> got = e.logits();
                    s.release(slot);
                    e.slot_reset(1 - slot);  // a fresh prefill elsewhere as the reference
                    const std::vector<float> fresh = e.prefill(1 - slot, p);
                    const int32_t top = argmax(fresh);
                    const double d = std::fabs(logprob(fresh, top) - logprob(got, top));
                    const bool again = i == 0 && seen0;
                    seen0 = seen0 || i == 0;
                    const bool ok = argmax(got) == top && d < 0.25 && (!again || reused == int64_t(p.size()));
                    fails += !ok;
                    const auto st = s.cache_stats();
                    std::printf("conversation %d, %zu tokens: reused %5lld, |dlogprob| %.4f; store %zu blocks, %zu "
                                "snapshots, %.2f GB %s\n",
                                i, p.size(), (long long)reused, d, st.blocks, st.snapshots,
                                double(st.ram_bytes) / 1e9, ok ? "" : "FAIL");
                    // the reference prefill clobbered the other slot: the session must not trust it
                    s.set_prompt(1 - slot, {1});
                }
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
