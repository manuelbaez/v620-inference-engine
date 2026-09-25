// Checks the batched decode path on real weights:
//   0. decoding a prompt token by token matches prefill (logprobs);
//   1. several sequences decoded together == each decoded alone (greedy tokens,
//      and max |logit difference|, expected 0: rows share no arithmetic);
//   2. k tokens of one sequence as one multi-row step (the speculative
//      verification shape) == the same tokens as k single steps;
//   3. timing: tokens/s for batch sizes 1..8.
// Usage: qw_batch_test --a 1,2,3 --b 4,5,6 [--gen 24]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> parse_ids(const std::string &s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(int32_t(std::stol(tok)));
    return v;
}

static int argmax(const float *l) {
    return int(std::max_element(l, l + cfg::VOCAB) - l);
}
static float maxdiff(const float *a, const float *b) {
    float m = 0;
    for (int i = 0; i < cfg::VOCAB; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char **argv) {
    std::string a_s, b_s;
    int gen = 24;
    EngineOptions opt;
    opt.slot_tokens = {32768, 32768, 32768, 32768};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--a")
            a_s = next();
        else if (a == "--b")
            b_s = next();
        else if (a == "--gen")
            gen = std::stoi(next());
        else if (a == "--no-warmup")
            opt.warmup = false;
    }
    try {
        const auto A = parse_ids(a_s), B = parse_ids(b_s);
        Engine e(opt);
        int fails = 0;

        // ---- 0. decode vs prefill: the prompt token by token must give the
        // prefill's distributions (different kernels, same math)
        {
            auto logprob = [](const float *l, int32_t tok) {
                const float mx = *std::max_element(l, l + cfg::VOCAB);
                double z = 0;
                for (int i = 0; i < cfg::VOCAB; ++i) z += std::exp(double(l[i] - mx));
                return double(l[tok] - mx) - std::log(z);
            };
            std::vector<float> all;
            e.slot_reset(0);
            e.prefill(0, A, &all);
            e.slot_reset(0);
            double sum = 0, mx = 0;
            for (size_t t = 0; t + 1 < A.size(); ++t) {
                const auto &d = e.decode({{0, A[t]}});
                const double diff =
                    std::fabs(logprob(all.data() + t * cfg::VOCAB, A[t + 1]) - logprob(d.data(), A[t + 1]));
                sum += diff;
                mx = std::max(mx, diff);
            }
            const double mean = sum / double(A.size() - 1);
            const bool ok = mean < 0.05 && mx < 0.3;  // fp16 paths differ ~0.01-0.03; a real bug is ~1 nat
            fails += !ok;
            std::printf("decode vs prefill on A: mean |dlogprob| %.4f, max %.4f: %s\n", mean, mx,
                        ok ? "ok" : "MISMATCH");
        }

        // ---- 1. alone vs together
        auto alone = [&](const std::vector<int32_t> &p, std::vector<int> &toks, std::vector<std::vector<float>> &lg) {
            e.slot_reset(0);
            auto l = e.prefill(0, p);
            int t = argmax(l.data());
            for (int i = 0; i < gen; ++i) {
                toks.push_back(t);
                const auto &d = e.decode({{0, t}});
                lg.emplace_back(d.begin(), d.begin() + cfg::VOCAB);
                t = argmax(d.data());
            }
        };
        std::vector<int> ta, tb;
        std::vector<std::vector<float>> la, lb;
        alone(A, ta, la);
        alone(B, tb, lb);
        std::printf("A alone:");
        for (int t : ta) std::printf(" %d", t);
        std::printf("\nB alone:");
        for (int t : tb) std::printf(" %d", t);
        std::printf("\n");

        for (int s = 0; s < 3; ++s) e.slot_reset(s);
        int t0 = argmax(e.prefill(0, A).data());
        int t1 = argmax(e.prefill(1, B).data());
        int t2 = argmax(e.prefill(2, A).data());
        std::vector<int> ga, gb, ga2;
        float worst = 0;
        for (int i = 0; i < gen; ++i) {
            ga.push_back(t0);
            gb.push_back(t1);
            ga2.push_back(t2);
            const auto &d = e.decode({{0, t0}, {1, t1}, {2, t2}});
            worst = std::max({worst, maxdiff(d.data(), la[size_t(i)].data()),
                              maxdiff(d.data() + cfg::VOCAB, lb[size_t(i)].data()),
                              maxdiff(d.data() + 2 * size_t(cfg::VOCAB), la[size_t(i)].data())});
            t0 = argmax(d.data());
            t1 = argmax(d.data() + cfg::VOCAB);
            t2 = argmax(d.data() + 2 * size_t(cfg::VOCAB));
        }
        const bool same = ga == ta && ga2 == ta && gb == tb;
        fails += !same;
        std::printf("batched (A, B, A) vs alone: tokens %s, max |logit diff| %.3g\n", same ? "SAME" : "DIFFERENT",
                    worst);

        // ---- 2. multi-row step vs single steps
        {
            e.slot_reset(0);
            e.prefill(0, A);
            std::vector<std::vector<float>> single;
            for (int i = 0; i < 4; ++i) {
                const auto &d = e.decode({{0, ta[size_t(i)]}});
                single.emplace_back(d.begin(), d.begin() + cfg::VOCAB);
            }
            e.slot_reset(0);
            e.prefill(0, A);
            const auto &d = e.decode({{0, ta[0]}, {0, ta[1]}, {0, ta[2]}, {0, ta[3]}});
            float w = 0;
            for (int i = 0; i < 4; ++i)
                w = std::max(w, maxdiff(d.data() + size_t(i) * cfg::VOCAB, single[size_t(i)].data()));
            // and the state after it continues identically
            const auto &n1 = e.decode({{0, ta[4]}});
            e.slot_reset(0);
            e.prefill(0, A);
            for (int i = 0; i < 4; ++i) e.decode({{0, ta[size_t(i)]}});
            std::vector<float> n1s = e.decode({{0, ta[4]}});
            const float w2 = maxdiff(n1.data(), n1s.data());
            std::printf("4-row step vs 4 single steps: max |logit diff| %.3g, next step after it %.3g\n", w, w2);
            fails += !(w < 0.05f && w2 < 0.05f);
        }

        // ---- 3. throughput by batch size
        for (int M : {1, 2, 4, 8}) {
            std::vector<Engine::Row> rows;
            for (int s = 0; s < std::min(M, e.num_slots()); ++s) {
                e.slot_reset(s);
                e.prefill(s, A);
            }
            // M rows over the slots: one row per slot when M <= slots
            const int S = std::min(M, e.num_slots());
            std::vector<int> cur(size_t(S), ta[0]);
            const int steps = 32;
            auto t_0 = std::chrono::steady_clock::now();
            for (int i = 0; i < steps; ++i) {
                rows.clear();
                for (int s = 0; s < S; ++s)
                    for (int k = 0; k < M / S; ++k) rows.push_back({s, cur[size_t(s)]});
                const auto &d = e.decode(rows);
                for (int s = 0; s < S; ++s) cur[size_t(s)] = argmax(d.data() + size_t(s * (M / S)) * cfg::VOCAB);
            }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_0).count();
            std::printf("M=%d (%d slots): %.2f ms/step, %.1f tok/s aggregate, %.1f tok/s per sequence\n", M, S,
                        dt / steps * 1e3, M * steps / dt, steps / dt);
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
