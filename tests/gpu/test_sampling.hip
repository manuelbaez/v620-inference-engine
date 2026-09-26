// GPU sampling (kernels/sampling.hpp + sample_candidates) against the CPU
// sampler (sample_token) on synthetic logits, the vocabulary split into the
// engine's 4 shards on one GPU:
//   1. penalized logits equal the CPU's (slot counts plus a run's tokens);
//   2. greedy and truncated sampling (top_k / top_p / min_p) draw the same
//      token as the CPU sampler for the same seed (the host draw is the same
//      code over the same candidates), or report a fallback;
//   3. untruncated sampling (Gumbel-max per shard): token frequencies over
//      many keys match the softmax probabilities.
// Usage: test_sampling
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "core/shard.hpp"
#include "engine/device_mem.hpp"
#include "kernels/sampling.hpp"
#include "session/sampling.hpp"

using namespace qw;
using namespace qw::cfg;

namespace {

// Synthetic logits: a few strong tokens over a Gaussian background of width sd.
std::vector<float> make_logits(std::mt19937_64 &g, float sd, int strong, float lead) {
    std::normal_distribution<float> n(0.f, sd);
    std::vector<float> l(static_cast<size_t>(VOCAB));
    for (auto &x : l) x = n(g);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int i = 0; i < strong; ++i) l[size_t(tok(g))] = lead - float(i) * 0.7f;
    return l;
}

struct Shards {
    float *raw, *L;
    gpu::SampleRow *rows;
    gpu::SampleOut *out;
    int32_t **tab;
    int32_t *counts[RANKS];
};

}  // namespace

int main() {
    try {
        CK(hipSetDevice(0));
        std::mt19937_64 g(7);
        int fails = 0;
        // rows: peaked, moderate, flat distributions
        struct Case {
            float sd;
            int strong;
            float lead;
        };
        const std::vector<Case> cases = {{1.f, 3, 18.f}, {2.f, 20, 14.f}, {1.5f, 50, 9.f}, {0.5f, 0, 0.f}};
        const int M = int(cases.size());
        std::vector<std::vector<float>> logits;
        for (const Case &c : cases) logits.push_back(make_logits(g, c.sd, c.strong, c.lead));

        // a history: prompt tokens, then generated ones (the strong tokens of row 0 among them)
        std::uniform_int_distribution<int> tok(0, VOCAB - 1);
        std::vector<int32_t> hist;
        for (int i = 0; i < 3000; ++i) hist.push_back(tok(g));
        const int64_t prompt_end = int64_t(hist.size());
        for (int i = 0; i < 400; ++i) hist.push_back(i % 3 ? tok(g) : int32_t(std::max_element(logits[0].begin(), logits[0].end()) - logits[0].begin()));
        const std::vector<int32_t> run = {tok(g), int32_t(std::max_element(logits[1].begin(), logits[1].end()) - logits[1].begin())};

        // device state per shard: counts, rows (all rows of slot 0; rows 2..3 a run of 2)
        Shards sh{};
        sh.raw = zeros<float>(size_t(M) * VOCAB_L);
        sh.L = zeros<float>(size_t(M) * VOCAB_L);
        sh.rows = zeros<gpu::SampleRow>(size_t(M));
        sh.out = zeros<gpu::SampleOut>(size_t(M));
        std::vector<int32_t> htok(hist.size());
        std::vector<uint8_t> hgen(hist.size());
        for (size_t i = 0; i < hist.size(); ++i) {
            htok[i] = hist[i];
            hgen[i] = int64_t(i) >= prompt_end;
        }
        int32_t *dtok = upload(htok);
        uint8_t *dgen = upload(hgen);
        std::vector<int32_t *> tabs;
        for (int r = 0; r < RANKS; ++r) {
            sh.counts[r] = zeros<int32_t>(size_t(2) * VOCAB_L);
            gpu::penalty_add(sh.counts[r], dtok, dgen, int(hist.size()), VOCAB_L, r * VOCAB_L, nullptr);
        }

        auto run_gpu = [&](int r, const std::vector<gpu::SampleRow> &rows, std::vector<gpu::SampleOut> &out,
                           std::vector<float> *L_out) {
            std::vector<float> raw(size_t(M) * VOCAB_L);
            for (int m = 0; m < M; ++m)
                std::copy(logits[size_t(m)].begin() + ptrdiff_t(r) * VOCAB_L,
                          logits[size_t(m)].begin() + ptrdiff_t(r + 1) * VOCAB_L, raw.begin() + ptrdiff_t(m) * VOCAB_L);
            CK(hipMemcpy(sh.raw, raw.data(), raw.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(sh.rows, rows.data(), rows.size() * sizeof(gpu::SampleRow), hipMemcpyHostToDevice));
            int32_t **tab = upload(std::vector<int32_t *>{sh.counts[r]});
            gpu::sample_penalize(sh.raw, sh.L, sh.rows, tab, VOCAB_L, r * VOCAB_L, M, nullptr);
            gpu::sample_select(sh.L, sh.raw, sh.rows, VOCAB_L, r * VOCAB_L, M, sh.out, nullptr);
            CK(hipDeviceSynchronize());
            CK(hipFree(tab));
            for (int m = 0; m < M; ++m) {
                CK(hipMemcpy(&out[size_t(m) * RANKS + size_t(r)], sh.out + m, sizeof(gpu::SampleOut),
                             hipMemcpyDeviceToHost));
                if (L_out)
                    CK(hipMemcpy(L_out->data() + size_t(m) * VOCAB + size_t(r) * VOCAB_L, sh.L + size_t(m) * VOCAB_L,
                                 VOCAB_L * 4, hipMemcpyDeviceToHost));
            }
        };
        auto spec_rows = [&](const SamplingParams &p, uint64_t key) {
            std::vector<gpu::SampleRow> rows(static_cast<size_t>(M));
            for (int m = 0; m < M; ++m) {
                gpu::SampleRow &s = rows[size_t(m)];
                s.inv_t = p.temperature > 0.f ? 1.f / p.temperature : 0.f;
                s.presence = p.presence_penalty;
                s.frequency = p.frequency_penalty;
                s.repetition = p.repetition_penalty;
                s.key = key * 1000003ull + uint64_t(m);
                s.slot = 0;
                s.first = m < 2 ? m : 2;  // rows 2..3: one request's run
                s.token = m < 2 ? tok(g) : run[size_t(m - 2)];
            }
            return rows;
        };
        // the CPU view of row m: history plus its run's tokens so far
        auto cpu_hist = [&](const std::vector<gpu::SampleRow> &rows, int m) {
            std::vector<int32_t> h = hist;
            for (int j = rows[size_t(m)].first; j <= m; ++j) h.push_back(rows[size_t(j)].token);
            return h;
        };

        // 1. penalties
        {
            SamplingParams p;
            p.temperature = 0.7f;
            p.presence_penalty = 0.5f;
            p.frequency_penalty = 0.3f;
            p.repetition_penalty = 1.1f;
            const auto rows = spec_rows(p, 1);
            std::vector<gpu::SampleOut> out(size_t(M) * RANKS);
            std::vector<float> L(size_t(M) * VOCAB);
            for (int r = 0; r < RANKS; ++r) run_gpu(r, rows, out, &L);
            for (int m = 0; m < M; ++m) {
                const auto h = cpu_hist(rows, m);
                std::vector<float> want = logits[size_t(m)];
                std::vector<int> all(static_cast<size_t>(VOCAB)), gen(static_cast<size_t>(VOCAB));
                for (size_t i = 0; i < h.size(); ++i) {
                    ++all[size_t(h[i])];
                    if (int64_t(i) >= prompt_end) ++gen[size_t(h[i])];
                }
                for (int i = 0; i < VOCAB; ++i) {
                    float &x = want[size_t(i)];
                    if (all[size_t(i)]) x = x > 0 ? x / p.repetition_penalty : x * p.repetition_penalty;
                    if (gen[size_t(i)]) x -= p.presence_penalty + p.frequency_penalty * float(gen[size_t(i)]);
                }
                int bad = 0;
                for (int i = 0; i < VOCAB; ++i) bad += want[size_t(i)] != L[size_t(m) * VOCAB + size_t(i)];
                std::printf("penalties, row %d: %d of %d logits differ from the CPU's  %s\n", m, bad, VOCAB,
                            bad ? "FAIL" : "ok");
                fails += bad != 0;
            }
        }

        // 2. greedy and truncated: same token as the CPU sampler for the same seed
        struct P {
            const char *name;
            float t, top_p, min_p;
            int top_k;
            float pres, rep;
        };
        const std::vector<P> ps = {{"greedy", 0.f, 1.f, 0.f, 0, 0.f, 1.f},
                                   {"greedy + penalties", 0.f, 1.f, 0.f, 0, 0.5f, 1.1f},
                                   {"top_p 0.95", 0.7f, 0.95f, 0.f, 0, 0.f, 1.f},
                                   {"top_k 20", 1.0f, 1.f, 0.f, 20, 0.f, 1.f},
                                   {"top_k 20 top_p 0.8 (Qwen)", 0.6f, 0.8f, 0.f, 20, 0.f, 1.f},
                                   {"min_p 0.05", 0.8f, 1.f, 0.05f, 0, 0.f, 1.f},
                                   {"top_p 0.9 + penalties", 0.9f, 0.9f, 0.f, 0, 1.0f, 1.05f}};
        for (const P &pp : ps) {
            SamplingParams p;
            p.temperature = pp.t;
            p.top_p = pp.top_p;
            p.top_k = pp.top_k;
            p.min_p = pp.min_p;
            p.presence_penalty = pp.pres;
            p.repetition_penalty = pp.rep;
            const auto rows = spec_rows(p, 2);
            std::vector<gpu::SampleOut> out(size_t(M) * RANKS);
            for (int r = 0; r < RANKS; ++r) run_gpu(r, rows, out, nullptr);
            for (int m = 0; m < M; ++m) {
                const auto h = cpu_hist(rows, m);
                int same = 0, diff = 0, fallback = 0;
                for (uint64_t seed = 1; seed <= 200; ++seed) {
                    std::mt19937_64 g1(seed), g2(seed);
                    float lp1 = 0, lp2 = 0;
                    const float lse = log_sum_exp(logits[size_t(m)].data());
                    const int32_t a = sample_token(logits[size_t(m)].data(), p, h, h.size(), prompt_end, g1, &lp1, lse);
                    const int32_t b = sample_candidates(out.data() + size_t(m) * RANKS, RANKS, p, h.size(), g2, &lp2, lse);
                    if (b < 0)
                        ++fallback;
                    else if (a == b && std::fabs(lp1 - lp2) < 1e-4f)
                        ++same;
                    else
                        ++diff;
                    if (pp.t <= 0.f) break;  // greedy: one draw
                }
                const bool ok = diff == 0;
                std::printf("%-26s row %d: same %d, different %d, fallback %d  %s\n", pp.name, m, same, diff, fallback,
                            ok ? "ok" : "FAIL");
                fails += !ok;
            }
        }

        // 3. untruncated (Gumbel-max): frequencies vs probabilities, rows 1 and 2
        {
            SamplingParams p;
            p.temperature = 0.8f;
            const int N = 3000;
            for (int m : {1, 2}) {
                std::vector<int> hits(static_cast<size_t>(VOCAB));
                for (int it = 0; it < N; ++it) {
                    const auto rows = spec_rows(p, uint64_t(1000 + it));
                    std::vector<gpu::SampleOut> out(size_t(M) * RANKS);
                    for (int r = 0; r < RANKS; ++r) run_gpu(r, rows, out, nullptr);
                    std::mt19937_64 gg(1);
                    const int32_t t = sample_candidates(out.data() + size_t(m) * RANKS, RANKS, p, 0, gg, nullptr, 0.f);
                    ++hits[size_t(t)];
                }
                // expected: softmax(l / T); chi-square over the 10 likeliest tokens and the rest
                const auto &l = logits[size_t(m)];
                const float mx = *std::max_element(l.begin(), l.end());
                std::vector<double> pr(static_cast<size_t>(VOCAB));
                double z = 0;
                for (int i = 0; i < VOCAB; ++i) z += pr[size_t(i)] = std::exp(double(l[size_t(i)] - mx) / p.temperature);
                std::vector<int> idx(static_cast<size_t>(VOCAB));
                for (int i = 0; i < VOCAB; ++i) idx[size_t(i)] = i;
                std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return pr[size_t(a)] > pr[size_t(b)]; });
                double chi = 0, rest_p = 1, rest_h = N;
                for (int k = 0; k < 10; ++k) {
                    const double e = N * pr[size_t(idx[size_t(k)])] / z;
                    const double o = hits[size_t(idx[size_t(k)])];
                    rest_p -= pr[size_t(idx[size_t(k)])] / z;
                    rest_h -= o;
                    if (e >= 5) chi += (o - e) * (o - e) / e;
                }
                if (rest_p * N >= 5) chi += (rest_h - rest_p * N) * (rest_h - rest_p * N) / (rest_p * N);
                const bool ok = chi < 29.6;  // chi-square, <= 11 degrees of freedom, p = 0.002
                std::printf("untruncated T=0.8, row %d: chi-square %.1f over the top 10 + rest (%d draws)  %s\n", m, chi,
                            N, ok ? "ok" : "FAIL");
                fails += !ok;
            }
        }
        std::printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
