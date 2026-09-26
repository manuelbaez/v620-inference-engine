// MTP speculative decoding check on real weights. Greedy speculative decoding
// must emit exactly the plain greedy tokens (the verification rows are
// bit-identical to single steps); reports the draft acceptance and speed.
//   1. each prompt alone: plain greedy vs speculative (K drafts)
//   2. all prompts at once in separate slots, speculative
// Usage: qw_spec_test --p 1,2,3 [--p 4,5,6 ...] [--gen 128] [--k 3]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"
#include "session/session.hpp"

using namespace qw;

static std::vector<int32_t> parse_ids(const std::string &s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(int32_t(std::stol(tok)));
    return v;
}

static int32_t argmax(const float *l) {
    return int32_t(std::max_element(l, l + cfg::VOCAB) - l);
}
// top-1 minus top-2 logit: how close a greedy pick was to a tie
static float margin(const float *l) {
    const int32_t a = argmax(l);
    float b = -INFINITY;
    for (int i = 0; i < cfg::VOCAB; ++i)
        if (i != a) b = std::max(b, l[i]);
    return l[a] - b;
}
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Seq {
    int slot;
    std::vector<int32_t> out;  // emitted tokens (first = from the prompt logits)
    std::vector<int32_t> drafts;
    int steps = 0, accepted = 0;
};

struct Timing {
    double decode = 0, accept = 0, draft = 0;
};
static Timing g_time;

// Speculative greedy decoding of every sequence (one slot each) until each
// has `gen` tokens.
static void run_spec(Engine &e, std::vector<Seq> &seqs, int gen, int K) {
    std::vector<Engine::DraftReq> reqs;
    for (auto &q : seqs) reqs.push_back({q.slot, {q.out.back()}});
    auto d = e.draft(reqs, K);
    for (size_t i = 0; i < seqs.size(); ++i) seqs[i].drafts = d[i];
    for (;;) {
        std::vector<Engine::Row> rows;
        std::vector<int> first(seqs.size(), -1), n_rows(seqs.size(), 0);
        for (size_t i = 0; i < seqs.size(); ++i) {
            Seq &q = seqs[i];
            if (int(q.out.size()) >= gen) continue;
            int k = std::min<int>(int(q.drafts.size()), gen - int(q.out.size()) - 1);
            if (std::getenv("QW_SPEC_MIXED")) k = std::min(k, int((q.steps * 7 + i * 3) % 4));  // mixed run lengths
            first[i] = int(rows.size());
            rows.push_back({q.slot, q.out.back()});
            for (int j = 0; j < k; ++j) rows.push_back({q.slot, q.drafts[size_t(j)]});
            n_rows[i] = 1 + k;
        }
        if (rows.empty()) break;
        double t = now();
        const auto &lg = e.decode(rows);
        g_time.decode += now() - t;
        reqs.clear();
        std::vector<size_t> which;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (first[i] < 0) continue;
            Seq &q = seqs[i];
            int keep = 0;
            std::vector<int32_t> next;
            for (int j = 0; j < n_rows[i]; ++j) {
                const int32_t t = argmax(lg.data() + size_t(first[i] + j) * cfg::VOCAB);
                q.out.push_back(t);
                next.push_back(t);
                ++keep;
                if (j + 1 < n_rows[i] && t != q.drafts[size_t(j)]) break;
            }
            q.steps += 1;
            q.accepted += keep - 1;
            if (std::getenv("QW_SPEC_LOG")) {
                std::printf("step %d len %zu drafts", q.steps, q.out.size());
                for (int j = 0; j + 1 < n_rows[i]; ++j) std::printf(" %d", q.drafts[size_t(j)]);
                std::printf(" | sampled");
                for (int32_t t : next) std::printf(" %d", t);
                std::printf("\n");
            }
            t = now();
            e.accept(q.slot, keep);
            g_time.accept += now() - t;
            if (int(q.out.size()) < gen) {
                reqs.push_back({q.slot, next});
                which.push_back(i);
            }
        }
        if (reqs.empty()) continue;
        t = now();
        auto dd = e.draft(reqs, K);
        g_time.draft += now() - t;
        for (size_t j = 0; j < which.size(); ++j) seqs[which[j]].drafts = dd[j];
    }
    for (auto &q : seqs) q.out.resize(size_t(gen));
}

int main(int argc, char **argv) {
    std::vector<std::vector<int32_t>> prompts;
    int gen = 128, K = 3, repeat = 1;  // repeat: the single-prompt comparisons, that many times
    bool session_only = false;
    EngineOptions opt;
    opt.slot_tokens = {32768, 32768, 32768, 32768};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--p")
            prompts.push_back(parse_ids(next()));
        else if (a == "--gen")
            gen = std::stoi(next());
        else if (a == "--k")
            K = std::stoi(next());
        else if (a == "--repeat")
            repeat = std::stoi(next());
        else if (a == "--session-only")
            session_only = true;
        else if (a == "--no-warmup")
            opt.warmup = false;
    }
    try {
        QW_CHECK(!prompts.empty() && prompts.size() <= 4, "1..4 prompts");
        Engine e(opt);
        QW_CHECK(e.has_mtp(), "no MTP head loaded");
        int fails = 0;

        std::vector<std::vector<int32_t>> ref;
        std::vector<std::vector<float>> ref_marg;  // top-2 margin of each plain pick
        if (!session_only) {
            for (size_t pr = 0; pr < prompts.size() * size_t(repeat); ++pr) {
                const size_t p = pr % prompts.size();
                e.slot_reset(0);
                std::vector<int32_t> out{argmax(e.prefill(0, prompts[p]).data())};
                std::vector<float> marg{0.f};  // margin of each plain pick
                const double t0 = now();
                while (int(out.size()) < gen) {
                    const auto &l = e.decode({{0, out.back()}});
                    out.push_back(argmax(l.data()));
                    marg.push_back(margin(l.data()));
                }
                const double t_plain = now() - t0;
                if (pr < prompts.size()) {
                    ref.push_back(out);
                    ref_marg.push_back(marg);
                }
                else if (out != ref[p]) {
                    size_t k = 0;
                    while (out[k] == ref[p][k]) ++k;
                    std::printf("prompt %zu: PLAIN decoding differs from its first run at %zu (margin %.4f)\n", p, k,
                                marg[k]);
                    ++fails;
                }

                e.slot_reset(0);
                std::vector<Seq> one{{0, {argmax(e.prefill(0, prompts[p]).data())}}};
                const double t1 = now();
                g_time = {};
                run_spec(e, one, gen, K);
                const double t_spec = now() - t1;
                std::printf("  per step: verify %.2f ms, accept %.2f ms, draft %.2f ms\n",
                            1e3 * g_time.decode / one[0].steps, 1e3 * g_time.accept / one[0].steps,
                            1e3 * g_time.draft / one[0].steps);
                const bool same = one[0].out == out;
                fails += !same;
                std::printf("prompt %zu: plain %.1f tok/s, spec %.1f tok/s, %.2f tokens/step (%d steps), %s\n", p,
                            (gen - 1) / t_plain, (gen - 1) / t_spec, double(gen - 1) / one[0].steps, one[0].steps,
                            same ? "SAME tokens" : "DIFFERENT tokens");
                if (!same) {
                    size_t k = 0;
                    while (k < out.size() && out[k] == one[0].out[k]) ++k;
                    std::printf("  first difference at %zu: plain %d, spec %d; plain's top-2 margin there %.4f\n", k,
                                out[k], one[0].out[k], marg[k]);
                }
            }
            if (prompts.size() > 1) {
                std::vector<Seq> seqs;
                for (size_t p = 0; p < prompts.size(); ++p) {
                    e.slot_reset(int(p));
                    seqs.push_back({int(p), {argmax(e.prefill(int(p), prompts[p]).data())}});
                }
                const double t0 = now();
                run_spec(e, seqs, gen, std::min(K, Engine::MAX_BATCH_ROWS / int(seqs.size()) - 1));  // <= 16 rows
                const double dt = now() - t0;
                bool same = true;
                for (size_t p = 0; p < prompts.size(); ++p) same = same && seqs[p].out == ref[p];
                fails += !same;
                std::printf("%zu prompts together: %.1f tok/s aggregate, %s\n", prompts.size(),
                            double(prompts.size() * size_t(gen - 1)) / dt, same ? "SAME tokens" : "DIFFERENT tokens");
            }
        }
        // Session::generate (the server's path): speculative (k = 3) vs plain (k = 0), all prompts at once
        {
            Session ses(e);
            SamplingParams greedy;
            greedy.temperature = 0.f;
            auto gen_all = [&](int k) {
                std::vector<int> slots;
                std::vector<std::vector<int32_t>> outs(prompts.size());
                std::vector<int32_t> pending(prompts.size());
                for (size_t p = 0; p < prompts.size(); ++p) {
                    const int sl = ses.acquire(prompts[p], gen);
                    ses.set_prompt(sl, prompts[p]);
                    pending[p] = ses.sample_prompt(sl, greedy);
                    outs[p].push_back(pending[p]);
                    slots.push_back(sl);
                }
                for (;;) {
                    std::vector<Session::StepReq> reqs;
                    std::vector<size_t> idx;
                    for (size_t p = 0; p < prompts.size(); ++p)
                        if (int(outs[p].size()) < gen) {
                            reqs.push_back({slots[p], pending[p], gen - int(outs[p].size()), greedy});
                            idx.push_back(p);
                        }
                    if (reqs.empty()) break;
                    const auto &res = ses.generate(reqs, k);
                    for (size_t j = 0; j < idx.size(); ++j) {
                        auto &o = outs[idx[j]];
                        o.insert(o.end(), res[j].tokens.begin(), res[j].tokens.end());
                        pending[idx[j]] = res[j].tokens.back();
                    }
                }
                for (int sl : slots) ses.release(sl);
                for (auto &o : outs) o.resize(size_t(gen));
                return outs;
            };
            for (int sl = 0; sl < e.num_slots(); ++sl) e.slot_reset(sl);
            const auto plain = gen_all(0);
            for (int sl = 0; sl < e.num_slots(); ++sl) e.slot_reset(sl);
            const auto spec = gen_all(K);
            bool same = true;
            for (size_t p = 0; p < prompts.size(); ++p) {
                const bool sp = plain[p] == spec[p] && (ref.empty() || plain[p] == ref[p]);
                if (!sp) {
                    size_t kk = 0;
                    while (kk < plain[p].size() && plain[p][kk] == spec[p][kk] &&
                           (ref.empty() || plain[p][kk] == ref[p][kk]))
                        ++kk;
                    std::printf("  session prompt %zu differs at %zu: ref %d plain %d spec %d\n", p, kk,
                                ref.empty() ? -1 : ref[p][kk], plain[p][kk], spec[p][kk]);
                }
                same = same && sp;
            }
            fails += !same;
            std::printf("session generate, %zu prompts at once: plain == spec == engine greedy: %s\n", prompts.size(),
                        same ? "SAME" : "DIFFERENT");
            // one prompt at a time through Session::generate (adaptive drafts, MTP off when it does not pay)
            for (size_t pr = 0; pr < prompts.size() * size_t(repeat); ++pr) {
                const size_t p = pr % prompts.size();
                auto one = [&](int k, double &tps) {
                    for (int sl = 0; sl < e.num_slots(); ++sl) e.slot_reset(sl);
                    Session s1(e);
                    const int sl = s1.acquire(prompts[p], gen);
                    s1.set_prompt(sl, prompts[p]);
                    int32_t pend = s1.sample_prompt(sl, greedy);
                    std::vector<int32_t> o{pend};
                    const auto t0 = std::chrono::steady_clock::now();
                    while (int(o.size()) < gen) {
                        const auto &res = s1.generate({{sl, pend, gen - int(o.size()), greedy}}, k);
                        o.insert(o.end(), res[0].tokens.begin(), res[0].tokens.end());
                        pend = o.back();
                    }
                    tps = double(o.size() - 1) /
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    o.resize(size_t(gen));
                    return o;
                };
                double tp = 0, ts = 0;
                const auto a = one(0, tp), b = one(K, ts);
                fails += a != b;
                std::printf("session, prompt %zu alone: plain %.1f tok/s, adaptive spec %.1f tok/s, %s\n", p, tp, ts,
                            a == b ? "SAME tokens" : "DIFFERENT");
                if (a != b) {
                    size_t k = 0;
                    while (a[k] == b[k]) ++k;
                    std::printf("  first difference at %zu: plain %d, spec %d%s; engine greedy %d, its top-2 margin %.4f\n",
                                k, a[k], b[k], ref.empty() ? "" : "", ref.empty() ? -1 : ref[p][k],
                                ref_marg.empty() ? NAN : double(ref_marg[p][k]));
                }
            }
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
