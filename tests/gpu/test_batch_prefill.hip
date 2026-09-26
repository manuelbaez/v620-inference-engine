// Checks batched prefill (Engine::prefill_batch) on real weights against
// prefilling each sequence alone:
//   0. noise floor: each prompt alone, prefilled in two pieces;
//   1. up to four prompts in fresh slots in one pass: the logits after each
//      prompt (max |difference|, argmax), the greedy tokens decoded after it,
//      and the MTP drafts (the MTP pass over the batched rows);
//   2. the same prompts in two batched pieces (the second continuing slots
//      mid-sequence, a segment straddling the micro-batch split);
//   timing: the prompts one by one vs batched;
//   3. one prompt in several slots of one pass: the segments agree.
// Nothing here is bit-identical: GEMM results depend on the row count and a
// row's place in the GEMM at rounding level (~1e-3 in the logits after one
// layer), and 48 layers amplify that to ~1 (the same prompt alone, cut at
// another point, differs as much: case 0, the noise floor). Cases 1-3 pass
// when their max |dlogit| stays within twice case 0's largest (or TOL) and the
// first greedy token matches. To check the segment handling itself, run the
// first layers only (QW_NOGRAPH=1 DEBUG_LAYERS=8): case 3 then agrees to ~3e-3.
// Usage: test_batch_prefill --p 1,2,3 --p 4,5,6 [--p ...] [--gen 16]
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

struct Result {
    std::vector<std::vector<float>> logits;   // after each prompt
    std::vector<std::vector<int32_t>> greedy;  // decoded after each prompt, one slot at a time
    std::vector<std::vector<int32_t>> drafts;  // MTP drafts after each prompt
};

// Greedy continuation and MTP drafts of every slot, from its prompt logits.
static void finish(Engine &e, Result &res, int gen) {
    const int n = int(res.logits.size());
    res.greedy.assign(size_t(n), {});
    res.drafts.assign(size_t(n), {});
    for (int i = 0; i < n; ++i) res.greedy[size_t(i)].push_back(argmax(res.logits[size_t(i)].data()));
    if (e.has_mtp()) {
        std::vector<Engine::DraftReq> reqs;
        for (int i = 0; i < n; ++i) reqs.push_back({i, {res.greedy[size_t(i)][0]}});
        const auto &d = e.draft(reqs, 3);
        for (int i = 0; i < n; ++i) res.drafts[size_t(i)] = d[size_t(i)];
    }
    for (int i = 0; i < n; ++i)
        for (int g = 1; g < gen; ++g) {
            const auto &l = e.decode({{i, res.greedy[size_t(i)].back()}});
            res.greedy[size_t(i)].push_back(argmax(l.data()));
        }
}

int main(int argc, char **argv) {
    std::vector<std::vector<int32_t>> P;
    int gen = 16;
    EngineOptions opt;
    opt.slot_tokens = {32768, 32768, 32768, 32768};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--p")
            P.push_back(parse_ids(next()));
        else if (a == "--gen")
            gen = std::stoi(next());
        else if (a == "--no-warmup")
            opt.warmup = false;
    }
    try {
        QW_CHECK(P.size() >= 2 && P.size() <= 4, "give 2..4 prompts (--p ids)");
        size_t total = 0;
        for (auto &p : P) total += p.size();
        opt.prefill_chunk = std::max<int>(8192, int((total + 1023) / 1024 * 1024));
        Engine e(opt);
        if (const char *dl = std::getenv("DEBUG_LAYERS")) e.set_debug_layers(std::atoi(dl));  // (needs QW_NOGRAPH)
        const int n = int(P.size());
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        auto reset = [&] {
            for (int i = 0; i < n; ++i) e.slot_reset(i);
        };
        int fails = 0;
        float tol = std::getenv("TOL") ? float(std::atof(std::getenv("TOL"))) : 0.f;  // 0: from case 0
        float floor = 0;  // case 0's largest difference
        auto compare = [&](const char *what, const Result &ref, const Result &got, bool is_floor = false) {
            for (int i = 0; i < n; ++i) {
                const float d = maxdiff(ref.logits[size_t(i)].data(), got.logits[size_t(i)].data());
                const bool same_tok = ref.greedy[size_t(i)] == got.greedy[size_t(i)];
                const bool same_draft = ref.drafts[size_t(i)] == got.drafts[size_t(i)];
                if (is_floor) floor = std::max(floor, d);
                const float bound = tol > 0 ? tol : 2 * floor + 0.05f;
                const bool ok = is_floor || (d < bound && ref.greedy[size_t(i)][0] == got.greedy[size_t(i)][0]);
                std::printf("%s, prompt %d (%zu tokens): max |dlogit| %.4f, greedy %s, drafts %s  %s\n", what, i,
                            P[size_t(i)].size(), d, same_tok ? "SAME" : "DIFF", same_draft ? "SAME" : "DIFF",
                            ok ? "ok" : "FAIL");
                fails += !ok;
            }
        };

        // reference: one prompt at a time
        Result ref;
        reset();
        auto t0 = now();
        for (int i = 0; i < n; ++i) ref.logits.push_back(e.prefill(i, P[size_t(i)]));
        const double t_single = ms(t0, now());
        finish(e, ref, gen);

        // 0. noise floor: each prompt alone again, in two prefills (other row counts, other GEMM shapes)
        {
            Result got;
            reset();
            for (int i = 0; i < n; ++i) {
                const auto &p = P[size_t(i)];
                const size_t cut = std::max<size_t>(1, p.size() * 2 / 5);
                e.prefill(i, std::vector<int32_t>(p.begin(), p.begin() + ptrdiff_t(cut)));
                got.logits.push_back(e.prefill(i, std::vector<int32_t>(p.begin() + ptrdiff_t(cut), p.end())));
            }
            finish(e, got, gen);
            compare("alone, cut in two (noise floor)", ref, got, true);
        }
        // 1. all prompts in one pass
        {
            Result got;
            reset();
            std::vector<Engine::Segment> segs;
            for (int i = 0; i < n; ++i) segs.push_back({i, P[size_t(i)], {}});
            t0 = now();
            got.logits = e.prefill_batch(segs);
            const double t_batch = ms(t0, now());
            finish(e, got, gen);
            compare("one pass", ref, got);
            std::printf("timing: %d prompts (%zu tokens) one by one %.1f ms, batched %.1f ms\n", n, total, t_single,
                        t_batch);
        }
        // 2. two passes: the first ~40% of each prompt, then the rest
        {
            Result got;
            reset();
            std::vector<Engine::Segment> a, b;
            for (int i = 0; i < n; ++i) {
                const auto &p = P[size_t(i)];
                const size_t cut = std::max<size_t>(1, p.size() * 2 / 5);
                a.push_back({i, std::vector<int32_t>(p.begin(), p.begin() + ptrdiff_t(cut)), {}});
                b.push_back({i, std::vector<int32_t>(p.begin() + ptrdiff_t(cut), p.end()), {}});
            }
            e.prefill_batch(a);
            got.logits = e.prefill_batch(b);
            finish(e, got, gen);
            compare("two passes", ref, got);
        }
        // 3. the same prompt in several slots of one pass (3 copies: the middle one straddles the
        // micro-batch split): per-row arithmetic is shared, so the segments must agree to rounding
        for (int copies : {3, n}) {
            const auto &p = P[0];
            reset();
            std::vector<Engine::Segment> segs;
            for (int i = 0; i < copies; ++i) segs.push_back({i, p, {}});
            const auto logits = e.prefill_batch(segs);
            std::vector<std::vector<int32_t>> drafts(static_cast<size_t>(copies));
            if (e.has_mtp()) {
                std::vector<Engine::DraftReq> reqs;
                for (int i = 0; i < copies; ++i) reqs.push_back({i, {argmax(logits[0].data())}});
                const auto &d = e.draft(reqs, 3);
                for (int i = 0; i < copies; ++i) drafts[size_t(i)] = d[size_t(i)];
            }
            for (int i = 1; i < copies; ++i) {
                const float d = maxdiff(logits[0].data(), logits[size_t(i)].data());
                const bool ok = d < (tol > 0 ? tol : 2 * floor + 0.05f);
                std::printf("same prompt x%d, segment %d vs 0: max |dlogit| %.6f, drafts %s  %s\n", copies, i, d,
                            drafts[size_t(i)] == drafts[0] ? "SAME" : "DIFF", ok ? "ok" : "FAIL");
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
