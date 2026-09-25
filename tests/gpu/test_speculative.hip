// MTP speculative decoding check on real weights. Greedy speculative decoding
// must emit exactly the plain greedy tokens (the verification rows are
// bit-identical to single steps); reports the draft acceptance and speed.
//   1. each prompt alone: plain greedy vs speculative (K drafts)
//   2. all prompts at once in separate slots, speculative
// Usage: qw_spec_test --p 1,2,3 [--p 4,5,6 ...] [--gen 128] [--k 3]
#include <algorithm>
#include <chrono>
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

static int32_t argmax(const float *l) {
    return int32_t(std::max_element(l, l + cfg::VOCAB) - l);
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
            const int k = std::min<int>(int(q.drafts.size()), gen - int(q.out.size()) - 1);
            first[i] = int(rows.size());
            rows.push_back({q.slot, q.out.back()});
            for (int j = 0; j < k; ++j) rows.push_back({q.slot, q.drafts[size_t(j)]});
            n_rows[i] = 1 + k;
        }
        if (rows.empty()) break;
        const auto &lg = e.decode(rows);
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
            e.accept(q.slot, keep);
            if (int(q.out.size()) < gen) {
                reqs.push_back({q.slot, next});
                which.push_back(i);
            }
        }
        if (reqs.empty()) continue;
        auto dd = e.draft(reqs, K);
        for (size_t j = 0; j < which.size(); ++j) seqs[which[j]].drafts = dd[j];
    }
    for (auto &q : seqs) q.out.resize(size_t(gen));
}

int main(int argc, char **argv) {
    std::vector<std::vector<int32_t>> prompts;
    int gen = 128, K = 3;
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
        else if (a == "--no-warmup")
            opt.warmup = false;
    }
    try {
        QW_CHECK(!prompts.empty() && prompts.size() <= 4, "1..4 prompts");
        Engine e(opt);
        QW_CHECK(e.has_mtp(), "no MTP head loaded");
        int fails = 0;
        std::vector<std::vector<int32_t>> ref;
        for (size_t p = 0; p < prompts.size(); ++p) {
            e.slot_reset(0);
            std::vector<int32_t> out{argmax(e.prefill(0, prompts[p]).data())};
            const double t0 = now();
            while (int(out.size()) < gen) out.push_back(argmax(e.decode({{0, out.back()}}).data()));
            const double t_plain = now() - t0;
            ref.push_back(out);

            e.slot_reset(0);
            std::vector<Seq> one{{0, {argmax(e.prefill(0, prompts[p]).data())}}};
            const double t1 = now();
            run_spec(e, one, gen, K);
            const double t_spec = now() - t1;
            const bool same = one[0].out == out;
            fails += !same;
            std::printf("prompt %zu: plain %.1f tok/s, spec %.1f tok/s, %.2f tokens/step (%d steps), %s\n", p,
                        (gen - 1) / t_plain, (gen - 1) / t_spec, double(gen - 1) / one[0].steps, one[0].steps,
                        same ? "SAME tokens" : "DIFFERENT tokens");
            if (!same) {
                size_t k = 0;
                while (k < out.size() && out[k] == one[0].out[k]) ++k;
                std::printf("  first difference at %zu: plain %d, spec %d\n", k, out[k], one[0].out[k]);
            }
        }
        if (prompts.size() > 1) {
            std::vector<Seq> seqs;
            for (size_t p = 0; p < prompts.size(); ++p) {
                e.slot_reset(int(p));
                seqs.push_back({int(p), {argmax(e.prefill(int(p), prompts[p]).data())}});
            }
            const double t0 = now();
            run_spec(e, seqs, gen, K);
            const double dt = now() - t0;
            bool same = true;
            for (size_t p = 0; p < prompts.size(); ++p) same = same && seqs[p].out == ref[p];
            fails += !same;
            std::printf("%zu prompts together: %.1f tok/s aggregate, %s\n", prompts.size(),
                        double(prompts.size() * size_t(gen - 1)) / dt, same ? "SAME tokens" : "DIFFERENT tokens");
        }
        std::printf("%s\n", fails ? "FAILED" : "all ok");
        return fails ? 1 : 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
