// CacheBlend quality experiment (docs/DESIGN.md). Each case is a chunk seen
// after one prefix (the source) and reused after another (the target):
//   source: [pre_src][chunk]          target: [pre_dst][chunk][query]
// The chunk's state is recorded in the source context (its GDN transfer and
// KV); the target prefills pre_dst and the chunk's first W tokens, splices
// the rest of the chunk, then prefills the query. Compared with a full
// prefill of the target: KL and top-1 agreement over the query's tokens, and
// how many of `gen` greedy tokens after the query agree. Baselines: the same
// splice without the GDN correction (the source's end state as is), and the
// target without the chunk at all (how much the chunk matters).
//
// Usage: qw_blend_eval --cases FILE [--w 0,32,128,512] [--gen 48]
// FILE: blank-line separated cases of 4 lines of comma-separated token ids:
// pre_src, pre_dst, chunk, query (a line may start with "# name").
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

using namespace qw;
using Tokens = std::vector<int32_t>;

struct Case {
    std::string name;
    Tokens pre_src, pre_dst, chunk, query;
};

static std::vector<Case> read_cases(const std::string &path) {
    std::ifstream f(path);
    QW_CHECK(f.good(), "cannot read " + path);
    std::vector<Case> out;
    std::vector<Tokens> lines;
    std::string name, line;
    auto flush = [&] {
        if (lines.size() == 4) out.push_back({name, lines[0], lines[1], lines[2], lines[3]});
        QW_CHECK(lines.empty() || lines.size() == 4, "a case needs 4 token lines");
        lines.clear();
        name.clear();
    };
    while (std::getline(f, line)) {
        if (line.empty()) {
            flush();
            continue;
        }
        if (line[0] == '#') {
            name = line.substr(1);
            continue;
        }
        Tokens t;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) t.push_back(int32_t(std::stol(tok)));
        lines.push_back(std::move(t));
    }
    flush();
    return out;
}

static Tokens cat(Tokens a, const Tokens &b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

static std::vector<double> log_softmax(const float *l) {
    std::vector<double> o(static_cast<size_t>(cfg::VOCAB));
    const float mx = *std::max_element(l, l + cfg::VOCAB);
    double z = 0;
    for (int i = 0; i < cfg::VOCAB; ++i) z += std::exp(double(l[i] - mx));
    const double lz = std::log(z);
    for (int i = 0; i < cfg::VOCAB; ++i) o[size_t(i)] = double(l[i] - mx) - lz;
    return o;
}

// Per-GDN-layer relative L2 difference of two VRAM snapshots' states (layers 0, 1, 2, 8, 17, 35).
static void compare_layers(Engine &e, int a, int b, const char *label) {
    const size_t n = Engine::recurrent_rank_bytes() / 4;
    Engine::RankBufs ba{}, bb{};
    for (int r = 0; r < RANKS; ++r) {
        ba[size_t(r)] = pinned<uint8_t>(n * 4);
        bb[size_t(r)] = pinned<uint8_t>(n * 4);
    }
    e.export_recurrent(a, ba);
    e.export_recurrent(b, bb);
    e.host_copies_wait();
    std::printf("  %-10s GDN state rel. diff vs full prefill, by layer:", label);
    for (int l : {0, 1, 2, 8, 17, 35}) {
        double d = 0, z = 0;
        for (int r = 0; r < RANKS; ++r) {
            const float *x = reinterpret_cast<const float *>(ba[size_t(r)]) + size_t(l) * GDN_STATE;
            const float *y = reinterpret_cast<const float *>(bb[size_t(r)]) + size_t(l) * GDN_STATE;
            for (size_t i = 0; i < GDN_STATE; ++i) {
                d += double(x[i] - y[i]) * (x[i] - y[i]);
                z += double(y[i]) * y[i];
            }
        }
        std::printf(" %d:%.3f", l, std::sqrt(d / std::max(z, 1e-30)));
    }
    std::printf("\n");
    for (int r = 0; r < RANKS; ++r) {
        (void)hipHostFree(ba[size_t(r)]);
        (void)hipHostFree(bb[size_t(r)]);
    }
}

struct Result {
    std::vector<float> logits;  // [query][VOCAB]
    Tokens gen;
};

// Query logits (teacher-forced) and greedy continuation from slot 0's current state.
static Result finish(Engine &e, const Tokens &query, int gen) {
    Result r;
    e.prefill(0, query, &r.logits);
    std::vector<float> last(r.logits.end() - cfg::VOCAB, r.logits.end());
    for (int i = 0; i < gen; ++i) {
        const int32_t t = int32_t(std::max_element(last.begin(), last.end()) - last.begin());
        r.gen.push_back(t);
        last = e.step(t);
    }
    return r;
}

struct Score {
    double kl = 0, top1 = 0;
    int gen_agree = 0;
};

static Score score(const Result &ref, const Result &got) {
    Score s;
    const size_t n = ref.logits.size() / size_t(cfg::VOCAB);
    for (size_t q = 0; q < n; ++q) {
        const auto p = log_softmax(ref.logits.data() + q * cfg::VOCAB);
        const auto g = log_softmax(got.logits.data() + q * cfg::VOCAB);
        double kl = 0;
        for (int i = 0; i < cfg::VOCAB; ++i) kl += std::exp(p[size_t(i)]) * (p[size_t(i)] - g[size_t(i)]);
        s.kl += kl / double(n);
        s.top1 += (std::max_element(p.begin(), p.end()) - p.begin() == std::max_element(g.begin(), g.end()) - g.begin())
                      ? 1.0 / double(n)
                      : 0.0;
    }
    while (s.gen_agree < int(ref.gen.size()) && ref.gen[size_t(s.gen_agree)] == got.gen[size_t(s.gen_agree)])
        ++s.gen_agree;
    return s;
}

int main(int argc, char **argv) {
    try {
        std::string path;
        std::vector<int> Ws{0, 32, 128, 512};
        int gen = 48;
        bool check = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--cases") path = argv[++i];
            if (a == "--gen") gen = std::atoi(argv[++i]);
            if (a == "--check") check = true;  // per-layer state errors instead of the query scores
            if (a == "--w") {
                Ws.clear();
                std::stringstream ss(argv[++i]);
                std::string w;
                while (std::getline(ss, w, ',')) Ws.push_back(std::stoi(w));
            }
        }
        const auto cases = read_cases(path);
        EngineOptions opt;
        opt.slot_tokens = {65536, 65536};
        Engine e(opt);
        std::printf("%-28s %6s %6s %6s | %-9s %8s %6s %7s\n", "case", "src", "dst", "chunk", "method", "KL", "top1",
                    "gen");
        for (const Case &c : cases) {
            const int64_t Ls = int64_t(c.pre_src.size()), Ld = int64_t(c.pre_dst.size());
            const int64_t Lc = int64_t(c.chunk.size());
            auto row = [&](const char *method, const Score &s) {
                std::printf("%-28s %6lld %6lld %6lld | %-9s %8.4f %6.3f %3d/%-3d\n", c.name.c_str(), (long long)Ls,
                            (long long)Ld, (long long)Lc, method, s.kl, s.top1, s.gen_agree, gen);
                std::fflush(stdout);
            };
            if (check) {
                e.slot_reset(0);
                e.prefill(0, cat(c.pre_dst, c.chunk));
                e.snapshot_save(3, 0);
                e.slot_reset(0);  // noise floor: the same, prefilled in other pieces
                e.prefill(0, c.pre_dst);
                e.prefill(0, c.chunk);
                e.snapshot_save(2, 0);
                std::printf("%s\n", c.name.c_str());
                compare_layers(e, 2, 3, "rechunked");
                for (int W : Ws) {
                    if (W >= Lc) continue;
                    e.slot_reset(1);
                    e.prefill(1, cat(c.pre_src, Tokens(c.chunk.begin(), c.chunk.begin() + W)));
                    e.snapshot_save(0, 1);
                    e.blend_transfer_begin();
                    e.prefill(1, Tokens(c.chunk.begin() + W, c.chunk.end()));
                    e.blend_transfer_end();
                    e.snapshot_save(1, 1);
                    for (int corrected = 1; corrected >= 0; --corrected) {
                        e.slot_reset(0);
                        e.prefill(0, cat(c.pre_dst, Tokens(c.chunk.begin(), c.chunk.begin() + W)));
                        e.blend_splice(0, 1, Ls + W, Ld + W, Lc - W);
                        e.blend_compose(0, 0, 1, Ls + Lc, Ld + Lc, cat(c.pre_dst, c.chunk), corrected);
                        e.snapshot_save(2, 0);
                        compare_layers(e, 2, 3, ((corrected ? "blend W=" : "naive W=") + std::to_string(W)).c_str());
                    }
                }
                continue;
            }
            // reference: the target prefilled in full
            e.slot_reset(0);
            e.prefill(0, cat(c.pre_dst, c.chunk));
            const Result ref = finish(e, c.query, gen);
            e.slot_reset(0);  // noise floor: the same prompt prefilled in other pieces
            e.prefill(0, c.pre_dst);
            e.prefill(0, c.chunk);
            row("rechunked", score(ref, finish(e, c.query, gen)));
            e.slot_reset(0);  // the chunk left out
            e.prefill(0, c.pre_dst);
            row("no chunk", score(ref, finish(e, c.query, gen)));
            for (int W : Ws) {
                if (W >= Lc) continue;
                // the chunk's record in the source context
                e.slot_reset(1);
                e.prefill(1, cat(c.pre_src, Tokens(c.chunk.begin(), c.chunk.begin() + W)));
                e.snapshot_save(0, 1);
                e.blend_transfer_begin();
                e.prefill(1, Tokens(c.chunk.begin() + W, c.chunk.end()));
                e.blend_transfer_end();
                e.snapshot_save(1, 1);
                const Tokens dst_prefix = cat(c.pre_dst, c.chunk);
                for (int corrected = 1; corrected >= 0; --corrected) {
                    e.slot_reset(0);
                    e.prefill(0, cat(c.pre_dst, Tokens(c.chunk.begin(), c.chunk.begin() + W)));
                    e.blend_splice(0, 1, Ls + W, Ld + W, Lc - W);
                    e.blend_compose(0, 0, 1, Ls + Lc, Ld + Lc, dst_prefix, corrected);
                    const std::string m = (corrected ? "blend W=" : "naive W=") + std::to_string(W);
                    row(m.c_str(), score(ref, finish(e, c.query, gen)));
                }
            }
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
