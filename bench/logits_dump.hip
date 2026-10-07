// Writes every logit the engine produces along a fixed script, so two builds can be compared bit for bit (a change
// that must not move any output: cmp the two files). The script, on a resident slot and, where
// the tree has KV spill, on a slot spilled at 8192:
//   a 13,001-token prompt (4,096-token chunks, past the 2,048 tokens where QSA starts selecting), 48 greedy decode
//   steps, 4 verification-shaped runs of 4 rows each rolled back to 2 (accept), a snapshot at a length that is not a
//   multiple of 4, 6,000 more tokens and 8 steps, then the snapshot restored and 32 steps from it.
//   logits_dump OUT_FILE
// With --deep N the script is instead N tokens prefilled into one resident slot in pieces of 32,768 (the logits after
// each piece) and 16 greedy steps: for changes that only act at long context (the sharded indexer selection).
//   logits_dump OUT_FILE --deep N
// Prints one line per phase with the argmax tokens; the file holds the rows' fp32 logits in order.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> text(int n, int seed) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    uint32_t x = uint32_t(seed) * 2654435761u + 1;
    for (auto &t : v) {
        x = x * 1664525u + 1013904223u;
        t = int32_t(100 + (x >> 8) % 30000);
    }
    return v;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: logits_dump OUT_FILE [--deep N]\n");
        return 2;
    }
    try {
        const int deep = argc >= 4 && std::string(argv[2]) == "--deep" ? std::atoi(argv[3]) : 0;
        EngineOptions opt;
        opt.prefill_chunk = 4096;
        opt.slot_tokens = {deep ? (deep + 4096 + 255) / 256 * 256 : 32768};
#ifdef QW_HAVE_SPILL
        if (!deep) {
            opt.slot_tokens.push_back(8192);
            opt.slot_spill = {0, 24576};
        }
#endif
        Engine e(opt);
        FILE *f = std::fopen(argv[1], "wb");
        QW_CHECK(f, "cannot write the output file");
        size_t rows = 0;
        const auto put = [&](const float *l, int n) {
            std::fwrite(l, sizeof(float), size_t(n) * cfg::VOCAB, f);
            rows += size_t(n);
        };
        const auto argmax = [](const float *l) { return int(std::max_element(l, l + cfg::VOCAB) - l); };
        if (deep) {
            const std::vector<int32_t> all = text(deep, 3);
            e.slot_reset(0);
            int tok = 0;
            for (int at = 0; at < deep; at += 32768) {
                const int n = std::min(32768, deep - at);
                const auto &l = e.prefill(0, std::vector<int32_t>(all.begin() + at, all.begin() + at + n));
                put(l.data(), 1);
                tok = argmax(l.data());
                std::printf("%d tokens -> %d\n", at + n, tok);
            }
            std::string toks;
            for (int i = 0; i < 16; ++i) {
                const auto &d = e.decode({{0, tok}});
                put(d.data(), 1);
                tok = argmax(d.data());
                toks += " " + std::to_string(tok);
            }
            std::printf("decode:%s\n", toks.c_str());
        }
        for (int slot = 0; !deep && slot < e.num_slots(); ++slot) {
            e.slot_reset(slot);
            const auto &l0 = e.prefill(slot, text(13001, 1));
            put(l0.data(), 1);
            int tok = argmax(l0.data());
            std::printf("slot %d prefill -> %d\n", slot, tok);
            std::string toks;
            for (int i = 0; i < 48; ++i) {
                const auto &d = e.decode({{slot, tok}});
                put(d.data(), 1);
                tok = argmax(d.data());
                toks += " " + std::to_string(tok);
            }
            std::printf("slot %d decode:%s\n", slot, toks.c_str());
            for (int i = 0; i < 4; ++i) {
                const auto &d = e.decode({{slot, tok}, {slot, tok + 1}, {slot, tok + 2}, {slot, tok + 3}});
                put(d.data(), 4);
                tok = argmax(d.data() + cfg::VOCAB);
                e.accept(slot, 2);
            }
            const int64_t n0 = e.slot_len(slot);
            e.snapshot_save(0, slot);
            const int tok0 = tok;
            const auto &l1 = e.prefill(slot, text(6000, 2));
            put(l1.data(), 1);
            tok = argmax(l1.data());
            for (int i = 0; i < 8; ++i) {
                const auto &d = e.decode({{slot, tok}});
                put(d.data(), 1);
                tok = argmax(d.data());
            }
            e.snapshot_restore(0, slot, n0, {tok0, tok0});
            tok = tok0;
            toks.clear();
            for (int i = 0; i < 32; ++i) {
                const auto &d = e.decode({{slot, tok}});
                put(d.data(), 1);
                tok = argmax(d.data());
                toks += " " + std::to_string(tok);
            }
            std::printf("slot %d after restoring at %lld (mod 4 = %lld):%s\n", slot, (long long)n0,
                        (long long)(n0 % 4), toks.c_str());
        }
        std::fclose(f);
        std::printf("%zu rows written\n", rows);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "logits_dump: %s\n", ex.what());
        return 1;
    }
}
