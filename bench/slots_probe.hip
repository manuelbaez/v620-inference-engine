// Does a slot configuration load and run? Builds the engine with the given slots, prints each card's free VRAM, fills
// every slot with --fill tokens and decodes one step across all slots (the widest batch the configuration allows).
//   slots_probe --slots 262144,32768,... [--max 524288] [--fill 4096] [--chunk 4096] [--no-warmup]
// --max sets the capacity of every slot (the rest of it is KV spill in pinned host RAM); without it the slots hold
// what they hold in VRAM. QW_SPILL=0 forces spill off, QW_SPILL_NC=1, QW_SPILL_NUMA=0 as in the engine.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "core/common.hpp"
#include "engine/engine.hpp"

using namespace qw;

int main(int argc, char **argv) {
    try {
        EngineOptions opt;
        opt.slot_tokens.clear();
        int fill = 4096;
        opt.prefill_chunk = 4096;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--slots" && i + 1 < argc) {
                std::stringstream ss(argv[++i]);
                for (std::string t; std::getline(ss, t, ',');) opt.slot_tokens.push_back(std::atoi(t.c_str()));
            } else if (a == "--max" && i + 1 < argc) opt.slot_max_tokens = std::atoi(argv[++i]);
            else if (a == "--fill" && i + 1 < argc) fill = std::atoi(argv[++i]);
            else if (a == "--chunk" && i + 1 < argc) opt.prefill_chunk = std::atoi(argv[++i]);
            else if (a == "--no-warmup") opt.warmup = false;
        }
        QW_CHECK(!opt.slot_tokens.empty(), "slots_probe: --slots a,b,...");
        if (const char *d = std::getenv("QW_PLE_DIR")) opt.ple_dir = d;
        Engine e(opt);
        std::vector<int32_t> toks(static_cast<size_t>(fill));
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = int32_t(100 + (i * 7919) % 30000);
        std::fprintf(stderr, "engine constructed\n");
        for (int s = 0; s < e.num_slots(); ++s) {
            e.prefill(s, toks);
            std::fprintf(stderr, "slot %d prefilled\n", s);
        }
        std::vector<Engine::Row> rows;
        for (int s = 0; s < e.num_slots(); ++s) rows.push_back({s, 1000});
        double best = 1e9;
        for (int i = 0; i < 12; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            e.decode(rows);
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3);
        }
        std::printf("%d slots ran: prefilled %d tokens each, one decode step over all %d slots %.1f ms\n", e.num_slots(),
                    fill, e.num_slots(), best);
        for (int d = 0; d < 4; ++d) {  // what is left after the load, warmup (graphs, rocBLAS kernels) and the run
            size_t f = 0, t = 0;
            if (hipSetDevice(d) == hipSuccess && hipMemGetInfo(&f, &t) == hipSuccess)
                std::printf("  GPU %d: %.2f GiB of VRAM free now\n", d, double(f) / double(1ull << 30));
        }
        for (int s = 0; s < e.num_slots(); ++s)
            std::printf("  slot %d: %lld tokens in VRAM, capacity %lld\n", s, (long long)e.slot_vram_tokens(s),
                        (long long)e.slot_capacity(s));
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "slots_probe: %s\n", ex.what());
        return 1;
    }
}
