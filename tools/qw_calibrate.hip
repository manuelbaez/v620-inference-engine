// GPTQ calibration: prefills the calibration sequences (one per line, comma-
// separated token ids) through the engine and writes the Hessians X^T X of
// the inputs of every int8-able decode matrix, per rank (Engine::calib_begin).
// Layers go in groups so the Hessians fit next to the model.
//   qw_calibrate SEQS OUTDIR
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"

using namespace qw;

int main(int argc, char **argv) {
    try {
        QW_CHECK(argc >= 3, "usage: qw_calibrate SEQS OUTDIR");
        setenv("QW_NO_SPLIT", "1", 1);   // one stream per rank: the Hessians accumulate in order
        setenv("QW_NO_VISION", "1", 1);  // room for the Hessians
        std::vector<std::vector<int32_t>> seqs;
        std::ifstream f(argv[1]);
        std::string line;
        size_t total = 0;
        while (std::getline(f, line)) {
            std::vector<int32_t> v;
            std::stringstream ss(line);
            std::string t;
            while (std::getline(ss, t, ',')) v.push_back(int32_t(std::stol(t)));
            if (!v.empty()) {
                total += v.size();
                seqs.push_back(std::move(v));
            }
        }
        std::printf("%zu sequences, %zu tokens\n", seqs.size(), total);
        EngineOptions opt;
        opt.slot_tokens = {32768};
        opt.prefill_chunk = 2048;  // all-logits buffers stay small
        opt.warmup = false;
        Engine e(opt);
        const int groups[][2] = {{0, 12}, {12, 24}, {24, 36}, {36, cfg::N_LAYERS + 1}};
        for (const auto &g : groups) {
            e.calib_begin(g[0], g[1]);
            const bool final = g[1] > cfg::N_LAYERS;  // the final mixer runs on every row only in all-logits mode
            for (const auto &s : seqs) {
                e.slot_reset(0);
                std::vector<float> all;
                e.prefill(0, s, final ? &all : nullptr);
            }
            e.calib_dump(argv[2]);
            std::printf("layers [%d, %d) done\n", g[0], g[1]);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
