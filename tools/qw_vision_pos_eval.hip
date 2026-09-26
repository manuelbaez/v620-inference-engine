// Multimodal RoPE experiment: image QA cases with known answers
// (tools/vision_pos_cases.py), scored teacher-forced under the engine's plain
// token positions (the vLLM fork's scheme) and HF's 3D positions for vision
// tokens. Reports, per task, the mean log-probability of the answer and how
// often every answer token is the argmax (greedy would produce the answer).
//   qw_vision_pos_eval /tmp/vpos/cases.txt
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/engine.hpp"

using namespace qw;

static std::vector<int32_t> ints(const std::string &line) {
    std::vector<int32_t> v;
    std::stringstream ss(line);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(int32_t(std::stol(t)));
    return v;
}

int main(int argc, char **argv) {
    try {
        QW_CHECK(argc >= 2, "usage: qw_vision_pos_eval cases.txt");
        std::ifstream f(argv[1]);
        QW_CHECK(f.good(), "cannot read cases");
        EngineOptions opt;
        opt.slot_tokens = {16384};
        Engine e(opt);
        QW_CHECK(e.has_vision(), "no vision tower");
        struct Acc {
            double lp[2] = {0, 0};
            int exact[2] = {0, 0}, n = 0;
        };
        std::map<std::string, Acc> by_task;
        std::string head, ids_l, pos_l;
        while (std::getline(f, head) && std::getline(f, ids_l) && std::getline(f, pos_l)) {
            std::stringstream hs(head);
            std::string name, pf;
            int ans0, t, h, w;
            int64_t img0;
            hs >> name >> ans0 >> t >> h >> w >> img0 >> pf;
            auto ids = ints(ids_l);
            auto pos3 = ints(pos_l);
            QW_CHECK(pos3.size() == ids.size() * 3, "positions size");
            std::ifstream pfs(pf, std::ios::binary | std::ios::ate);
            std::vector<float> patches(size_t(pfs.tellg()) / 4);
            pfs.seekg(0);
            pfs.read(reinterpret_cast<char *>(patches.data()), std::streamsize(patches.size() * 4));
            const int n_img = t * h * w / 4;
            std::vector<float> emb(size_t(n_img) * cfg::H);
            e.encode_vision({{patches.data(), h, w, emb.data()}});
            std::vector<int32_t> toks = ids;
            for (int j = 0; j < n_img; ++j) toks[size_t(img0 + j)] = -(3 + 2 * j);  // vision tokens (odd: image)
            const std::vector<Engine::EmbedSpan> spans{{img0, n_img, emb.data()}};
            Acc &a = by_task[name.substr(0, name.find('_'))];
            ++a.n;
            std::printf("%-10s", name.c_str());
            for (int scheme = 0; scheme < 2; ++scheme) {
                e.slot_reset(0);
                std::vector<float> all;
                e.prefill(0, toks, &all, nullptr, {}, spans, scheme ? &pos3 : nullptr);
                double lp = 0;
                bool exact = true;
                for (size_t i = size_t(ans0); i < ids.size(); ++i) {  // logits at i-1 predict token i
                    const float *l = all.data() + (i - 1) * cfg::VOCAB;
                    const float mx = *std::max_element(l, l + cfg::VOCAB);
                    double z = 0;
                    for (int v = 0; v < cfg::VOCAB; ++v) z += std::exp(double(l[v] - mx));
                    lp += double(l[ids[i]] - mx) - std::log(z);
                    exact = exact && int32_t(std::max_element(l, l + cfg::VOCAB) - l) == ids[i];
                }
                if (std::getenv("QW_SHOW"))  // the argmax token at each answer position
                    for (size_t i = size_t(ans0); i < ids.size(); ++i) {
                        const float *l = all.data() + (i - 1) * cfg::VOCAB;
                        std::printf(" [%d>%d]", ids[i], int(std::max_element(l, l + cfg::VOCAB) - l));
                    }
                a.lp[scheme] += lp;
                a.exact[scheme] += exact;
                std::printf("  %s lp %8.3f %s", scheme ? "3D" : "1D", lp, exact ? "exact" : "     ");
            }
            std::printf("\n");
            std::fflush(stdout);
        }
        std::printf("\ntask       n | plain positions: mean lp, exact | HF 3D positions: mean lp, exact\n");
        for (const auto &[task, a] : by_task)
            std::printf("%-9s %2d | %8.3f  %2d/%-2d          | %8.3f  %2d/%d\n", task.c_str(), a.n, a.lp[0] / a.n,
                        a.exact[0], a.n, a.lp[1] / a.n, a.exact[1], a.n);
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
