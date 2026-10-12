// QW_MOE_STATS: balance of the routed experts' work over the ranks in a prefill chunk (see Engine::moe_counts_).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

#include "core/common.hpp"
#include "engine/engine.hpp"
#include "kernels/moe_tiling.hpp"

namespace qw {

// After a prefill job: per micro-batch and layer, the tiles (what the expert kernels' time follows: tokens per
// expert rounded up to MT) and (token, expert) pairs each rank had, and how far the busiest rank is above the mean.
void Engine::collect_moe_balance() {
    const int nmb = prefill_mbs(pjob_.T);    // short chunks run as one micro-batch
    const int layers = cfg::N_LAYERS + 1;    // the MTP layer last
    if (moe_bal_.layer_max.empty()) moe_bal_.layer_max.assign(size_t(layers), 0.0), moe_bal_.layer_mean.assign(size_t(layers), 0.0);
    moe_expert_tiles_.resize(size_t(layers) * cfg::N_EXPERTS);
    for (int mb = 0; mb < nmb; ++mb)
        for (int l = 0; l < layers; ++l) {
            double tmax = 0, tsum = 0, pmax = 0, psum = 0;
            for (int r = 0; r < RANKS; ++r) {
                const int32_t *c = moe_counts_[size_t(r)] + (size_t(mb) * size_t(layers) + size_t(l)) * size_t(cfg::EXP_L);
                double tiles = 0, pairs = 0;
                for (int e = 0; e < cfg::EXP_L; ++e) {
                    tiles += (c[e] + gpu::MT - 1) / gpu::MT;
                    pairs += c[e];
                    moe_expert_tiles_[size_t(l) * cfg::N_EXPERTS + size_t(r * cfg::EXP_L + e)] += uint64_t((c[e] + gpu::MT - 1) / gpu::MT);
                }
                tmax = std::max(tmax, tiles);
                tsum += tiles;
                pmax = std::max(pmax, pairs);
                psum += pairs;
            }
            moe_bal_.max_tiles += tmax;
            moe_bal_.mean_tiles += tsum / RANKS;
            moe_bal_.max_pairs += pmax;
            moe_bal_.mean_pairs += psum / RANKS;
            moe_bal_.layer_max[size_t(l)] += tmax;
            moe_bal_.layer_mean[size_t(l)] += tsum / RANKS;
        }
    ++moe_bal_.chunks;
}

void Engine::log_moe_balance() {
    if (!moe_bal_.chunks) return;
    std::vector<std::pair<double, int>> worst;
    for (size_t l = 0; l < moe_bal_.layer_max.size(); ++l)
        if (moe_bal_.layer_mean[l] > 0) worst.push_back({moe_bal_.layer_max[l] / moe_bal_.layer_mean[l], int(l)});
    std::sort(worst.rbegin(), worst.rend());
    std::string w;
    for (size_t i = 0; i < worst.size() && i < 5; ++i) {
        char b[32];
        std::snprintf(b, sizeof b, " L%d %.2f", worst[i].second, worst[i].first);
        w += b;
    }
    log("moe balance over %d chunks: busiest rank / mean = %.3f in expert tiles (%.3f in token pairs); worst layers:%s",
        moe_bal_.chunks, moe_bal_.max_tiles / std::max(moe_bal_.mean_tiles, 1.0),
        moe_bal_.max_pairs / std::max(moe_bal_.mean_pairs, 1.0), w.c_str());
    moe_bal_ = MoeBalance{};
    if (const char *f = std::getenv("QW_MOE_COUNTS_FILE")) {
        std::ofstream out(f);
        for (size_t i = 0; i < moe_expert_tiles_.size(); ++i)
            out << moe_expert_tiles_[i] << ((i + 1) % cfg::N_EXPERTS ? ' ' : '\n');
    }
}

void Engine::load_expert_perm() {
    const char *f = std::getenv("QW_EXPERT_PERM");
    if (!f || !*f) return;
    std::ifstream in(f);
    QW_CHECK(bool(in), std::string("QW_EXPERT_PERM: cannot read ") + f);
    expert_perm_.resize(size_t(cfg::N_LAYERS) * cfg::N_EXPERTS);
    int moved = 0;
    for (int l = 0; l < cfg::N_LAYERS; ++l) {
        std::vector<bool> seen(cfg::N_EXPERTS, false);
        for (int i = 0; i < cfg::N_EXPERTS; ++i) {
            int32_t &e = expert_perm_[size_t(l) * cfg::N_EXPERTS + size_t(i)];
            QW_CHECK(bool(in >> e) && e >= 0 && e < cfg::N_EXPERTS && !seen[size_t(e)],
                     "QW_EXPERT_PERM: layer " + std::to_string(l) + " is not a permutation of the experts");
            seen[size_t(e)] = true;
            moved += e / cfg::EXP_L != i / cfg::EXP_L;
        }
    }
    log("expert placement from %s: %d experts on another card than their number's", f, moved);
}

}  // namespace qw
