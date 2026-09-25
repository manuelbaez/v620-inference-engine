// Vision job: slices (an image, or one temporal slice of a video) spread over
// the cards, each encoded whole by that card's copy of the vision tower.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "core/common.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"

namespace qw {

void Engine::encode_vision(const std::vector<VisionSlice> &slices) {
    QW_CHECK(vision_, "encode_vision: the engine has no vision tower");
    if (slices.empty()) return;
    // longest first onto the least loaded card (cost ~ patches^2 for attention + patches for the rest)
    std::vector<size_t> order(slices.size());
    std::iota(order.begin(), order.end(), 0);
    auto cost = [&](size_t i) {
        const double L = double(slices[i].h) * slices[i].w;
        return L * (1.0 + L / 4096.0);
    };
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cost(a) > cost(b); });
    vjob_.assign(RANKS, {});
    std::vector<double> load(RANKS, 0.0);
    for (size_t i : order) {
        const int r = int(std::min_element(load.begin(), load.end()) - load.begin());
        vjob_[size_t(r)].push_back(slices[i]);
        load[size_t(r)] += cost(i);
    }
    job_ = Job::Vision;
    dispatch();
}

void Engine::run_vision_rank(Rank &rk) {
    for (const VisionSlice &s : vjob_[size_t(rk.r)]) {
        phase_[size_t(rk.r)] = 2;
        rk.vit->encode(s.patches, s.h, s.w, s.out);
    }
}

}  // namespace qw
