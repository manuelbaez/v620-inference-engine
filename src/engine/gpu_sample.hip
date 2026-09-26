// Sampling on the GPUs, host side: the penalties' per-slot token counts and
// reading back a row's full logits (the kernels run in the decode job,
// decode.hip; the draw from their results is session/sampling.cpp's).
#include <hip/hip_runtime.h>

#include <vector>

#include "core/common.hpp"
#include "core/config.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"
#include "kernels/sampling.hpp"

namespace qw {

void Engine::penalty_sync(int slot, uint64_t epoch, int64_t prompt_end, const std::vector<int32_t> &hist) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "penalty_sync: bad slot");
    pen_.resize(slots_.size());
    PenState &ps = pen_[size_t(slot)];
    if (ps.epoch != epoch || ps.prompt_end != prompt_end || ps.len > hist.size()) {
        ps.reset = true;
        ps.tok.clear();
        ps.gen.clear();
        ps.len = 0;
        ps.epoch = epoch;
        ps.prompt_end = prompt_end;
    }
    for (size_t i = ps.len; i < hist.size(); ++i) {
        ps.tok.push_back(cfg::real_token(hist[i]));
        ps.gen.push_back(int64_t(i) >= prompt_end ? 1 : 0);
    }
    ps.len = hist.size();
}

void Engine::apply_penalty_updates(Rank &rk) {
    hipStream_t s = rk.s;
    auto &b = rk.bt;
    for (size_t slot = 0; slot < pen_.size(); ++slot) {
        const PenState &ps = pen_[slot];
        int32_t *counts = rk.slots[slot].pen;
        if (ps.reset) CK(hipMemsetAsync(counts, 0, size_t(2) * VOCAB_L * 4, s));
        const size_t n = ps.tok.size();
        if (n == 0) continue;
        if (b.pen_cap < n) {
            CK(hipStreamSynchronize(s));
            if (b.pen_tok) CK(hipFree(b.pen_tok));
            if (b.pen_gen) CK(hipFree(b.pen_gen));
            b.pen_cap = std::max(n, size_t(4096));
            b.pen_tok = zeros<int32_t>(b.pen_cap);
            b.pen_gen = zeros<uint8_t>(b.pen_cap);
        }
        CK(hipMemcpyAsync(b.pen_tok, ps.tok.data(), n * 4, hipMemcpyHostToDevice, s));
        CK(hipMemcpyAsync(b.pen_gen, ps.gen.data(), n, hipMemcpyHostToDevice, s));
        gpu::penalty_add(counts, b.pen_tok, b.pen_gen, int(n), VOCAB_L, rk.r * VOCAB_L, s);
    }
}

void Engine::row_logits(int row, float *out) {
    QW_CHECK(dlogits_on_gpu_ && row >= 0 && row < djob_.M, "row_logits: no such row of the last decode on the GPUs");
    for (auto &rkp : ranks_) {
        Rank &rk = *rkp;
        CK(hipSetDevice(rk.dev));
        CK(hipMemcpy(out + size_t(rk.r) * VOCAB_L, rk.bt.logits + size_t(row) * VOCAB_L, VOCAB_L * 4,
                     hipMemcpyDeviceToHost));
    }
}

}  // namespace qw
