// Engine lifecycle: loading, the per-rank host threads and job dispatch,
// warmup, and slot bookkeeping.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"
#include "engine/weight_convert.hpp"

namespace qw {

Engine::Engine(const EngineOptions &opt) : opt_(opt) {
    // Copies as compute-shader blits, not SDMA: on this box an SDMA copy's
    // completion signal occasionally never fires (PCIe AER events), which
    // wedges that rank forever (home-infra docs/llm-backend-amd-tuning.md,
    // "SDMA code-object wedge"). Must be set before the HIP runtime starts;
    // an explicit HSA_ENABLE_SDMA in the environment wins.
    setenv("HSA_ENABLE_SDMA", "0", 0);
    check_config(opt_.model_dir);
    st_.add_index(opt_.model_dir);
    ple_ = std::make_unique<PleTable>(opt_.ple_dir);
    ple_->pin_in_background();
    const size_t slot = std::max<size_t>(64 * 1024, size_t(opt_.prefill_chunk / 2) * gpu::D_PAD * 4 + 4096);
    comm_ = std::make_unique<Comm>(opt_.devices, slot);
    comm2_ = std::make_unique<Comm>(opt_.devices, slot);
    QW_CHECK(!opt_.slot_tokens.empty() && opt_.slot_tokens.size() <= size_t(MAX_SLOTS), "1..8 slots");
    for (int c : opt_.slot_tokens) QW_CHECK(c > 0 && c % 256 == 0, "slot sizes must be positive multiples of 256");
    slots_.resize(opt_.slot_tokens.size());
    logits_.resize(VOCAB);
    mtp_ = opt_.mtp && !std::getenv("QW_NO_MTP") && st_.has("mtp.fc_embedding.weight");

    ThreadPool pool(opt_.load_threads * RANKS);
    for (int r = 0; r < RANKS; ++r) ranks_.push_back(std::make_unique<Rank>());
    // Ranks load one after another (each already uses the whole pool for its
    // experts); a rank's uploads overlap the next rank's conversions poorly,
    // but loading is not the bottleneck yet.
    for (int r = 0; r < RANKS; ++r) {
        Rank &rk = *ranks_[size_t(r)];
        rk.r = r;
        rk.dev = opt_.devices[size_t(r)];
        CK(hipSetDevice(rk.dev));
        CK(hipStreamCreateWithFlags(&rk.s, hipStreamNonBlocking));
        load_rank_weights(rk, st_, pool, mtp_);
        alloc_rank_buffers(rk, opt_, max_slot_tokens(), mtp_);
        log("rank %d loaded", r);
    }
    if (saturated_weights)
        log("warning: %lld bf16 weights saturated to fp16 range", (long long)saturated_weights.load());
    for (int r = 0; r < RANKS; ++r) threads_.emplace_back([this, r] { rank_loop(r); });
    if (opt_.warmup) warmup();
}

// rocBLAS loads Tensile kernels lazily per shape (the first real chunk was
// ~40% slower without this), and every decode / MTP row count has its own
// graph, captured on first use: run them all once.
void Engine::warmup() {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> toks(size_t(opt_.prefill_chunk) + 300, 0);
    for (size_t i = 0; i < toks.size(); ++i) toks[i] = int32_t(i % 1000 + 100);
    prefill(0, toks);
    std::vector<Row> rows;
    for (int m = 1; m <= gpu::MAX_ROWS; ++m) {  // capture every batch size's graph
        rows.clear();
        for (int i = 0; i < m; ++i) rows.push_back({i % num_slots(), int32_t(100 + i)});
        std::stable_sort(rows.begin(), rows.end(), [](const Row &x, const Row &y) { return x.slot < y.slot; });
        decode(rows);
        if (mtp_ && m > 1) {  // one run of m rows: speculative verification (state-saving graph)
            rows.assign(size_t(m), Row{0, 100});
            decode(rows);
            accept(0, 1);
        }
    }
    if (mtp_)
        for (int m = 1; m <= gpu::MAX_ROWS; ++m) {  // MTP graphs: m rows over the slots
            std::vector<DraftReq> reqs;
            for (int i = 0; i < m; ++i) {
                const int sl = i % num_slots();
                if (int(reqs.size()) <= sl) reqs.push_back({sl, {}});
                reqs[size_t(sl)].next.push_back(int32_t(100 + i));
            }
            draft(reqs, 1);
        }
    for (int sl = 0; sl < num_slots(); ++sl) slot_reset(sl);
    log("warmup done in %.1f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        ++gen_;
    }
    cv_.notify_all();
    for (auto &t : threads_) t.join();
}

void Engine::rank_loop(int r) {
    Rank &rk = *ranks_[size_t(r)];
    CK(hipSetDevice(rk.dev));
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return gen_ != seen; });
            seen = gen_;
            if (stop_) return;
        }
        std::string err;
        phase_[size_t(r)] = 1;
        try {
            if (job_ == Job::Prefill)
                run_prefill_rank(rk);
            else if (job_ == Job::Mtp)
                run_mtp_rank(rk);
            else
                run_decode_rank(rk);
        } catch (const std::exception &e) {
            err = e.what();
        }
        phase_[size_t(r)] = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!err.empty() && error_.empty()) error_ = err;
            ++done_;
        }
        done_cv_.notify_all();
    }
}

void Engine::dispatch() {
    static const bool trace = std::getenv("QW_TRACE") != nullptr;
    if (trace) {
        if (job_ == Job::Prefill)
            log("trace: prefill slot %d [%lld, +%d)%s", pjob_.slot, (long long)pjob_.start, pjob_.T,
                pjob_.mtp_pend ? " mtp_pend" : "");
        else
            log("trace: %s M=%d%s", job_ == Job::Mtp ? "mtp" : "decode", djob_.M, djob_.save ? " save" : "");
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = 0;
        ++gen_;
    }
    cv_.notify_all();
    {
        // Watchdog: a job never takes minutes; report what is stuck (and keep waiting).
        std::unique_lock<std::mutex> lk(mu_);
        int waited = 0;
        while (!done_cv_.wait_for(lk, std::chrono::seconds(60), [&] { return done_ == RANKS; })) {
            waited += 60;
            const char *kind = job_ == Job::Prefill ? "prefill" : job_ == Job::Mtp ? "mtp" : "decode";
            log("watchdog: %s job (M=%d, prefill T=%d) running for %d s; rank phases %d %d %d %d "
                "(1 enqueueing, 2 waiting for GPU, 0 done); comm errors: %s / %s",
                kind, djob_.M, pjob_.T, waited, phase_[0].load(), phase_[1].load(), phase_[2].load(), phase_[3].load(),
                Comm::describe_error(comm_->error()).c_str(), Comm::describe_error(comm2_->error()).c_str());
        }
    }
    if (guards_enabled() && check_guards())
        fail("guard check failed after a " +
             std::string(job_ == Job::Prefill ? "prefill"
                         : job_ == Job::Mtp   ? "mtp"
                                              : "decode") +
             " job (M=" + std::to_string(djob_.M) + ")");
    if (!error_.empty()) fail(error_);
    if (comm_->error()) fail("collective timed out: " + Comm::describe_error(comm_->error()));
    if (comm2_->error()) fail("collective timed out (prefill channel 2): " + Comm::describe_error(comm2_->error()));
}

void Engine::slot_reset(int slot) {
    QW_CHECK(slot >= 0 && slot < num_slots(), "bad slot");
    SlotHost &sh = slots_[size_t(slot)];
    sh.len = 0;
    sh.hist.clear();
    sh.reset = true;
    sh.pend_valid = false;
    sh.run_first = -1;
}

int64_t Engine::slot_capacity(int slot) const {
    return opt_.slot_tokens[size_t(slot)];
}
int64_t Engine::slot_len(int slot) const {
    return slots_[size_t(slot)].len;
}
int Engine::max_slot_tokens() const {
    return *std::max_element(opt_.slot_tokens.begin(), opt_.slot_tokens.end());
}

// single-sequence convenience API (slot 0)
void Engine::reset() {
    slot_reset(0);
}
int64_t Engine::position() const {
    return slot_len(0);
}
const std::vector<float> &Engine::step(int32_t token) {
    const auto &l = decode({{0, token}});
    logits_.assign(l.begin(), l.begin() + VOCAB);
    return logits_;
}

}  // namespace qw
