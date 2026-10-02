// Engine lifecycle: loading, the per-rank host threads and job dispatch,
// warmup, and slot bookkeeping.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "core/common.hpp"
#include "engine/device_mem.hpp"
#include "engine/engine.hpp"
#include "engine/rank.hpp"
#include "vision/vision_weights.hpp"
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
    // The n-gram table is reserved in RAM up front (mlock); QW_PLE_PIN=0 leaves it in page cache.
    {
        const char *pin = std::getenv("QW_PLE_PIN");
        ple_->pin_in_background(!(pin && std::string(pin) == "0"));
    }
    const size_t slot = std::max<size_t>(64 * 1024, size_t(opt_.prefill_chunk / 2) * gpu::D_PAD * 4 + 4096);
    comm_ = std::make_unique<Comm>(opt_.devices, slot);
    comm2_ = std::make_unique<Comm>(opt_.devices, slot);
    QW_CHECK(!opt_.slot_tokens.empty() && opt_.slot_tokens.size() <= size_t(MAX_SLOTS), "1..8 slots");
    for (int c : opt_.slot_tokens) QW_CHECK(c > 0 && c % 256 == 0, "slot sizes must be positive multiples of 256");
    slots_.resize(opt_.slot_tokens.size());
    pen_.resize(opt_.slot_tokens.size());
    logits_.resize(VOCAB);
    mtp_ = opt_.mtp && !std::getenv("QW_NO_MTP") && st_.has("mtp.fc_embedding.weight");
    moe_stats_ = std::getenv("QW_MOE_STATS") != nullptr;

    for (int r = 0; r < RANKS; ++r) ranks_.push_back(std::make_unique<Rank>());
    // The ranks load in parallel, each on its own thread with its own pool
    // (the checkpoint is a read-only mapping; everything else is per rank):
    // reading, converting and uploading one rank's shard overlaps the others'.
    // QW_LOAD_SERIAL=1: one after another, each with every thread.
    const bool serial = std::getenv("QW_LOAD_SERIAL") && std::string(std::getenv("QW_LOAD_SERIAL")) != "0";
    auto load = [&](int r, ThreadPool &pool) {
        Rank &rk = *ranks_[size_t(r)];
        rk.r = r;
        rk.dev = opt_.devices[size_t(r)];
        CK(hipSetDevice(rk.dev));
        CK(hipStreamCreateWithFlags(&rk.s, hipStreamNonBlocking));
        load_rank_weights(rk, st_, pool, mtp_);
        alloc_rank_buffers(rk, opt_, max_slot_tokens(), mtp_);
        log("rank %d loaded", r);
    };
    if (serial) {
        ThreadPool pool(opt_.load_threads * RANKS);
        for (int r = 0; r < RANKS; ++r) load(r, pool);
    } else {
        std::vector<std::thread> loaders;
        std::vector<std::string> errors(RANKS);
        for (int r = 0; r < RANKS; ++r)
            loaders.emplace_back([&, r] {
                try {
                    ThreadPool pool(opt_.load_threads);
                    load(r, pool);
                } catch (const std::exception &e) {
                    errors[size_t(r)] = e.what();
                }
            });
        for (auto &t : loaders) t.join();
        for (const auto &e : errors) QW_CHECK(e.empty(), e);
    }
    if (!std::getenv("QW_NO_VISION"))
        if (auto hw = load_vision_weights(st_)) {  // a copy of the vision tower on every card
            for (auto &rkp : ranks_) {
                CK(hipSetDevice(rkp->dev));
                rkp->vit = std::make_unique<VisionEncoder>(*hw, rkp->blas, rkp->s);
            }
            vision_ = true;
            log("vision tower: %.2f GB per card", double(ranks_[0]->vit->weight_bytes()) / 1e9);
        }
    {  // keep the token embeddings (host-side lookups); unmap the checkpoint
        const TensorView &emb = st_.get("model.language_model.embed_tokens.weight", DType::BF16, {VOCAB, H});
        const size_t bytes = size_t(VOCAB) * H * 2;
        CK(hipHostMalloc(reinterpret_cast<void **>(&embed_), bytes, hipHostMallocPortable | hipHostMallocMapped));
        std::memcpy(embed_, emb.u16(), bytes);
        for (auto &rkp : ranks_) {
            CK(hipSetDevice(rkp->dev));
            void *d = nullptr;
            CK(hipHostGetDevicePointer(&d, embed_, 0));
            rkp->embed_dev = static_cast<const uint16_t *>(d);
        }
        st_.clear();
    }
    // Ready means ready: the n-gram table has been reading into RAM while the
    // ranks loaded; wait for the rest before warmup (which would otherwise read
    // rows from disk) and before the server reports healthy.
    {
        const auto t0 = std::chrono::steady_clock::now();
        ple_->wait_pinned();
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (waited > 1.0) log("waited %.0f s for the PLE table to finish loading", waited);
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
    if (vision_) {  // the vision tower's GEMM shapes, on every card
        std::vector<float> px(size_t(32) * 32 * 1536, 0.1f);
        std::vector<std::vector<float>> outs(RANKS, std::vector<float>(size_t(32) * 32 / 4 * H));
        std::vector<VisionSlice> slices;
        for (int r = 0; r < RANKS; ++r) slices.push_back({px.data(), 32, 32, outs[size_t(r)].data()});
        encode_vision(slices);
    }
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
    for (int32_t *c : moe_counts_)
        if (c) (void)hipHostFree(c);
    if (embed_) (void)hipHostFree(embed_);
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
            else if (job_ == Job::Draft)
                run_draft_rank(rk);
            else if (job_ == Job::Vision)
                run_vision_rank(rk);
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
    if (job_ != Job::Decode) dlogits_on_gpu_ = false;  // the batch logits get overwritten
    static const bool trace = std::getenv("QW_TRACE") != nullptr;
    if (trace) {
        if (job_ == Job::Prefill)
            for (const PrefillSeg &sg : pjob_.segs)
                log("trace: prefill slot %d [%lld, +%d)%s%s", sg.slot, (long long)sg.start, sg.T,
                    sg.mtp_pend ? " mtp_pend" : "", pjob_.segs.size() > 1 ? " (batched)" : "");
        else
            log("trace: %s M=%d%s", job_ == Job::Mtp ? "mtp" : job_ == Job::Draft ? "draft" : "decode", djob_.M,
                djob_.save ? " save" : "");
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
            const char *kind = job_ == Job::Prefill ? "prefill" : job_ == Job::Mtp ? "mtp" : job_ == Job::Draft ? "draft" : "decode";
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

std::string Engine::failure() const {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!error_.empty()) return error_;
    }
    if (comm_->error()) return "collective timed out: " + Comm::describe_error(comm_->error());
    if (comm2_->error()) return "collective timed out (prefill channel 2): " + Comm::describe_error(comm2_->error());
    return {};
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
