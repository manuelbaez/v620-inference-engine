// Disk tier of the prefix cache: conversations saved to the host-RAM tier are
// also written to files by a background thread, the directory is indexed at
// startup, and a prompt that extends a stored conversation loads it back.
// Survives restarts; LRU within a byte budget.
//
// File (native endian): Header, tokens int32[n], logits float[vocab], then each
// rank's state (Engine::host_state_rank_bytes bytes). Written to a temporary
// name and renamed, so a crash never leaves a partial entry.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "engine/engine.hpp"

namespace qw {

class DiskTier {
public:
    DiskTier(Engine &e, std::string dir, size_t budget_bytes);
    ~DiskTier();  // finishes queued writes
    DiskTier(const DiskTier &) = delete;
    DiskTier &operator=(const DiskTier &) = delete;

    // Queues a write of a saved state (no-op if these exact tokens are stored or queued).
    void store(const std::vector<int32_t> &tokens, const std::vector<float> &logits,
               std::shared_ptr<Engine::HostState> state);
    // Whether these exact tokens are stored or queued.
    bool has(const std::vector<int32_t> &tokens) const;
    // Tokens of the longest stored prefix of `prompt` that is longer than at_least, or 0.
    size_t best(const std::vector<int32_t> &prompt, size_t at_least) const;
    // Loads the stored entry with exactly `len` tokens of `prompt` into pinned memory.
    bool load(const std::vector<int32_t> &prompt, size_t len, std::vector<int32_t> &tokens, std::vector<float> &logits,
              std::shared_ptr<Engine::HostState> &state);
    // Blocks until queued writes are on disk.
    void flush();
    size_t entries() const;

private:
    struct Entry {
        std::string path;
        std::vector<int32_t> tokens;
        size_t bytes = 0;
        int64_t used = 0;  // last use (seconds since epoch), for LRU
    };
    struct Job {
        std::vector<int32_t> tokens;
        std::vector<float> logits;
        std::shared_ptr<Engine::HostState> state;
    };
    void scan();
    void writer_loop();
    bool write_file(const Job &j, Entry &out);
    void enforce_budget();  // with mu_ held

    Engine &e_;
    std::string dir_;
    size_t budget_;
    uint64_t layout_;
    mutable std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    std::vector<Entry> index_;
    std::deque<Job> queue_;
    size_t bytes_ = 0;
    bool busy_ = false, stop_ = false;
    std::thread writer_;
};

}  // namespace qw
