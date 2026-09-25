// Disk tier of the prefix cache: the block store's blocks and snapshots as
// files, written by a background thread and indexed at startup, so the cache
// survives restarts. It only holds files; the block store decides what to
// keep (LRU within its budget).
//
// Files (native endian), written to a temporary name and renamed so a crash
// never leaves a partial file:
//   <hash>.qwb  a block:    Header, tokens int32[n], each rank's KV
//   <hash>.qws  a snapshot: Header, logits float[nlogits], each rank's recurrent state
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine/engine.hpp"

namespace qw {

class DiskTier {
public:
    struct Meta {
        uint64_t hash = 0, parent = 0;
        bool snap = false;             // a snapshot file (else a block)
        std::vector<int32_t> tokens;   // blocks only
        bool logits = false;           // snapshots: logits after the last token are stored
        size_t bytes = 0;              // file size
        int64_t used = 0;              // mtime (last use), seconds since epoch
    };

    DiskTier(std::string dir, uint64_t layout);
    ~DiskTier();  // finishes queued writes
    DiskTier(const DiskTier &) = delete;
    DiskTier &operator=(const DiskTier &) = delete;

    // Every usable file in the directory (removes unusable ones).
    std::vector<Meta> scan();
    // Queues a write of rank_bytes from each of bufs; `keep` holds the buffers
    // alive until then. Returns the file's size.
    size_t write(const Meta &m, std::vector<float> logits, const Engine::RankBufs &bufs, size_t rank_bytes,
                 std::shared_ptr<const void> keep);
    // Reads a file's payload into bufs (and its logits); waits for a queued write of it first.
    bool read(uint64_t hash, bool snap, const Engine::RankBufs &bufs, size_t rank_bytes, std::vector<float> *logits);
    void remove(uint64_t hash, bool snap);
    // Blocks until queued writes are on disk.
    void flush();

private:
    struct Job {
        Meta m;
        std::vector<float> logits;
        Engine::RankBufs bufs;
        size_t rank_bytes;
        std::shared_ptr<const void> keep;
    };
    std::string path(uint64_t hash, bool snap) const;
    void writer_loop();
    bool write_file(const Job &j);
    void wait_written(uint64_t hash, bool snap);  // with lk held

    std::string dir_;
    uint64_t layout_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> queue_;
    std::set<std::pair<uint64_t, bool>> pending_;  // queued or being written
    bool stop_ = false;
    std::thread writer_;
};

}  // namespace qw
