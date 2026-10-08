#include "session/disk_tier.hpp"

#include <sys/stat.h>
#include <utime.h>

#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/common.hpp"
#include "core/config.hpp"
#include "core/shard.hpp"

namespace qw {

namespace fs = std::filesystem;

namespace {

struct Header {
    char magic[8];
    uint64_t layout, hash, parent;
    int64_t ntok;
    uint32_t ranks, nlogits;
    uint64_t rank_bytes;
};
constexpr char MAGIC_BLOCK[8] = {'Q', 'W', 'B', 'L', 'O', 'C', 'K', '1'};
constexpr char MAGIC_SNAP[8] = {'Q', 'W', 'S', 'N', 'A', 'P', 'S', '1'};
// v2 block: the KV of a replica group of ranks is identical, so the file holds one buffer per group (the first
// rank's) instead of one per rank. v1 blocks (a buffer per rank) stay readable.
constexpr char MAGIC_BLOCK2[8] = {'Q', 'W', 'B', 'L', 'O', 'C', 'K', '2'};
constexpr int GROUPS = RANKS / cfg::KV_REPLICAS;

bool is_primary(int r) {
    return cfg::kv_primary(r) == r;
}

// Whether a block's buffers are one per group: every replica's is its group's first rank's (a store that keeps one
// KV copy per group). Snapshots have no identical pairs.
bool grouped(bool snap, const Engine::RankBufs &bufs) {
    if (snap) return false;
    for (int r = 0; r < RANKS; ++r)
        if (!is_primary(r) && bufs[size_t(r)] != bufs[size_t(cfg::kv_primary(r))]) return false;
    return true;
}

bool block_magic(const char *m) {
    return std::memcmp(m, MAGIC_BLOCK, 8) == 0 || std::memcmp(m, MAGIC_BLOCK2, 8) == 0;
}

bool read_all(std::FILE *f, void *p, size_t n) {
    return std::fread(p, 1, n, f) == n;
}
bool write_all(std::FILE *f, const void *p, size_t n) {
    return std::fwrite(p, 1, n, f) == n;
}

}  // namespace

DiskTier::DiskTier(std::string dir, uint64_t layout) : dir_(std::move(dir)), layout_(layout) {
    fs::create_directories(dir_);
    writer_ = std::thread([this] { writer_loop(); });
}

DiskTier::~DiskTier() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    writer_.join();
}

std::string DiskTier::path(uint64_t hash, bool snap) const {
    char name[32];
    std::snprintf(name, sizeof name, "%016llx.%s", (unsigned long long)hash, snap ? "qws" : "qwb");
    return dir_ + "/" + name;
}

std::vector<DiskTier::Meta> DiskTier::scan() {
    std::vector<Meta> out;
    size_t dropped = 0;
    for (const auto &de : fs::directory_iterator(dir_)) {
        if (!de.is_regular_file()) continue;
        const auto ext = de.path().extension();
        if (ext != ".qwb" && ext != ".qws") {  // temporaries of an interrupted write, old formats
            std::error_code ec;
            fs::remove(de.path(), ec);
            ++dropped;
            continue;
        }
        const bool snap = ext == ".qws";
        std::FILE *f = std::fopen(de.path().c_str(), "rb");
        if (!f) continue;
        Header h{};
        Meta m;
        bool ok = read_all(f, &h, sizeof h) &&
                  (snap ? std::memcmp(h.magic, MAGIC_SNAP, 8) == 0 : block_magic(h.magic)) && h.layout == layout_ &&
                  h.ranks == uint32_t(RANKS);
        if (ok && !snap) {
            m.tokens.resize(size_t(h.ntok));
            ok = h.ntok > 0 && read_all(f, m.tokens.data(), m.tokens.size() * 4);
        }
        std::fclose(f);
        if (!ok) {
            std::error_code ec;
            fs::remove(de.path(), ec);
            ++dropped;
            continue;
        }
        m.hash = h.hash;
        m.parent = h.parent;
        m.snap = snap;
        m.logits = h.nlogits > 0;
        m.bytes = size_t(de.file_size());
        struct stat st{};
        m.used = stat(de.path().c_str(), &st) == 0 ? int64_t(st.st_mtime) : 0;
        out.push_back(std::move(m));
    }
    if (dropped) log("disk tier: removed %zu unusable files", dropped);
    return out;
}

size_t DiskTier::write(const Meta &m, std::vector<float> logits, const Engine::RankBufs &bufs, size_t rank_bytes,
                       std::shared_ptr<const void> keep) {
    const size_t copies = grouped(m.snap, bufs) ? size_t(GROUPS) : size_t(RANKS);
    const size_t bytes = sizeof(Header) + m.tokens.size() * 4 + logits.size() * 4 + copies * rank_bytes;
    std::lock_guard<std::mutex> lk(mu_);
    pending_.insert({m.hash, m.snap});
    pending_bytes_ += bytes;
    queue_.push_back({m, std::move(logits), bufs, rank_bytes, std::move(keep), bytes});
    cv_.notify_all();
    return bytes;
}

void DiskTier::writer_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stopping, nothing left
            job = std::move(queue_.front());
            queue_.pop_front();
            working_ = true;
        }
        if (job.remove)
            std::remove(path(job.m.hash, job.m.snap).c_str());
        else
            write_file(job);
        job.keep.reset();  // release the buffers outside the lock
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!job.remove) {
                pending_.erase({job.m.hash, job.m.snap});
                pending_bytes_ -= job.bytes;
            }
            working_ = false;
        }
        done_cv_.notify_all();
    }
}

bool DiskTier::write_file(const Job &j) {
    const std::string p = path(j.m.hash, j.m.snap), tmp = p + ".tmp";
    std::FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        log("disk tier: cannot write %s", tmp.c_str());
        return false;
    }
    Header h{};
    const bool two = grouped(j.m.snap, j.bufs);
    std::memcpy(h.magic, j.m.snap ? MAGIC_SNAP : two ? MAGIC_BLOCK2 : MAGIC_BLOCK, 8);
    h.layout = layout_;
    h.hash = j.m.hash;
    h.parent = j.m.parent;
    h.ntok = int64_t(j.m.tokens.size());
    h.ranks = uint32_t(RANKS);
    h.nlogits = uint32_t(j.logits.size());
    h.rank_bytes = j.rank_bytes;
    bool ok = write_all(f, &h, sizeof h) && write_all(f, j.m.tokens.data(), j.m.tokens.size() * 4) &&
              write_all(f, j.logits.data(), j.logits.size() * 4);
    for (int r = 0; ok && r < RANKS; ++r)
        if (!two || is_primary(r)) ok = write_all(f, j.bufs[size_t(r)], j.rank_bytes);
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), p.c_str()) != 0) {
        std::remove(tmp.c_str());
        log("disk tier: write of %s failed", p.c_str());
        return false;
    }
    return true;
}

void DiskTier::wait_written(uint64_t hash, bool snap) {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return !pending_.count({hash, snap}); });
}

bool DiskTier::read(uint64_t hash, bool snap, const Engine::RankBufs &bufs, size_t rank_bytes,
                    std::vector<float> *logits) {
    wait_written(hash, snap);
    const std::string p = path(hash, snap);
    std::FILE *f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    Header h{};
    bool ok = read_all(f, &h, sizeof h) && (snap ? std::memcmp(h.magic, MAGIC_SNAP, 8) == 0 : block_magic(h.magic)) &&
              h.layout == layout_ && h.hash == hash && h.rank_bytes == rank_bytes && h.ranks == uint32_t(RANKS);
    const bool two = ok && !snap && std::memcmp(h.magic, MAGIC_BLOCK2, 8) == 0;
    ok = ok && std::fseek(f, long(h.ntok) * 4, SEEK_CUR) == 0;
    if (ok) {
        std::vector<float> l(h.nlogits);
        ok = read_all(f, l.data(), l.size() * 4);
        if (logits) *logits = std::move(l);
    }
    for (int r = 0; ok && r < RANKS; ++r) {
        if (two) {  // one buffer per group in the file: read for the first rank of each group
            if (is_primary(r)) ok = read_all(f, bufs[size_t(r)], rank_bytes);
            continue;
        }
        // a buffer that an earlier rank already took is a replica's (the KV of a replica group is identical): its
        // bytes in the file are skipped, not read again
        bool shared = false;
        for (int q = 0; q < r; ++q) shared = shared || bufs[size_t(q)] == bufs[size_t(r)];
        ok = shared ? std::fseek(f, long(rank_bytes), SEEK_CUR) == 0 : read_all(f, bufs[size_t(r)], rank_bytes);
    }
    std::fclose(f);
    if (ok && two)  // the caller's replicas have buffers of their own: they get the group's bytes
        for (int r = 0; r < RANKS; ++r)
            if (!is_primary(r) && bufs[size_t(r)] != bufs[size_t(cfg::kv_primary(r))])
                std::memcpy(bufs[size_t(r)], bufs[size_t(cfg::kv_primary(r))], rank_bytes);
    if (!ok) {
        log("disk tier: %s is unreadable", p.c_str());
        return false;
    }
    (void)utime(p.c_str(), nullptr);  // LRU across restarts
    return true;
}

void DiskTier::remove(uint64_t hash, bool snap) {
    std::shared_ptr<const void> dropped;  // a cancelled write's buffers, released outside the lock
    {
        std::lock_guard<std::mutex> lk(mu_);
        // a write of it still waiting in the queue is dropped (the one being written, if any, finishes first: the
        // unlink below is queued behind it)
        if (pending_.count({hash, snap}))
            for (auto it = queue_.begin(); it != queue_.end(); ++it)
                if (!it->remove && it->m.hash == hash && it->m.snap == snap) {
                    pending_.erase({hash, snap});
                    pending_bytes_ -= it->bytes;
                    dropped = std::move(it->keep);
                    queue_.erase(it);
                    break;
                }
        Job j;
        j.m.hash = hash;
        j.m.snap = snap;
        j.remove = true;
        queue_.push_back(std::move(j));
    }
    cv_.notify_all();
    done_cv_.notify_all();
}

size_t DiskTier::pending_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_bytes_;
}

void DiskTier::flush() {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return queue_.empty() && pending_.empty() && !working_; });
}

}  // namespace qw
