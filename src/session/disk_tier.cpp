#include "session/disk_tier.hpp"

#include <sys/stat.h>
#include <utime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/common.hpp"
#include "core/config.hpp"

namespace qw {

namespace fs = std::filesystem;

namespace {

struct Header {
    char magic[8];
    uint64_t layout;
    int64_t n;
    uint32_t ranks;
    uint32_t vocab;
    uint64_t rank_bytes;
};
constexpr char MAGIC[8] = {'Q', 'W', 'K', 'V', 'C', 'A', 'C', '1'};

int64_t now_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

uint64_t token_hash(const std::vector<int32_t> &t) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t v : t)
        for (int b = 0; b < 4; ++b) {
            h ^= (uint32_t(v) >> (8 * b)) & 0xff;
            h *= 1099511628211ull;
        }
    return h;
}

bool is_prefix(const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

bool read_all(std::FILE *f, void *p, size_t n) {
    return std::fread(p, 1, n, f) == n;
}
bool write_all(std::FILE *f, const void *p, size_t n) {
    return std::fwrite(p, 1, n, f) == n;
}

}  // namespace

DiskTier::DiskTier(Engine &e, std::string dir, size_t budget_bytes)
    : e_(e), dir_(std::move(dir)), budget_(budget_bytes), layout_(e.state_layout_id()) {
    fs::create_directories(dir_);
    scan();
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

// Index the directory: headers and tokens only; skip foreign or partial files.
void DiskTier::scan() {
    size_t skipped = 0;
    for (const auto &de : fs::directory_iterator(dir_)) {
        if (!de.is_regular_file() || de.path().extension() != ".qwkv") continue;
        std::FILE *f = std::fopen(de.path().c_str(), "rb");
        if (!f) continue;
        Header h{};
        Entry en;
        bool ok = read_all(f, &h, sizeof h) && std::memcmp(h.magic, MAGIC, 8) == 0 && h.layout == layout_ && h.n > 0 &&
                  h.vocab == uint32_t(cfg::VOCAB);
        if (ok) {
            en.tokens.resize(size_t(h.n));
            ok = read_all(f, en.tokens.data(), en.tokens.size() * 4);
        }
        std::fclose(f);
        if (!ok) {
            ++skipped;
            continue;
        }
        en.path = de.path().string();
        en.bytes = size_t(de.file_size());
        struct stat st{};
        en.used = stat(en.path.c_str(), &st) == 0 ? int64_t(st.st_mtime) : 0;
        bytes_ += en.bytes;
        index_.push_back(std::move(en));
    }
    enforce_budget();
    log("disk tier: %s: %zu entries, %.1f GB%s", dir_.c_str(), index_.size(), double(bytes_) / 1e9,
        skipped ? (" (" + std::to_string(skipped) + " unusable files skipped)").c_str() : "");
}

void DiskTier::store(const std::vector<int32_t> &tokens, const std::vector<float> &logits,
                     std::shared_ptr<Engine::HostState> state) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const Entry &en : index_)
        if (en.tokens == tokens) return;
    for (const Job &j : queue_)
        if (j.tokens == tokens) return;
    queue_.push_back({tokens, logits, std::move(state)});
    cv_.notify_all();
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
            busy_ = true;
        }
        Entry en;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = write_file(job, en);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        job.state.reset();  // release the pinned buffers outside the lock
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (ok) {
                bytes_ += en.bytes;
                index_.push_back(std::move(en));
                enforce_budget();
                log("disk tier: wrote %zu tokens (%.2f GB in %.1f s); %zu entries, %.1f GB", job.tokens.size(),
                    double(index_.back().bytes) / 1e9, dt, index_.size(), double(bytes_) / 1e9);
            }
            busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

bool DiskTier::write_file(const Job &j, Entry &out) {
    const auto &hs = *j.state;
    char name[64];
    std::snprintf(name, sizeof name, "%016llx.qwkv", (unsigned long long)token_hash(j.tokens));
    const std::string path = dir_ + "/" + name, tmp = path + ".tmp";
    std::FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        log("disk tier: cannot write %s", tmp.c_str());
        return false;
    }
    Header h{};
    std::memcpy(h.magic, MAGIC, 8);
    h.layout = layout_;
    h.n = int64_t(j.tokens.size());
    h.ranks = uint32_t(RANKS);
    h.vocab = uint32_t(j.logits.size());
    h.rank_bytes = Engine::host_state_rank_bytes(hs);
    bool ok = h.vocab == uint32_t(cfg::VOCAB) && Engine::host_state_tokens(hs) == h.n && write_all(f, &h, sizeof h) &&
              write_all(f, j.tokens.data(), j.tokens.size() * 4) && write_all(f, j.logits.data(), j.logits.size() * 4);
    for (int r = 0; ok && r < RANKS; ++r) ok = write_all(f, Engine::host_state_buffer(hs, r), h.rank_bytes);
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        log("disk tier: write of %s failed", path.c_str());
        return false;
    }
    out.path = path;
    out.tokens = j.tokens;
    out.bytes = sizeof h + j.tokens.size() * 4 + j.logits.size() * 4 + size_t(RANKS) * h.rank_bytes;
    out.used = now_s();
    return true;
}

void DiskTier::enforce_budget() {
    while (bytes_ > budget_ && !index_.empty()) {
        auto lru = std::min_element(index_.begin(), index_.end(),
                                    [](const Entry &a, const Entry &b) { return a.used < b.used; });
        std::remove(lru->path.c_str());
        bytes_ -= lru->bytes;
        index_.erase(lru);
    }
}

bool DiskTier::has(const std::vector<int32_t> &tokens) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const Entry &en : index_)
        if (en.tokens == tokens) return true;
    for (const Job &j : queue_)
        if (j.tokens == tokens) return true;
    return false;
}

size_t DiskTier::best(const std::vector<int32_t> &prompt, size_t at_least) const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t best = 0;
    for (const Entry &en : index_)
        if (en.tokens.size() > std::max(at_least, best) && is_prefix(en.tokens, prompt)) best = en.tokens.size();
    return best;
}

bool DiskTier::load(const std::vector<int32_t> &prompt, size_t len, std::vector<int32_t> &tokens,
                    std::vector<float> &logits, std::shared_ptr<Engine::HostState> &state) {
    std::string path;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (Entry &en : index_)
            if (en.tokens.size() == len && is_prefix(en.tokens, prompt)) {
                path = en.path;
                en.used = now_s();
                break;
            }
    }
    if (path.empty()) return false;
    const auto t0 = std::chrono::steady_clock::now();
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    Header h{};
    bool ok = read_all(f, &h, sizeof h) && std::memcmp(h.magic, MAGIC, 8) == 0 && h.layout == layout_ &&
              h.n == int64_t(len) && h.ranks == uint32_t(RANKS);
    if (ok) {
        tokens.resize(size_t(h.n));
        logits.resize(h.vocab);
        state = e_.alloc_host_state(h.n);
        ok = Engine::host_state_rank_bytes(*state) == h.rank_bytes && read_all(f, tokens.data(), tokens.size() * 4) &&
             read_all(f, logits.data(), logits.size() * 4);
        for (int r = 0; ok && r < RANKS; ++r) ok = read_all(f, Engine::host_state_buffer(*state, r), h.rank_bytes);
    }
    std::fclose(f);
    if (!ok) {
        state.reset();
        log("disk tier: %s is unreadable", path.c_str());
        return false;
    }
    (void)utime(path.c_str(), nullptr);  // LRU across restarts
    log("disk tier: loaded %zu tokens in %.2f s", tokens.size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

void DiskTier::flush() {
    std::unique_lock<std::mutex> lk(mu_);
    idle_cv_.wait(lk, [&] { return queue_.empty() && !busy_; });
}

size_t DiskTier::entries() const {
    std::lock_guard<std::mutex> lk(mu_);
    return index_.size();
}

}  // namespace qw
