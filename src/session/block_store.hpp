// The prefix cache's host tier, shared by every conversation.
//
// Prompts are cut into blocks of BLOCK tokens, each keyed by a hash chain (its
// parent's key and its own tokens), so a block is only ever reused after the
// same prefix: reuse is exact. A block holds the KV of its positions. The
// recurrent state (GDN, conv rings) is not position-indexed and cannot be
// rebuilt from KV, so a prompt resumes at a snapshot: the state at the end of
// some block, stored with it. The last block on a snapshot's path may be
// partial (fewer tokens, always a leaf), so snapshots sit at any position,
// e.g. a chat message boundary.
//
// Blocks and snapshots live in pinned RAM (LRU within a budget) and, with a
// disk tier, in files as well (written as they are added, their own LRU
// budget); a restore loads whatever is only on disk.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "engine/engine.hpp"
#include "engine/pinned_pool.hpp"
#include "session/disk_tier.hpp"

namespace qw {

class BlockStore {
public:
    static constexpr int BLOCK = 256;

    // disk_dir empty: no disk tier. Environment: QW_LOAD_THREADS, QW_POOL_ARENA (see PoolOptions),
    // QW_KV_PAIRS=1 keeps one copy of the KV of each replica group of ranks (they are identical: half the pinned
    // memory and half the device-to-host traffic of a block; the files stay as they were, four buffers, and a read
    // into shared buffers skips the replicas'), QW_KV_PAIR_CHECK=N exports the replicas of every Nth shared block
    // as well and counts the ones that differ (a check of the identity, off by default).
    BlockStore(Engine &e, size_t ram_budget, const std::string &disk_dir, size_t disk_budget);
    ~BlockStore();
    BlockStore(const BlockStore &) = delete;
    BlockStore &operator=(const BlockStore &) = delete;

    struct Hit {
        int64_t n = 0;  // tokens restored
        uint64_t node = 0;
    };
    // The deepest snapshot on `prompt`'s path with at_least < n <= prompt.size()
    // (n == prompt.size() only when its logits are stored); n = 0 if none.
    Hit lookup(const std::vector<int32_t> &prompt, int64_t at_least) const;
    // Loads a hit into `slot`, which already holds the prompt's tokens [0, keep):
    // the KV of the blocks not covered by that, then the recurrent state. Sets
    // *logits when they are stored (else clears it). False if a file could not
    // be read (the entry is dropped; the slot's state is then undefined).
    bool restore(const Hit &h, const std::vector<int32_t> &prompt, int slot, int64_t keep, std::vector<float> *logits);
    // Brings a hit's entries that are only on disk into RAM on background
    // threads, so restore() then reads nothing from disk: true while that load
    // runs (call again to poll), false once everything the hit needs is in
    // RAM. A hit that needs nothing from disk is never held up: false at once,
    // whatever load is running. Loads run one at a time, so a hit that needs
    // another read waits for the running one first. QW_LOAD_THREADS (default 1)
    // threads read the files of a load in parallel.
    bool load(const Hit &h);
    // Stores the state of `slot` at VRAM snapshot `snap`, which was taken at
    // tokens.size() tokens of the slot: the path's blocks not stored yet, and
    // the snapshot (with the logits after its last token, if given).
    void save(int slot, int snap, const std::vector<int32_t> &tokens, const std::vector<float> *logits);
    bool has_snapshot(const std::vector<int32_t> &tokens) const;
    // Pins memory for that many new blocks and snapshots in the background
    // (call before a prefill whose saves will need it).
    void reserve(size_t blocks, size_t snapshots);
    void flush();  // waits for the disk writes
    // Waits (at most `seconds`) for the pools to pin the standing reserve (QW_POOL_RESERVE_GB), so that the first
    // request after a start finds it there; returns at once without one. Logs how long it took.
    void wait_reserve(double seconds);

    struct Stats {
        uint64_t hits = 0, tokens_restored = 0, snapshots_saved = 0;
        size_t ram_bytes = 0, disk_bytes = 0, blocks = 0, snapshots = 0;
        // the pinned pools, summed: arenas pinned and the time that took (background threads), and the saves and
        // loads that had to wait for an arena (callers' time)
        uint64_t pins = 0, pin_waits = 0;
        double pin_s = 0, pin_wait_s = 0;
        uint64_t pair_mismatches = 0;  // QW_KV_PAIR_CHECK: replicas that differed from their group's first rank
    };
    Stats stats() const;

private:
    struct Payload;  // one pinned buffer per rank, returned to the pools on destruction
    struct Node {
        uint64_t parent = 0;
        int64_t start = 0;            // position of tokens[0]
        std::vector<int32_t> tokens;  // BLOCK, or fewer for a partial block (a leaf)
        std::vector<uint64_t> children;
        std::shared_ptr<Payload> kv;  // in RAM
        bool kv_disk = false;
        // snapshot at the block's end
        bool has_snap = false, snap_logits = false, snap_disk = false;
        std::shared_ptr<Payload> snap;  // in RAM, with its logits
        std::vector<float> logits;
        size_t disk_bytes = 0;
        uint64_t used = 0;
        int64_t end() const { return start + int64_t(tokens.size()); }
    };
    static constexpr uint64_t ROOT = 0x9e3779b97f4a7c15ull;
    static uint64_t key(uint64_t parent, const int32_t *t, size_t n);

    std::shared_ptr<Payload> alloc(bool snap);
    size_t kv_bytes() const { return kv_pool_[0]->unit_bytes() * size_t(kv_copies_); }    // RAM of a block's KV
    size_t snap_bytes() const { return snap_pool_[0]->unit_bytes() * size_t(RANKS); }     // ... of a snapshot's state
    void check_pairs(int slot, int64_t pos, int64_t n, const Node &nd);
    size_t ram_size(const Node &nd) const;
    const Node *find(uint64_t k) const;
    // The node for tokens [pos, pos + n) after `parent`, if stored.
    uint64_t child(uint64_t parent, const int32_t *t, size_t n) const;
    std::vector<uint64_t> path_to(uint64_t k) const;  // root's child first
    bool ensure_kv(Node &nd, uint64_t k);
    bool ensure_snap(Node &nd, uint64_t k);
    void remove_subtree(uint64_t k);
    void enforce_budgets();
    // A background load of on-disk entries (load()): the loader threads take
    // the pinned buffers themselves (pinning new arenas can take seconds) and
    // read the files into them; finish_load() attaches them to the nodes that
    // still want them.
    struct Load {
        struct Item {
            uint64_t key;
            bool snap;
            std::shared_ptr<Payload> pl;  // pinned on a load thread
            size_t rank_bytes;
            std::vector<float> logits;
        };
        std::vector<Item> items;
        std::vector<std::thread> workers;
        std::atomic<size_t> next{0};     // the item a worker takes next
        std::atomic<size_t> running{0};  // workers still going; the last sets done
        std::atomic<size_t> failed{0};
        std::atomic<int64_t> pin_us{0}, read_us{0};  // thread time getting buffers / reading files
        std::atomic<bool> done{false};
        std::chrono::steady_clock::time_point t0;
        size_t bytes = 0;
    };
    // The entries of the hit's path that are only on disk, and their size.
    std::vector<Load::Item> missing(const Hit &h, size_t *bytes) const;
    void load_worker(Load *l);
    size_t pinned_bytes() const;  // arenas held by the pools
    void finish_load();
    void load_index();

    Engine &e_;
    size_t ram_budget_, disk_budget_;
    size_t ram_bytes_ = 0, disk_bytes_ = 0;
    int load_threads_ = 1;
    bool kv_pairs_ = false;  // one KV buffer per replica group (QW_KV_PAIRS)
    int kv_copies_ = RANKS;  // KV buffers a block holds: RANKS, or one per replica group
    int pair_check_ = 0;     // QW_KV_PAIR_CHECK
    uint64_t pair_checked_ = 0;
    std::array<std::unique_ptr<PinnedPool>, RANKS> kv_pool_, snap_pool_;
    std::unordered_map<uint64_t, Node> nodes_;
    uint64_t clock_ = 0;
    Stats stats_;
    std::unique_ptr<Load> load_;
    std::unique_ptr<DiskTier> disk_;  // declared last: destroyed (writes finished) before the pools
};

}  // namespace qw
