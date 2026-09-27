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
#include <thread>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/engine.hpp"
#include "engine/pinned_pool.hpp"
#include "session/disk_tier.hpp"

namespace qw {

class BlockStore {
public:
    static constexpr int BLOCK = 256;

    // disk_dir empty: no disk tier.
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
    // Brings a hit's entries that are only on disk into RAM on a background
    // thread, so restore() then reads nothing from disk: true while that load
    // runs (call again to poll; one load at a time, so another hit's load also
    // reads as running), false once everything the hit needs is in RAM.
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

    struct Stats {
        uint64_t hits = 0, tokens_restored = 0, snapshots_saved = 0;
        size_t ram_bytes = 0, disk_bytes = 0, blocks = 0, snapshots = 0;
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
    size_t ram_size(const Node &nd) const;
    const Node *find(uint64_t k) const;
    // The node for tokens [pos, pos + n) after `parent`, if stored.
    uint64_t child(uint64_t parent, const int32_t *t, size_t n) const;
    std::vector<uint64_t> path_to(uint64_t k) const;  // root's child first
    bool ensure_kv(Node &nd, uint64_t k);
    bool ensure_snap(Node &nd, uint64_t k);
    void remove_subtree(uint64_t k);
    void enforce_budgets();
    // A background load of on-disk entries (load()): the payloads are allocated
    // by the caller's thread and filled by the loader; finish_load() attaches
    // them to the nodes that still want them.
    struct Load {
        struct Item {
            uint64_t key;
            bool snap;
            std::shared_ptr<Payload> pl;  // pinned on the load thread
            size_t rank_bytes;
            std::vector<float> logits;
        };
        std::vector<Item> items;
        std::atomic<bool> done{false};
        std::thread th;
    };
    void finish_load();
    void load_index();

    Engine &e_;
    size_t ram_budget_, disk_budget_;
    size_t ram_bytes_ = 0, disk_bytes_ = 0;
    std::array<std::unique_ptr<PinnedPool>, RANKS> kv_pool_, snap_pool_;
    std::unordered_map<uint64_t, Node> nodes_;
    uint64_t clock_ = 0;
    Stats stats_;
    std::unique_ptr<Load> load_;
    std::unique_ptr<DiskTier> disk_;  // declared last: destroyed (writes finished) before the pools
};

}  // namespace qw
