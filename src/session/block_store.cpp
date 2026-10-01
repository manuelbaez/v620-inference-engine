#include "session/block_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "core/common.hpp"
#include "core/config.hpp"

namespace qw {

namespace {
constexpr int KV_ARENA_UNITS = 48;   // ~250 MB per rank
constexpr int SNAP_ARENA_UNITS = 8;  // ~250 MB per rank
constexpr size_t KV_LOW_WATER = 8, SNAP_LOW_WATER = 2;  // free units kept ahead of need
}  // namespace

struct BlockStore::Payload {
    Engine::RankBufs p{};
    std::array<PinnedPool *, RANKS> pools{};
    ~Payload() {
        for (int r = 0; r < RANKS; ++r)
            if (p[size_t(r)]) pools[size_t(r)]->put(p[size_t(r)]);
    }
};

uint64_t BlockStore::key(uint64_t parent, const int32_t *t, size_t n) {
    uint64_t h = parent ^ (0xcbf29ce484222325ull + n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t v = uint32_t(t[i]);
        for (int b = 0; b < 4; ++b) {
            h ^= (v >> (8 * b)) & 0xff;
            h *= 1099511628211ull;
        }
    }
    return h == ROOT ? h + 1 : h;
}

BlockStore::BlockStore(Engine &e, size_t ram_budget, const std::string &disk_dir, size_t disk_budget)
    : e_(e), ram_budget_(ram_budget), disk_budget_(disk_budget) {
    for (int r = 0; r < RANKS; ++r) {
        kv_pool_[size_t(r)] =
            std::make_unique<PinnedPool>(e_.kv_rank_bytes(BLOCK), KV_ARENA_UNITS, e_.rank_device(r), 0);
        snap_pool_[size_t(r)] =
            std::make_unique<PinnedPool>(Engine::recurrent_rank_bytes(), SNAP_ARENA_UNITS, e_.rank_device(r), 0);
    }
    nodes_[ROOT] = Node{};
    if (const char *t = std::getenv("QW_LOAD_THREADS")) load_threads_ = std::max(1, std::atoi(t));
    if (!disk_dir.empty()) {
        disk_ = std::make_unique<DiskTier>(disk_dir, e_.state_layout_id());
        load_index();
    }
}

BlockStore::~BlockStore() {
    if (load_)
        for (auto &t : load_->workers)
            if (t.joinable()) t.join();
}

// Rebuilds the index from the disk tier's files; drops files whose chain is broken.
void BlockStore::load_index() {
    auto metas = disk_->scan();
    std::unordered_map<uint64_t, DiskTier::Meta *> snaps;
    for (auto &m : metas) {
        if (m.snap) {
            snaps[m.hash] = &m;
            continue;
        }
        Node &nd = nodes_[m.hash];
        nd.parent = m.parent;
        nd.tokens = std::move(m.tokens);
        nd.kv_disk = true;
        nd.disk_bytes += m.bytes;
        nd.used = uint64_t(m.used);
    }
    // link children and positions; nodes that do not reach the root are dropped
    std::vector<uint64_t> orphans;
    for (auto &[k, nd] : nodes_) {
        if (k == ROOT) continue;
        int64_t start = 0;
        uint64_t p = nd.parent;
        bool ok = true;
        for (int depth = 0; p != ROOT; ++depth) {
            auto it = nodes_.find(p);
            if (it == nodes_.end() || int(it->second.tokens.size()) != BLOCK || depth > 1 << 20) {
                ok = false;
                break;
            }
            start += BLOCK;
            p = it->second.parent;
        }
        if (ok)
            nd.start = start;
        else
            orphans.push_back(k);
    }
    for (uint64_t k : orphans) {
        disk_->remove(k, false);
        nodes_.erase(k);
    }
    for (auto &[k, nd] : nodes_)
        if (k != ROOT) nodes_[nd.parent].children.push_back(k);
    size_t snaps_used = 0;
    for (auto &[k, m] : snaps) {
        auto it = nodes_.find(k);
        if (it == nodes_.end() || k == ROOT) {
            disk_->remove(k, true);
            continue;
        }
        Node &nd = it->second;
        nd.has_snap = nd.snap_disk = true;
        nd.snap_logits = m->logits;
        nd.disk_bytes += m->bytes;
        nd.used = std::max(nd.used, uint64_t(m->used));
        ++snaps_used;
    }
    for (auto &[k, nd] : nodes_) {
        disk_bytes_ += nd.disk_bytes;
        clock_ = std::max(clock_, nd.used);
    }
    log("prefix cache: disk tier holds %zu blocks and %zu snapshots, %.1f GB", nodes_.size() - 1, snaps_used,
        double(disk_bytes_) / 1e9);
    enforce_budgets();
}

std::shared_ptr<BlockStore::Payload> BlockStore::alloc(bool snap) {
    auto pl = std::make_shared<Payload>();
    for (int r = 0; r < RANKS; ++r) {
        PinnedPool *pool = snap ? snap_pool_[size_t(r)].get() : kv_pool_[size_t(r)].get();
        pl->pools[size_t(r)] = pool;
        pl->p[size_t(r)] = pool->get();
    }
    return pl;
}

size_t BlockStore::ram_size(const Node &nd) const {
    size_t b = 0;
    if (nd.kv) b += kv_pool_[0]->unit_bytes() * RANKS;
    if (nd.snap) b += snap_pool_[0]->unit_bytes() * RANKS + nd.logits.size() * 4;
    return b;
}

const BlockStore::Node *BlockStore::find(uint64_t k) const {
    auto it = nodes_.find(k);
    return it == nodes_.end() ? nullptr : &it->second;
}

uint64_t BlockStore::child(uint64_t parent, const int32_t *t, size_t n) const {
    const uint64_t k = key(parent, t, n);
    const Node *nd = find(k);
    if (!nd || nd->parent != parent || nd->tokens.size() != n || !std::equal(t, t + n, nd->tokens.begin())) return 0;
    return k;
}

std::vector<uint64_t> BlockStore::path_to(uint64_t k) const {
    std::vector<uint64_t> path;
    for (; k != ROOT; k = nodes_.at(k).parent) path.push_back(k);
    std::reverse(path.begin(), path.end());
    return path;
}

BlockStore::Hit BlockStore::lookup(const std::vector<int32_t> &prompt, int64_t at_least) const {
    Hit best;
    const int64_t P = int64_t(prompt.size());
    auto consider = [&](uint64_t k, const Node &nd) {
        const int64_t end = nd.end();
        if (nd.has_snap && end > at_least && end > best.n && (end < P || nd.snap_logits)) best = {end, k};
    };
    uint64_t h = ROOT;
    for (int64_t pos = 0;;) {
        for (uint64_t c : nodes_.at(h).children) {  // partial blocks (snapshots mid-block)
            const Node &nd = nodes_.at(c);
            if (int(nd.tokens.size()) < BLOCK && pos + int64_t(nd.tokens.size()) <= P &&
                std::equal(nd.tokens.begin(), nd.tokens.end(), prompt.begin() + ptrdiff_t(pos)))
                consider(c, nd);
        }
        if (pos + BLOCK > P) break;
        const uint64_t k = child(h, prompt.data() + pos, BLOCK);
        if (!k) break;
        consider(k, nodes_.at(k));
        h = k;
        pos += BLOCK;
    }
    return best;
}

bool BlockStore::ensure_kv(Node &nd, uint64_t k) {
    if (nd.kv) return true;
    if (!nd.kv_disk) return false;
    auto pl = alloc(false);
    if (!disk_->read(k, false, pl->p, e_.kv_rank_bytes(int64_t(nd.tokens.size())), nullptr)) return false;
    nd.kv = std::move(pl);
    ram_bytes_ += kv_pool_[0]->unit_bytes() * RANKS;
    return true;
}

bool BlockStore::ensure_snap(Node &nd, uint64_t k) {
    if (nd.snap) return true;
    if (!nd.snap_disk) return false;
    auto pl = alloc(true);
    std::vector<float> logits;
    if (!disk_->read(k, true, pl->p, Engine::recurrent_rank_bytes(), &logits)) return false;
    nd.snap = std::move(pl);
    nd.logits = std::move(logits);
    ram_bytes_ += snap_pool_[0]->unit_bytes() * RANKS + nd.logits.size() * 4;
    return true;
}

std::vector<BlockStore::Load::Item> BlockStore::missing(const Hit &h, size_t *bytes) const {
    std::vector<Load::Item> items;
    *bytes = 0;
    for (uint64_t k : path_to(h.node)) {
        const Node &nd = nodes_.at(k);
        if (!nd.kv && nd.kv_disk) {
            const size_t rb = e_.kv_rank_bytes(int64_t(nd.tokens.size()));
            items.push_back({k, false, nullptr, rb, {}});
            *bytes += rb * RANKS;
        }
    }
    const Node &target = nodes_.at(h.node);
    if (!target.snap && target.snap_disk) {
        items.push_back({h.node, true, nullptr, Engine::recurrent_rank_bytes(), {}});
        *bytes += Engine::recurrent_rank_bytes() * RANKS;
    }
    return items;
}

size_t BlockStore::pinned_bytes() const {
    size_t b = 0;
    for (int r = 0; r < RANKS; ++r) b += kv_pool_[size_t(r)]->allocated_bytes() + snap_pool_[size_t(r)]->allocated_bytes();
    return b;
}

bool BlockStore::load(const Hit &h) {
    if (!disk_ || !nodes_.count(h.node)) return false;
    if (load_ && load_->done) finish_load();  // attaches what it read, which may be this hit's
    size_t bytes = 0;
    std::vector<Load::Item> items = missing(h, &bytes);
    if (items.empty()) return false;  // all in RAM: a load running for another hit does not hold it up
    if (load_) return true;           // one load at a time: this hit's starts once that one is attached
    auto ld = std::make_unique<Load>();
    ld->items = std::move(items);
    ld->bytes = bytes;
    ld->t0 = std::chrono::steady_clock::now();
    const size_t nthreads = std::min<size_t>(size_t(load_threads_), ld->items.size());
    log("prefix cache: loading %zu entries (%.1f GB) from disk in the background (%zu threads; RAM cache %.1f GB in "
        "%.1f GB of pinned arenas, %.2f GB of disk writes queued)",
        ld->items.size(), double(bytes) / 1e9, nthreads, double(ram_bytes_) / 1e9, double(pinned_bytes()) / 1e9,
        double(disk_->pending_bytes()) / 1e9);
    // taking the pinned buffers can mean pinning new arenas (seconds per GB when the pools are
    // empty, as after a start, or the host is short of memory), so the load threads take them
    Load *l = ld.get();
    l->running = nthreads;
    for (size_t w = 0; w < nthreads; ++w) l->workers.emplace_back([this, l] { load_worker(l); });
    load_ = std::move(ld);
    return true;
}

void BlockStore::load_worker(Load *l) {
    using clk = std::chrono::steady_clock;
    auto us = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    for (size_t i; (i = l->next++) < l->items.size();) {
        Load::Item &it = l->items[i];
        const auto t0 = clk::now();
        it.pl = alloc(it.snap);
        const auto t1 = clk::now();
        const bool ok = disk_->read(it.key, it.snap, it.pl->p, it.rank_bytes, it.snap ? &it.logits : nullptr);
        const auto t2 = clk::now();
        l->pin_us += us(t0, t1);
        l->read_us += us(t1, t2);
        if (!ok) {
            it.pl.reset();
            ++l->failed;
        }
    }
    if (--l->running == 0) {  // the last one reports
        const double s = std::chrono::duration<double>(clk::now() - l->t0).count();
        log("prefix cache: background load of %zu entries done in %.2f s (%.0f MB/s; thread time %.2f s getting "
            "pinned buffers, %.2f s reading)%s",
            l->items.size(), s, double(l->bytes) / 1e6 / std::max(s, 1e-3), double(l->pin_us) / 1e6,
            double(l->read_us) / 1e6, l->failed ? " (some unreadable)" : "");
        l->done = true;
    }
}

void BlockStore::finish_load() {
    for (auto &t : load_->workers)
        if (t.joinable()) t.join();
    for (auto &it : load_->items) {
        auto nd = nodes_.find(it.key);
        if (!it.pl || nd == nodes_.end()) continue;  // unreadable, or dropped meanwhile
        Node &n = nd->second;
        n.used = ++clock_;
        if (it.snap && !n.snap) {
            n.snap = std::move(it.pl);
            n.logits = std::move(it.logits);
            ram_bytes_ += snap_pool_[0]->unit_bytes() * RANKS + n.logits.size() * 4;
        } else if (!it.snap && !n.kv) {
            n.kv = std::move(it.pl);
            ram_bytes_ += kv_pool_[0]->unit_bytes() * RANKS;
        }
    }
    load_.reset();
    enforce_budgets();
}

bool BlockStore::restore(const Hit &h, const std::vector<int32_t> &prompt, int slot, int64_t keep,
                         std::vector<float> *logits) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto path = path_to(h.node);
    int64_t imported = 0;
    for (uint64_t k : path) {
        Node &nd = nodes_.at(k);
        nd.used = ++clock_;
        // positions < keep - 1 are in the slot already (the MTP layer's KV at
        // keep - 1 depends on the token at keep, which differs)
        if (nd.end() < keep) continue;
        if (!ensure_kv(nd, k)) {
            log("prefix cache: block %016llx is unreadable; dropping it", (unsigned long long)k);
            remove_subtree(k);
            return false;
        }
        e_.import_kv(slot, nd.start, int64_t(nd.tokens.size()), nd.kv->p);
        imported += int64_t(nd.tokens.size());
    }
    Node &target = nodes_.at(h.node);
    if (!ensure_snap(target, h.node)) {
        log("prefix cache: snapshot %016llx is unreadable; dropping it", (unsigned long long)h.node);
        remove_subtree(h.node);
        return false;
    }
    const std::vector<int32_t> tail(prompt.begin() + ptrdiff_t(std::max<int64_t>(0, h.n - 32)),
                                    prompt.begin() + ptrdiff_t(h.n));
    e_.import_recurrent(slot, target.snap->p, h.n, tail);
    if (logits) *logits = target.logits;
    ++stats_.hits;
    stats_.tokens_restored += uint64_t(h.n);
    log("prefix cache: restored %lld tokens into slot %d (%lld imported) in %.3f s", (long long)h.n, slot,
        (long long)imported, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    enforce_budgets();
    return true;
}

void BlockStore::save(int slot, int snap, const std::vector<int32_t> &tokens, const std::vector<float> *logits) {
    static const bool trace = std::getenv("QW_TRACE") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t n = int64_t(tokens.size());
    QW_CHECK(n > 0, "BlockStore::save: no tokens");
    std::vector<uint64_t> path, fresh;
    uint64_t h = ROOT;
    for (int64_t pos = 0; pos < n;) {
        const size_t len = size_t(std::min<int64_t>(BLOCK, n - pos));
        uint64_t k = child(h, tokens.data() + pos, len);
        if (!k) {
            k = key(h, tokens.data() + pos, len);
            if (nodes_.count(k)) {  // a hash collision: keep the stored block
                log("prefix cache: hash collision at position %lld; not saving", (long long)pos);
                return;
            }
            Node &nd = nodes_[k];
            nd.parent = h;
            nd.start = pos;
            nd.tokens.assign(tokens.begin() + ptrdiff_t(pos), tokens.begin() + ptrdiff_t(pos) + ptrdiff_t(len));
            nd.kv = alloc(false);
            e_.export_kv(slot, pos, int64_t(len), nd.kv->p);
            ram_bytes_ += kv_pool_[0]->unit_bytes() * RANKS;
            nodes_.at(h).children.push_back(k);
            fresh.push_back(k);
        }
        path.push_back(k);
        h = k;
        pos += int64_t(len);
    }
    Node &target = nodes_.at(h);
    const bool new_snap = !target.has_snap;
    if (new_snap) {
        target.snap = alloc(true);
        e_.export_recurrent(snap, target.snap->p);
        target.has_snap = true;
        if (logits) target.logits = *logits;
        target.snap_logits = logits != nullptr;
        ram_bytes_ += snap_pool_[0]->unit_bytes() * RANKS + target.logits.size() * 4;
        ++stats_.snapshots_saved;
    }
    const auto t1 = std::chrono::steady_clock::now();
    e_.host_copies_wait();
    if (trace)
        log("prefix cache: saved %lld tokens (%zu new blocks%s): enqueue %.1f ms, copies %.1f ms", (long long)n,
            fresh.size(), new_snap ? ", snapshot" : "", std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
    for (uint64_t k : path) nodes_.at(k).used = ++clock_;
    if (disk_) {
        for (uint64_t k : fresh) {
            Node &nd = nodes_.at(k);
            DiskTier::Meta m;
            m.hash = k;
            m.parent = nd.parent;
            m.tokens = nd.tokens;
            const size_t b = disk_->write(m, {}, nd.kv->p, e_.kv_rank_bytes(int64_t(nd.tokens.size())), nd.kv);
            nd.kv_disk = true;
            nd.disk_bytes += b;
            disk_bytes_ += b;
        }
        if (new_snap) {
            DiskTier::Meta m;
            m.hash = h;
            m.parent = target.parent;
            m.snap = true;
            m.logits = target.snap_logits;
            const size_t b = disk_->write(m, target.logits, target.snap->p, Engine::recurrent_rank_bytes(), target.snap);
            target.snap_disk = true;
            target.disk_bytes += b;
            disk_bytes_ += b;
        }
    }
    enforce_budgets();
}

bool BlockStore::has_snapshot(const std::vector<int32_t> &tokens) const {
    uint64_t h = ROOT;
    for (int64_t pos = 0; pos < int64_t(tokens.size());) {
        const size_t len = size_t(std::min<int64_t>(BLOCK, int64_t(tokens.size()) - pos));
        h = child(h, tokens.data() + pos, len);
        if (!h) return false;
        pos += int64_t(len);
    }
    return h != ROOT && nodes_.at(h).has_snap;
}

// Removes a node and its descendants, then the ancestors left without any
// snapshot below them (their blocks could never be restored).
void BlockStore::remove_subtree(uint64_t k) {
    std::vector<uint64_t> stack{k};
    uint64_t up = nodes_.at(k).parent;
    {  // unlink from the parent
        Node &par = nodes_.at(up);
        par.children.erase(std::remove(par.children.begin(), par.children.end(), k), par.children.end());
    }
    while (!stack.empty()) {
        const uint64_t c = stack.back();
        stack.pop_back();
        Node &nd = nodes_.at(c);
        stack.insert(stack.end(), nd.children.begin(), nd.children.end());
        ram_bytes_ -= ram_size(nd);
        disk_bytes_ -= nd.disk_bytes;
        if (disk_) {
            if (nd.kv_disk) disk_->remove(c, false);
            if (nd.snap_disk) disk_->remove(c, true);
        }
        nodes_.erase(c);
    }
    if (up != ROOT) {
        const Node &par = nodes_.at(up);
        if (par.children.empty() && !par.has_snap) remove_subtree(up);
    }
}

// RAM: the least recently used node whose children hold nothing in RAM gives
// its RAM copies up (kept on disk, or removed without a disk tier). Disk: the
// least recently used leaf is removed. Deeper nodes go first on ties.
void BlockStore::enforce_budgets() {
    auto older = [](const Node &a, const Node &b) {
        return a.used < b.used || (a.used == b.used && a.start > b.start);
    };
    while (ram_bytes_ > ram_budget_) {
        uint64_t victim = 0;
        for (const auto &[k, nd] : nodes_) {
            if (k == ROOT || (!nd.kv && !nd.snap)) continue;
            bool leaf = true;
            for (uint64_t c : nd.children) leaf = leaf && !nodes_.at(c).kv && !nodes_.at(c).snap;
            if (leaf && (!victim || older(nd, nodes_.at(victim)))) victim = k;
        }
        if (!victim) break;
        Node &nd = nodes_.at(victim);
        if (disk_) {
            ram_bytes_ -= ram_size(nd);
            nd.kv.reset();
            nd.snap.reset();
            nd.logits = {};
            nd.logits.shrink_to_fit();
        } else {
            remove_subtree(victim);
        }
    }
    while (disk_ && disk_bytes_ > disk_budget_) {
        uint64_t victim = 0;
        for (const auto &[k, nd] : nodes_)
            if (k != ROOT && nd.children.empty() && (!victim || older(nd, nodes_.at(victim)))) victim = k;
        if (!victim) break;
        remove_subtree(victim);
    }
}

void BlockStore::reserve(size_t blocks, size_t snapshots) {
    for (int r = 0; r < RANKS; ++r) {
        kv_pool_[size_t(r)]->reserve(blocks + KV_LOW_WATER);
        snap_pool_[size_t(r)]->reserve(snapshots + SNAP_LOW_WATER);
    }
}

void BlockStore::flush() {
    if (disk_) disk_->flush();
}

BlockStore::Stats BlockStore::stats() const {
    Stats s = stats_;
    s.ram_bytes = ram_bytes_;
    s.disk_bytes = disk_bytes_;
    for (const auto &[k, nd] : nodes_) {
        if (k == ROOT) continue;
        ++s.blocks;
        s.snapshots += nd.has_snap;
    }
    return s;
}

}  // namespace qw
