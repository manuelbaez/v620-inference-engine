#include "engine/spill.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "core/common.hpp"
#include "engine/device_mem.hpp"

namespace qw {

namespace {

constexpr int MPOL_DEFAULT = 0, MPOL_PREFERRED = 1, MPOL_BIND = 2, MPOL_INTERLEAVE = 3;

// This thread's policy for the pages it allocates next: `mode` over the nodes in `mask` (bit n: node n); mode
// MPOL_DEFAULT is back to the process's policy (what numactl set, if it did).
void set_policy(int mode, unsigned long mask_bits) {
    unsigned long mask[16] = {};
    mask[0] = mask_bits;
    syscall(SYS_set_mempolicy, mode, mode == MPOL_DEFAULT ? nullptr : mask, mode == MPOL_DEFAULT ? 0 : 64 * 16);
}

// Memory a node can give a bind policy without the OOM killer: free pages plus the page cache it can reclaim (kB -> bytes).
size_t node_available_bytes(int node) {
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo");
    size_t free_kb = 0, inactive_kb = 0, active_kb = 0;
    for (std::string line; std::getline(f, line);) {
        std::sscanf(line.c_str(), "%*s %*d MemFree: %zu kB", &free_kb);
        std::sscanf(line.c_str(), "%*s %*d Inactive(file): %zu kB", &inactive_kb);
        std::sscanf(line.c_str(), "%*s %*d Active(file): %zu kB", &active_kb);
    }
    return (free_kb + inactive_kb + active_kb / 2) * 1024;  // half the active cache: it is not given up easily
}

// The node of the page holding p, or -1 (not populated, or the call failed).
int page_node(void *p) {
    void *pages[1] = {p};
    int status[1] = {-1};
    if (syscall(SYS_move_pages, 0, 1ul, pages, nullptr, status, 0) != 0) return -1;
    return status[0] < 0 ? -1 : status[0];
}

}  // namespace

int device_numa_node(int device) {
    char id[32] = {};
    if (hipDeviceGetPCIBusId(id, sizeof id, device) != hipSuccess) return -1;
    std::string s = id;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    std::ifstream f("/sys/bus/pci/devices/" + s + "/numa_node");
    int node = -1;
    if (!(f >> node)) return -1;
    return node;
}

SpillPool::SpillPool(const std::vector<int> &spill_tokens, int layers, const std::array<int, cfg::RANKS> &devices)
    : layers_(layers) {
    const char *numa_env = std::getenv("QW_SPILL_NUMA");
    const bool numa = !(numa_env && std::atoi(numa_env) == 0);
    const char *nc_env = std::getenv("QW_SPILL_NC");
    unsigned flags = hipHostMallocPortable | hipHostMallocMapped;
    if (nc_env && std::atoi(nc_env) != 0) flags |= hipHostMallocNonCoherent;
    // Without this flag HIP places host memory on the node nearest the current device, whatever the thread's policy.
    if (numa) flags |= hipHostMallocNumaUser;
    std::array<int, cfg::RANKS> node{};
    for (int r = 0; r < cfg::RANKS; ++r) node[size_t(r)] = numa ? device_numa_node(devices[size_t(r)]) : -1;
    // How the pages are placed (QW_SPILL_POLICY):
    //   auto       bind a node's layers to it when it can give that much without the OOM killer (it reclaims page
    //              cache to do so), prefer it otherwise
    //   bind       always bind: the pages come from the node or the allocation fails
    //   preferred  the node first, any other when its free list runs low (the kernel does not reclaim for it)
    //   interleave round-robin over the GPUs' nodes
    //   process    no policy of ours: HIP follows the process's (numactl --interleave=all, --membind=...)
    const char *pol_env = std::getenv("QW_SPILL_POLICY");
    const std::string policy = pol_env ? pol_env : "auto";
    QW_CHECK(policy == "auto" || policy == "bind" || policy == "preferred" || policy == "interleave" || policy == "process",
             "QW_SPILL_POLICY must be auto, bind, preferred, interleave or process");
    unsigned long all_nodes = 0;
    for (int r = 0; r < cfg::RANKS; ++r)
        if (node[size_t(r)] >= 0) all_nodes |= 1ul << node[size_t(r)];
    std::map<int, size_t> need_on;  // what the layers meant for each node add up to
    for (size_t si = 0; si < spill_tokens.size(); ++si)
        for (int g = 0; g < cfg::KV_HEADS; ++g)
            for (int l = 0; l < layers; ++l)
                if (spill_tokens[si] > 0)
                    need_on[node[size_t(g * cfg::KV_REPLICAS + l % cfg::KV_REPLICAS)]] += size_t(spill_tokens[si]) * token_bytes();
    std::map<int, bool> bind_node;
    for (const auto &[nd, need] : need_on)
        bind_node[nd] = nd >= 0 && (policy == "bind" || (policy == "auto" && node_available_bytes(nd) >= need + (size_t(4) << 30)));

    const auto t0 = std::chrono::steady_clock::now();
    std::map<int, size_t> bytes_on;             // by the node the pages were meant for
    size_t sampled = 0, on_target = 0;          // pages whose node was read back
    std::map<int, size_t> seen;                 // ... and where they were
    bufs_.resize(spill_tokens.size());
    for (size_t si = 0; si < spill_tokens.size(); ++si) {
        const size_t n = size_t(std::max(0, spill_tokens[si]));
        if (n == 0) continue;
        bufs_[si].resize(size_t(cfg::KV_HEADS) * size_t(layers));
        for (int g = 0; g < cfg::KV_HEADS; ++g)
            for (int l = 0; l < layers; ++l) {
                // the layer's pages go near one of the group's ranks, alternating
                const int owner = g * cfg::KV_REPLICAS + l % cfg::KV_REPLICAS;
                const int target = node[size_t(owner)];
                const size_t total = n * token_bytes();
                if (target < 0 || policy == "process") set_policy(MPOL_DEFAULT, 0);
                else if (policy == "interleave") set_policy(MPOL_INTERLEAVE, all_nodes);
                else set_policy(bind_node[target] ? MPOL_BIND : MPOL_PREFERRED, 1ul << target);
                void *p = nullptr;
                const hipError_t err = hipHostMalloc(&p, total, flags);
                set_policy(MPOL_DEFAULT, 0);
                if (err != hipSuccess) fail(std::string("KV spill: hipHostMalloc ") + std::to_string(total >> 20) + " MB: " + hipGetErrorString(err));
                allocs_.push_back(p);
                bytes_ += total;
                bytes_on[target] += total;
                uint8_t *b = static_cast<uint8_t *>(p);
                Buf &buf = bufs_[si][size_t(g * layers + l)];
                buf.K = reinterpret_cast<uint16_t *>(b);
                buf.V = reinterpret_cast<uint16_t *>(b + n * 512);
                if (target >= 0)
                    for (size_t off : {size_t(0), total / 3, total / 3 * 2, total - 4096}) {
                        const int got = page_node(b + off / 4096 * 4096);
                        if (got >= 0) {
                            ++sampled;
                            ++seen[got];
                            on_target += got == target;
                        }
                    }
            }
    }
    if (bytes_ == 0) return;
    // each rank sees the same bytes at the same address (unified addressing); the kernels rely on it
    for (int r = 0; r < cfg::RANKS; ++r) {
        CK(hipSetDevice(devices[size_t(r)]));
        for (void *p : allocs_) {
            void *d = nullptr;
            CK(hipHostGetDevicePointer(&d, p, 0));
            QW_CHECK(d == p, "KV spill: the device address of a mapped host buffer differs from its host address");
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string where;
    for (const auto &[nd, b] : bytes_on) where += " node " + std::to_string(nd) + ": " + std::to_string(b >> 20) + " MB;";
    std::string got;
    for (const auto &[nd, c] : seen) got += " node " + std::to_string(nd) + ": " + std::to_string(100 * c / sampled) + "%;";
    std::string binds;
    for (const auto &[nd, b] : bind_node) binds += " node " + std::to_string(nd) + (b ? " bound;" : " preferred;");
    log("KV spill: %.2f GB pinned in %.1f s (%s, policy %s:%s); intended%s", double(bytes_) / 1e9, secs,
        (flags & hipHostMallocNonCoherent) ? "non-coherent" : "coherent", policy.c_str(),
        policy == "auto" || policy == "bind" || policy == "preferred" ? binds.c_str() : " n/a", where.c_str());
    if (sampled)
        log("KV spill: %.1f%% of the sampled pages are on the intended node; sampled pages by node:%s",
            100.0 * double(on_target) / double(sampled), got.c_str());
    else
        log("KV spill: NUMA placement off");
}

SpillPool::~SpillPool() {
    for (void *p : allocs_) hipHostFree(p);
}

const SpillPool::Buf &SpillPool::buf(int slot, int rank, int layer) const {
    static const Buf none;
    if (size_t(slot) >= bufs_.size() || bufs_[size_t(slot)].empty()) return none;
    const int g = rank / cfg::KV_REPLICAS;
    return bufs_[size_t(slot)][size_t(g * layers_ + layer)];
}

}  // namespace qw
