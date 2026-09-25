#include "engine/device_mem.hpp"

#include <cstdlib>
#include <mutex>

namespace qw {

namespace {
std::mutex g_mu;
std::vector<GuardedAlloc> g_allocs;
}  // namespace

bool guards_enabled() {
    static const bool on = std::getenv("QW_GUARD") != nullptr;
    return on;
}

void register_guard(const GuardedAlloc &a) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_allocs.push_back(a);
}

int check_guards() {
    if (!guards_enabled()) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    int bad = 0, cur = -1;
    (void)hipGetDevice(&cur);
    std::vector<uint8_t> h(GUARD_BYTES);
    for (const auto &a : g_allocs) {
        CK(hipSetDevice(a.device));
        CK(hipMemcpy(h.data(), static_cast<uint8_t *>(a.ptr) + a.bytes, GUARD_BYTES, hipMemcpyDeviceToHost));
        size_t first = GUARD_BYTES;
        for (size_t i = 0; i < GUARD_BYTES; ++i)
            if (h[i] != GUARD_BYTE) {
                first = i;
                break;
            }
        if (first < GUARD_BYTES) {
            ++bad;
            log("GUARD OVERWRITTEN: device %d, allocation of %zu bytes at %s:%d, first bad byte +%zu past the end",
                a.device, a.bytes, a.file, a.line, first);
        }
    }
    if (cur >= 0) (void)hipSetDevice(cur);
    return bad;
}

}  // namespace qw
